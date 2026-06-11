"""
fsm_node.py
===========
yolo_node.py 의 최소 수정판.
YOLO 추론 대신 rolling-background FSM (fsm_count_spe 로직 이식) 을 실행.

변경 사항 (yolo_node.py 대비):
  - YOLO / CvBridge / ROI publish 제거
  - /camera(Image) → /ksem_count(Float32MultiArray) 구독
  - 추론 → FSM 상태 계산 (NOMINAL / PRE_ALERT / ALERT)
  - /target_bbox(String) 발행 포맷 유지 → offboard_node 무수정 연동
    {"fsm_state": "ALERT", "channel": "PD3A-OU", "count": 1234.5,
     "threshold": 456.7, "conf": 1.0}  ← detections 리스트 1개 원소로 래핑
  - TegraMonitor / log.csv 구조 동일 유지 (yolo_infer_ms → fsm_infer_ms)

FSM 파라미터:
  BG_WINDOW_DAYS  : rolling 배경 윈도우
  K               : median + K·σ 배수
  ONSET_FLOOR     : 임계 하한 (proton=0.5, electron=0.0)
  PEAK_FLOOR      : 사후 필터 (0.0 = 비활성)
  MIN_DURATION_H  : 최소 지속 시간
"""

from __future__ import annotations
import json
import os
import csv
import time
import subprocess
import threading
import re
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

RESULT_DIR = os.environ.get("RESULT_DIR", ".")

# ══════════════════════════════════════════════════════
# FSM 파라미터 블록
# ══════════════════════════════════════════════════════
BG_WINDOW_DAYS = 30       # 배경 윈도우 (포인트 수 = WINDOW_DAYS * 96 at 15min)
K              = 10       # proton; 전자 채널 테스트 시 20 으로 변경
ONSET_FLOOR    = 0.5      # proton=0.5, electron=0.0
PEAK_FLOOR     = 2.0      # 0.0 = 비활성
MIN_DURATION_H = 1.0      # 최소 이벤트 지속 시간
# ══════════════════════════════════════════════════════

CADENCE_MIN   = 15
PTS_PER_DAY   = 24 * 60 // CADENCE_MIN          # 96
BG_WINDOW_PTS = BG_WINDOW_DAYS * PTS_PER_DAY    # 2880


# ── FSM 상태 ──────────────────────────────────────────
STATE_NOMINAL   = "NOMINAL"
STATE_PRE_ALERT = "PRE_ALERT"
STATE_ALERT     = "ALERT"

# ── offboard_node 가 기대하는 "탐지 있음" 조건 ─────────
# detections 리스트가 비어있지 않으면 TRACKING 진입
# NOMINAL → 빈 리스트, PRE_ALERT/ALERT → 원소 1개


def _mad_sigma(arr: np.ndarray) -> float:
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return np.nan
    med = np.median(arr)
    return float(1.4826 * np.median(np.abs(arr - med)))


class ChannelFSM:
    """단일 채널 FSM. 새 count 포인트 하나씩 push."""

    def __init__(self, name: str):
        self.name    = name
        self.buf     = deque(maxlen=BG_WINDOW_PTS)   # rolling 배경 버퍼
        self.state   = STATE_NOMINAL
        self.onset_pts = 0          # 연속 초과 포인트 수
        self.alert_start: float | None = None

    def push(self, count: float) -> tuple[str, float, float]:
        """count 1포인트 입력 → (state, threshold, bg_median) 반환."""
        self.buf.append(count if np.isfinite(count) else 0.0)

        arr = np.array(self.buf)
        bg_median = float(np.median(arr))
        sigma     = _mad_sigma(arr)
        sigma     = sigma if (np.isfinite(sigma) and sigma > 0) else 0.0

        threshold = bg_median + K * sigma
        if ONSET_FLOOR > 0:
            threshold = max(threshold, ONSET_FLOOR)

        above = count >= threshold

        # 상태 전이
        if above:
            self.onset_pts += 1
        else:
            self.onset_pts = 0

        min_pts = int(MIN_DURATION_H * 60 / CADENCE_MIN)   # 1h = 4pts

        if self.state == STATE_NOMINAL:
            if self.onset_pts >= min_pts:
                self.state = STATE_ALERT
                self.alert_start = time.time()
            elif self.onset_pts >= 1:
                self.state = STATE_PRE_ALERT
        elif self.state == STATE_PRE_ALERT:
            if self.onset_pts >= min_pts:
                self.state = STATE_ALERT
            elif self.onset_pts == 0:
                self.state = STATE_NOMINAL
        elif self.state == STATE_ALERT:
            if self.onset_pts == 0:
                self.state = STATE_NOMINAL
                self.alert_start = None

        return self.state, threshold, bg_median


# ── TegraMonitor (yolo_node 에서 그대로 복사) ──────────
class TegraMonitor:
    def __init__(self):
        self.data    = {"gpu": 0, "cpu": 0, "power": 0, "ram": 0}
        self.running = True

    def start(self):
        def run():
            try:
                p = subprocess.Popen(["tegrastats"], stdout=subprocess.PIPE, text=True)
                while self.running:
                    line = p.stdout.readline()
                    gpu = re.search(r"GR3D_FREQ (\d+)%", line)
                    if gpu:
                        self.data["gpu"] = int(gpu.group(1))
                    cpu = re.search(r"CPU \[(.*?)\]", line)
                    if cpu:
                        vals = [int(v.split("%")[0]) for v in cpu.group(1).split(",")]
                        self.data["cpu"] = sum(vals) / len(vals)
                    ram = re.search(r"RAM (\d+)/", line)
                    if ram:
                        self.data["ram"] = int(ram.group(1))
                    power = re.search(r"VDD_IN (\d+)mW", line)
                    if power:
                        self.data["power"] = int(power.group(1))
            except FileNotFoundError:
                pass   # tegrastats 없는 환경(데스크탑)에서도 동작

        threading.Thread(target=run, daemon=True).start()

    def stop(self):
        self.running = False


class FsmNode(Node):
    def __init__(self):
        super().__init__("fsm_node")

        # ── 구독/발행 (/target_bbox 포맷 유지 → offboard_node 무수정) ──
        self.sub      = self.create_subscription(
            Float32MultiArray, "/ksem_count", self.callback, 10
        )
        self.bbox_pub = self.create_publisher(String, "/target_bbox", 10)

        self.channel_fsms: dict[str, ChannelFSM] = {}   # 첫 메시지에서 초기화
        self.channel_names: list[str] = []

        self.monitor = TegraMonitor()
        self.monitor.start()

        # ── log.csv (yolo_node 와 동일 컬럼, yolo_infer_ms → fsm_infer_ms) ──
        log_path    = os.path.join(RESULT_DIR, "log.csv")
        self.csv    = open(log_path, "w", newline="")
        self.writer = csv.writer(self.csv)
        self.writer.writerow([
            "timestamp", "ros_delay_ms", "fsm_infer_ms", "node_latency_ms",
            "fsm_state", "channel", "threshold",
            "gpu_%", "cpu_%", "power_mW", "ram_MB"
        ])

        # ── detection_log.csv ──
        det_path         = os.path.join(RESULT_DIR, "detection_log.csv")
        self.det_csv     = open(det_path, "w", newline="")
        self.det_writer  = csv.writer(self.det_csv)
        self.det_writer.writerow([
            "timestamp", "event", "channel", "fsm_state", "count", "threshold"
        ])

        self.prev_states: dict[str, str] = {}
        self.get_logger().info("FsmNode ready. Waiting for /ksem_count ...")

    # ────────────────────────────────────────────────
    def callback(self, msg: Float32MultiArray):
        t0 = time.perf_counter()

        # ROS delay: data_offset(물리 시각 unix sec) vs 현재 시각
        phys_ts   = msg.layout.data_offset   # unix epoch sec (ksem_node 에서 삽입)
        ros_delay = (time.time() - phys_ts) * 1000 if phys_ts else 0.0

        # 채널명 파싱 (첫 메시지에서 FSM 초기화)
        if msg.layout.dim:
            names = msg.layout.dim[0].label.split(",")
            if names != self.channel_names:
                self.channel_names = names
                for n in names:
                    if n not in self.channel_fsms:
                        self.channel_fsms[n] = ChannelFSM(n)
                self.get_logger().info(f"Channels: {self.channel_names}")

        if not self.channel_names:
            return

        values = list(msg.data)

        # ── FSM 추론 ──────────────────────────────────
        t_fsm = time.perf_counter()
        results: list[dict] = []
        worst_state = STATE_NOMINAL

        for name, count in zip(self.channel_names, values):
            fsm = self.channel_fsms[name]
            state, threshold, bg_median = fsm.push(count)

            results.append({
                "channel":   name,
                "fsm_state": state,
                "count":     round(float(count), 3),
                "threshold": round(threshold, 4),
                "bg_median": round(bg_median, 4),
                "conf":      1.0 if state == STATE_ALERT else
                             0.6 if state == STATE_PRE_ALERT else 0.0,
            })

            # 가장 심각한 상태 추적
            order = {STATE_NOMINAL: 0, STATE_PRE_ALERT: 1, STATE_ALERT: 2}
            if order[state] > order[worst_state]:
                worst_state = state

            # 이벤트 로깅 (상태 변화 시)
            prev = self.prev_states.get(name, STATE_NOMINAL)
            if state != prev:
                event = f"{prev}→{state}"
                self.det_writer.writerow([
                    time.time(), event, name, state, round(float(count), 3),
                    round(threshold, 4)
                ])
                self.det_csv.flush()
                self.get_logger().info(f"[FSM] {name}: {event}  count={count:.2f}  thr={threshold:.4f}")
            self.prev_states[name] = state

        fsm_infer_ms = (time.perf_counter() - t_fsm) * 1000

        # ── offboard_node 로 발행 ─────────────────────
        # NOMINAL → 빈 리스트 (IDLE 유지)
        # PRE_ALERT / ALERT → 리스트 1개 원소 (TRACKING 진입)
        alert_results = [r for r in results if r["fsm_state"] != STATE_NOMINAL]
        bbox_msg      = String()
        bbox_msg.data = json.dumps(alert_results)
        self.bbox_pub.publish(bbox_msg)

        node_latency_ms = (time.perf_counter() - t0) * 1000
        stats = self.monitor.data

        self.writer.writerow([
            time.time(),
            round(ros_delay, 2),
            round(fsm_infer_ms, 3),
            round(node_latency_ms, 2),
            worst_state,
            self.channel_names[0] if self.channel_names else "",
            round(results[0]["threshold"], 4) if results else "",
            stats["gpu"],
            round(stats["cpu"], 1),
            stats["power"],
            stats["ram"],
        ])
        self.csv.flush()

        self.get_logger().info(
            f"ROS_delay={ros_delay:.0f}ms | "
            f"FSM={fsm_infer_ms:.3f}ms | "
            f"total={node_latency_ms:.1f}ms | "
            f"worst={worst_state} | "
            f"alerts={len(alert_results)}"
        )

    def destroy_node(self):
        self.monitor.stop()
        self.csv.close()
        self.det_csv.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = FsmNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
