import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import time
import cv2
import subprocess
import threading
import re
import csv
import json
from collections import deque
from ultralytics import YOLO

import os
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR  = os.environ.get("RESULT_DIR", ".")

# ═══════════════════════════════════════════════════════
# ▶ 필터링 파라미터  ← 여기만 바꾸면 됩니다
# ═══════════════════════════════════════════════════════
CONF_THRESHOLD      = 0.45    # 이 값 미만 bbox 무시
TARGET_CLASSES      = [0]     # 추적할 YOLO class id 목록  (비워두면 전체 허용)
                               #   예) COCO: 0=person, 2=car
N_CONFIRM_FRAMES    = 3       # IDLE → TRACKING 진입에 필요한 연속 탐지 프레임 수
SMOOTH_ALPHA        = 0.4     # bbox 중심 EMA 계수 (0<α≤1, 작을수록 더 부드러움)

# bbox 면적 기반 거리 정책
# 탐지된 bbox 면적(픽셀²)에 따라 z 고도를 조정합니다.
# 면적이 크면(가까움) 고도를 높이고, 작으면(멀리) 낮춥니다.
AREA_ZONES = [
    # (면적 상한,  목표 z [NED, 음수=위])
    (5_000,   -2.5),   # 매우 작은 bbox → 더 낮게 내려가서 접근
    (20_000,  -1.5),   # 중간 → 기본 고도
    (60_000,  -0.8),   # 큰 bbox (가까움) → 살짝 올라가서 거리 확보
    (float("inf"), -0.5),  # 매우 큰 bbox → 최대한 올라가기
]
# ═══════════════════════════════════════════════════════


# -------------------------------------------------------
# tegrastats 모니터
# -------------------------------------------------------
class TegraMonitor:
    def __init__(self):
        self.data    = {"gpu": 0, "cpu": 0, "power": 0, "ram": 0}
        self.running = True

    def start(self):
        def run():
            p = subprocess.Popen(["tegrastats"], stdout=subprocess.PIPE, text=True)
            while self.running:
                line = p.stdout.readline()
                m = re.search(r"GR3D_FREQ (\d+)%", line)
                if m: self.data["gpu"] = int(m.group(1))
                m = re.search(r"CPU \[(.*?)\]", line)
                if m:
                    vals = [int(v.split('%')[0]) for v in m.group(1).split(',')]
                    self.data["cpu"] = sum(vals) / len(vals)
                m = re.search(r"RAM (\d+)/", line)
                if m: self.data["ram"] = int(m.group(1))
                m = re.search(r"VDD_IN (\d+)mW", line)
                if m: self.data["power"] = int(m.group(1))
        threading.Thread(target=run, daemon=True).start()

    def stop(self):
        self.running = False


# -------------------------------------------------------
# EMA 스무더
# -------------------------------------------------------
class EMASmoother:
    """지수이동평균으로 bbox 중심 좌표 jitter 억제"""
    def __init__(self, alpha: float):
        self.alpha = alpha
        self._cx   = None
        self._cy   = None

    def update(self, cx: float, cy: float):
        if self._cx is None:
            self._cx, self._cy = cx, cy
        else:
            self._cx = self.alpha * cx + (1 - self.alpha) * self._cx
            self._cy = self.alpha * cy + (1 - self.alpha) * self._cy
        return self._cx, self._cy

    def reset(self):
        self._cx = self._cy = None


# -------------------------------------------------------
# YOLO BBox Node  (TensorRT FP16)
# -------------------------------------------------------
class YoloBboxNode(Node):
    def __init__(self):
        super().__init__('yolo_bbox_node')

        self.sub = self.create_subscription(
            Image, '/camera', self.callback, 10
        )
        self.roi_pub  = self.create_publisher(Image,  '/target_roi',  10)
        self.bbox_pub = self.create_publisher(String, '/target_bbox', 10)

        self.bridge  = CvBridge()
        self.smoother = EMASmoother(alpha=SMOOTH_ALPHA)

        # N프레임 확인용 링버퍼  (True = 해당 프레임에 유효 탐지 있음)
        self._confirm_buf: deque[bool] = deque(maxlen=N_CONFIRM_FRAMES)
        self._triggered = False  # 한 번 trigger 되면 True → confirm 불필요

        self.get_logger().info("Loading TensorRT FP16 model...")
        self.model = YOLO(os.path.join(BASE_DIR, "../models/yolo26n.engine"))
        self.get_logger().info("Model loaded.")

        self.monitor = TegraMonitor()
        self.monitor.start()

        log_path = os.path.join(RESULT_DIR, "yolo_log.csv")
        self.csv    = open(log_path, "w", newline="")
        self.writer = csv.writer(self.csv)
        self.writer.writerow([
            "timestamp",
            "ros_delay_ms",
            "yolo_infer_ms",
            "node_latency_ms",
            "roi_kb",
            "detections_raw",   # conf/class 필터 전
            "detections_filt",  # 필터 후
            "triggered",        # N-frame confirm 통과 여부
            "smooth_cx", "smooth_cy",
            "gpu_%", "cpu_%", "power_mW", "ram_MB"
        ])
        self.get_logger().info(f"Logging to {log_path}")

    # ──────────────────────────────────────────
    def callback(self, msg):
        t0 = time.perf_counter()

        # ROS end-to-end delay
        now  = self.get_clock().now()
        sent = rclpy.time.Time.from_msg(msg.header.stamp)
        ros_delay = (now - sent).nanoseconds / 1e6

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # ── YOLO 추론 ──
        t_inf = time.perf_counter()
        results = self.model(frame, verbose=False)
        yolo_ms = (time.perf_counter() - t_inf) * 1000

        # ── 1단계: confidence + class 필터 ──
        raw_dets  = []
        filt_dets = []
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                conf = float(box.conf[0])
                cls  = int(box.cls[0])
                raw_dets.append({"cls": cls, "conf": conf, "bbox": [x1, y1, x2, y2]})

                if conf < CONF_THRESHOLD:
                    continue
                if TARGET_CLASSES and cls not in TARGET_CLASSES:
                    continue
                filt_dets.append({
                    "cls":  cls,
                    "conf": round(conf, 3),
                    "bbox": [x1, y1, x2, y2]
                })

        has_valid = len(filt_dets) > 0

        # ── 2단계: N-frame debounce (IDLE→TRACKING 진입 조건) ──
        self._confirm_buf.append(has_valid)
        if not self._triggered:
            # 링버퍼가 가득 차고 모두 True일 때만 trigger
            all_confirmed = (
                len(self._confirm_buf) == N_CONFIRM_FRAMES
                and all(self._confirm_buf)
            )
        else:
            all_confirmed = has_valid  # 이미 trigger됨 → 단일 프레임으로 유지

        roi_kb     = 0.0
        smooth_cx  = smooth_cy = None

        if all_confirmed and filt_dets:
            self._triggered = True

            # ── 3단계: 최고 confidence 선택 ──
            best = max(filt_dets, key=lambda d: d["conf"])
            x1, y1, x2, y2 = best["bbox"]

            # ── 4단계: EMA smoothing ──
            raw_cx = (x1 + x2) / 2.0
            raw_cy = (y1 + y2) / 2.0
            smooth_cx, smooth_cy = self.smoother.update(raw_cx, raw_cy)

            # ── ROI 크롭 & 발행 ──
            h, w = frame.shape[:2]
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w, x2), min(h, y2)
            roi = frame[y1c:y2c, x1c:x2c]

            if roi.size > 0:
                roi_msg = self.bridge.cv2_to_imgmsg(roi, encoding='bgr8')
                roi_msg.header.stamp = msg.header.stamp
                self.roi_pub.publish(roi_msg)
                _, buf = cv2.imencode('.jpg', roi, [cv2.IMWRITE_JPEG_QUALITY, 80])
                roi_kb = len(buf) / 1024

            # ── bbox 면적 → 거리 정책 메타 추가 ──
            area = (x2 - x1) * (y2 - y1)
            best["area"]       = area
            best["smooth_cx"]  = round(smooth_cx, 1)
            best["smooth_cy"]  = round(smooth_cy, 1)

            # ── BBox JSON 발행 (전체 필터링 통과 목록) ──
            bbox_msg = String()
            bbox_msg.data = json.dumps(filt_dets)
            self.bbox_pub.publish(bbox_msg)

        elif not has_valid:
            # 탐지 소실 시 smoother·trigger 초기화
            self.smoother.reset()
            if not has_valid and self._triggered:
                self._triggered = False
                self._confirm_buf.clear()

        node_ms = (time.perf_counter() - t0) * 1000
        stats   = self.monitor.data
        self.writer.writerow([
            time.time(),
            round(ros_delay, 2),
            round(yolo_ms,   2),
            round(node_ms,   2),
            round(roi_kb,    1),
            len(raw_dets),
            len(filt_dets),
            int(all_confirmed),
            round(smooth_cx, 1) if smooth_cx else "",
            round(smooth_cy, 1) if smooth_cy else "",
            stats["gpu"],
            round(stats["cpu"], 1),
            stats["power"],
            stats["ram"],
        ])

        print(
            f"ROS={ros_delay:.1f}ms | YOLO={yolo_ms:.1f}ms | total={node_ms:.1f}ms | "
            f"raw={len(raw_dets)} filt={len(filt_dets)} trig={int(all_confirmed)} | "
            f"ROI={roi_kb:.1f}KB | GPU={stats['gpu']}% | pwr={stats['power']}mW"
        )

    def destroy_node(self):
        self.monitor.stop()
        self.csv.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = YoloBboxNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()