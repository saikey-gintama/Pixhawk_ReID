"""
fsm_node.py  —  cFS LC_action (AP 층)
=====================================
ksem_node(WP)가 보낸 채널별 판정(/wp_results)을 받아
**RPN voting + persistence** 로 단일 운용 상태를 결정하고 alert 를 발행한다.

cFS 매핑: LC_action (Actionpoint).
  - LC_EvaluateRPN    → evaluate_rpn()  (4채널 OR, 3-값 논리 STALE/ERROR 전파)
  - ConsecutiveFailCount + MaxFailsBeforeRTS → persistence
  - 발동 후 PASSIVE 전환 → 재발사 방지
  - LC_STATE_ACTIVE/PASSIVE → dry-run 스위치 (판정만 보고 액추에이션 막기)

임계 계산은 하지 않는다 — 그건 WP(ksem_node) 책임. 여기선 채널 결과 조합만.

구독: /wp_results (ksem_node)
발행: /sep_alert (String JSON):
    {"ts": <unix_sec>, "ap_state": "NOMINAL"/"PRE_ALERT"/"ALERT",
     "fail_count": 2, "trigger_channels": ["PD3A-OU", ...],
     "rpn_result": "TRIGGERED"/"NORMAL"/"STALE"}

파라미터(상단 블록):
  PERSISTENCE_N   : 연속 N슬롯 TRIGGERED 여야 ALERT (N=2 or 3)
  AP_ACTIVE       : True=실제 발동, False=dry-run(로그만)
  RPN_EQUATION    : voting 논리식 (후위표기)
"""

from __future__ import annotations
import os
import csv
import json
import time
import argparse
import subprocess
import threading
import re

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

RESULT_DIR = os.environ.get("RESULT_DIR", ".")

# ══════════════════════════════════════════════════════
# AP 파라미터 블록 — 기본값(argparse 로 런타임 덮어쓰기, cFS TBL 철학)
#   변경 가능 인자: --persistence --dry-run / --active
# ══════════════════════════════════════════════════════
PERSISTENCE_N = 3        # 연속 N슬롯 TRIGGERED → ALERT (N=2:30분 / N=3:1h @15min)
AP_ACTIVE     = True     # True=발동, False=dry-run (판정만, 액추에이션 막음)

# HK 로그 provenance — 이 런이 어떤 window/k 로 돌았는지 기록만 한다(판정 미사용).
# ksem_node 가 실제로 window/k 를 적용하고, fsm 은 동일 값을 --window/--k 로 받아
# 조인 테이블에 함께 남긴다. W=30 vs W=7 비교가 로그만 봐도 자기설명되게.
BG_WINDOW_DAYS_PROV = 30
K_PROV              = 10

# RPN voting 논리식 (후위표기, LC_EvaluateRPN 형식)
# 4채널 OR: (((PD3A-OU ∨ PD3A-OUT) ∨ PD1A-OUT) ∨ PD2A-OU)
RPN_EQUATION = [
    "PD3A-OU", "PD3A-OUT", "OR",
    "PD1A-OUT", "OR",
    "PD2A-OU", "OR",
    "EQUAL",
]
# ══════════════════════════════════════════════════════

# WP 판정값 (ksem_node 와 일치)
WATCH_TRUE  = "TRUE"
WATCH_FALSE = "FALSE"
WATCH_STALE = "STALE"
WATCH_ERROR = "ERROR"

# RPN 평가 결과 (LC_ACTION_* 대응)
RPN_TRIGGERED = "TRIGGERED"   # LC_ACTION_FAIL  (수식 TRUE = 위반 = 발동조건)
RPN_NORMAL    = "NORMAL"      # LC_ACTION_PASS
RPN_STALE     = "STALE"
RPN_ERROR     = "ERROR"

# AP 운용 상태
STATE_NOMINAL   = "NOMINAL"
STATE_PRE_ALERT = "PRE_ALERT"
STATE_ALERT     = "ALERT"


def evaluate_rpn(equation: list[str], watch: dict[str, str]) -> tuple[str, list[str]]:
    """
    RPN voting 평가 (LC_EvaluateRPN 이식).
    equation: 후위표기 토큰 리스트 (채널명 / "AND" "OR" "XOR" "NOT" / "EQUAL")
    watch   : {채널명: "TRUE"/"FALSE"/"STALE"/"ERROR"}

    3-값 논리: FALSE 우선, 그 다음 ERROR, 그 다음 STALE, 나머지 TRUE.
    (lc_action.c 의 AND/OR 우선순위 규칙 그대로)

    반환: (RPN_TRIGGERED/NORMAL/STALE/ERROR, 기여한 TRUE 채널 목록)
    """
    stack: list[str] = []
    trigger_channels: list[str] = []

    def combine_or(a: str, b: str) -> str:
        # OR: TRUE 하나라도 있으면 TRUE. 아니면 ERROR>STALE>FALSE 순 전파.
        if a == WATCH_TRUE or b == WATCH_TRUE:
            return WATCH_TRUE
        if WATCH_ERROR in (a, b):
            return WATCH_ERROR
        if WATCH_STALE in (a, b):
            return WATCH_STALE
        return WATCH_FALSE

    def combine_and(a: str, b: str) -> str:
        # AND: FALSE 하나라도 있으면 FALSE. 아니면 ERROR>STALE>TRUE.
        if a == WATCH_FALSE or b == WATCH_FALSE:
            return WATCH_FALSE
        if WATCH_ERROR in (a, b):
            return WATCH_ERROR
        if WATCH_STALE in (a, b):
            return WATCH_STALE
        return WATCH_TRUE

    def combine_xor(a: str, b: str) -> str:
        if WATCH_ERROR in (a, b):
            return WATCH_ERROR
        if WATCH_STALE in (a, b):
            return WATCH_STALE
        return WATCH_TRUE if (a != b) else WATCH_FALSE

    def negate(a: str) -> str:
        if a == WATCH_TRUE:
            return WATCH_FALSE
        if a == WATCH_FALSE:
            return WATCH_TRUE
        return a   # STALE/ERROR 그대로

    for tok in equation:
        if tok == "EQUAL":
            break
        elif tok == "AND":
            b = stack.pop(); a = stack.pop(); stack.append(combine_and(a, b))
        elif tok == "OR":
            b = stack.pop(); a = stack.pop(); stack.append(combine_or(a, b))
        elif tok == "XOR":
            b = stack.pop(); a = stack.pop(); stack.append(combine_xor(a, b))
        elif tok == "NOT":
            a = stack.pop(); stack.append(negate(a))
        else:
            # 채널명 → WP 결과를 스택에 push
            val = watch.get(tok, WATCH_STALE)   # 결과 없으면 STALE
            if val == WATCH_TRUE:
                trigger_channels.append(tok)
            stack.append(val)

    if len(stack) != 1:
        return RPN_ERROR, []   # 수식 오류 (IllegalRPN)

    final = stack[0]
    mapping = {
        WATCH_TRUE:  RPN_TRIGGERED,   # 수식 참 = 위반 = 발동
        WATCH_FALSE: RPN_NORMAL,
        WATCH_STALE: RPN_STALE,
        WATCH_ERROR: RPN_ERROR,
    }
    return mapping[final], trigger_channels


# ── TegraMonitor (리소스 측정용) ──────────────────────
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
                pass   # 데스크탑 환경

        threading.Thread(target=run, daemon=True).start()

    def stop(self):
        self.running = False


# ──────────────────────────────────────────────────────
# FsmNode (AP)
# ──────────────────────────────────────────────────────
class FsmNode(Node):
    def __init__(self):
        super().__init__("fsm_node")

        self.sub       = self.create_subscription(
            String, "/wp_results", self.callback, 10
        )
        self.alert_pub = self.create_publisher(String, "/sep_alert", 10)

        # persistence 상태 (LC_action 의 단일 AP)
        self.consecutive_fail = 0          # ConsecutiveFailCount
        self.cumulative_fail  = 0          # CumulativeFailCount (통계)
        self.ap_state         = STATE_NOMINAL
        self.latched          = False      # 발동 후 PASSIVE (재발사 방지)

        self.monitor = TegraMonitor()
        self.monitor.start()

        # HK telemetry 로그 (cFS Housekeeping packet 의 ROS2 구현)
        # latency 분해: wp_eval(WP) → dds_transport(SB) → fsm_infer(AP) → node 전체
        log_path    = os.path.join(RESULT_DIR, "log.csv")
        self.csv    = open(log_path, "w", newline="")
        self.writer = csv.writer(self.csv)
        self.writer.writerow([
            "timestamp", "window_d", "k",
            "wp_eval_ms", "dds_transport_ms", "fsm_infer_ms", "node_latency_ms",
            "ap_state", "rpn_result", "fail_count", "trigger_channels",
            "gpu_%", "cpu_%", "power_mW", "ram_MB"
        ])

        det_path        = os.path.join(RESULT_DIR, "ap_event_log.csv")
        self.det_csv    = open(det_path, "w", newline="")
        self.det_writer = csv.writer(self.det_csv)
        self.det_writer.writerow([
            "timestamp", "event", "ap_state", "rpn_result", "fail_count", "trigger_channels"
        ])

        self.prev_state = STATE_NOMINAL
        self.get_logger().info(
            f"FsmNode (AP) ready. N={PERSISTENCE_N} active={AP_ACTIVE}. "
            f"Waiting for /wp_results ..."
        )

    def callback(self, msg: String):
        t0 = time.perf_counter()
        t0_wall = time.time()   # fsm 수신 시각 (transport 측정 기준점)

        payload = json.loads(msg.data)
        phys_ts = payload.get("ts", 0.0)          # 물리 관측시각 — 판정 전용
        results = payload.get("results", [])

        # ── transport 측정 (telemetry 전용, 판정에 절대 미사용) ──
        # ksem_pub_ts: ksem_node 발행시각(wall-clock). dds_transport = 수신 - 발행.
        # phys_ts(2019~2024 물리시각)로는 transport 를 잴 수 없으므로 분리한다.
        ksem_pub_ts = payload.get("ksem_pub_ts", None)
        dds_transport_ms = (
            (t0_wall - ksem_pub_ts) * 1000.0 if ksem_pub_ts else float("nan")
        )
        wp_eval_ms = payload.get("wp_eval_ms", float("nan"))   # WP 측 비용 (pass-through)

        # 채널명 → watch 결과 dict
        watch = {r["channel"]: r["watch"] for r in results}

        # ── RPN voting 평가 ──
        t_fsm = time.perf_counter()
        rpn_result, trigger_channels = evaluate_rpn(RPN_EQUATION, watch)

        # ── persistence (LC_SampleSingleAP 이식) ──
        if rpn_result == RPN_TRIGGERED:
            self.consecutive_fail += 1
            self.cumulative_fail  += 1

            if self.consecutive_fail >= PERSISTENCE_N:
                # 발동 조건 충족
                if not self.latched:
                    if AP_ACTIVE:
                        self.ap_state = STATE_ALERT
                        self.latched  = True        # 재발사 방지
                    else:
                        # dry-run: 판정은 ALERT 로 보이되 latch 안 함(로그만)
                        self.ap_state = STATE_ALERT
                # 이미 latched 면 ALERT 유지
            else:
                # N 미만 → PRE_ALERT
                if self.ap_state == STATE_NOMINAL:
                    self.ap_state = STATE_PRE_ALERT

        elif rpn_result == RPN_NORMAL:
            # PASS → 연속카운트 리셋 (단발 스파이크 무시)
            self.consecutive_fail = 0
            self.ap_state = STATE_NOMINAL
            self.latched  = False

        elif rpn_result in (RPN_STALE, RPN_ERROR):
            # STALE/ERROR → 리셋 (결손 데이터로 트리거 안 함)
            self.consecutive_fail = 0
            # 상태는 보수적으로 NOMINAL 복귀 (단, latch 는 유지하지 않음)
            self.ap_state = STATE_NOMINAL
            self.latched  = False

        fsm_infer_ms = (time.perf_counter() - t_fsm) * 1000

        # ── alert 발행 (가벼운 메시지: EVS 스타일) ──
        # ts        = phys_ts (판정 전용, pass-through)
        # fsm_pub_ts = fsm 발행시각 (action 구간 측정 전용 — offboard 가 읽음)
        # 각 발행 노드가 자기 secondary timestamp 를 새로 찍는다 (cFS SB 방식):
        # ksem_pub_ts 를 재사용하지 않고 fsm_pub_ts 를 독립으로 둔다.
        alert = {
            "ts":               phys_ts,
            "fsm_pub_ts":       time.time(),
            "ap_state":         self.ap_state,
            "fail_count":       self.consecutive_fail,
            "trigger_channels": trigger_channels,
            "rpn_result":       rpn_result,
        }
        amsg = String()
        amsg.data = json.dumps(alert)
        self.alert_pub.publish(amsg)

        # ── 이벤트 로깅 (상태 변화 시) ──
        if self.ap_state != self.prev_state:
            event = f"{self.prev_state}→{self.ap_state}"
            self.det_writer.writerow([
                time.time(), event, self.ap_state, rpn_result,
                self.consecutive_fail, "|".join(trigger_channels)
            ])
            self.det_csv.flush()
            mode = "" if AP_ACTIVE else " [DRY-RUN]"
            self.get_logger().info(
                f"[AP]{mode} {event}  rpn={rpn_result}  "
                f"fail={self.consecutive_fail}/{PERSISTENCE_N}  "
                f"trig={trigger_channels}"
            )
        self.prev_state = self.ap_state

        node_latency_ms = (time.perf_counter() - t0) * 1000
        stats = self.monitor.data
        self.writer.writerow([
            time.time(),
            BG_WINDOW_DAYS_PROV, K_PROV,
            round(wp_eval_ms, 4) if wp_eval_ms == wp_eval_ms else "",      # NaN→빈칸
            round(dds_transport_ms, 3) if dds_transport_ms == dds_transport_ms else "",
            round(fsm_infer_ms, 4),
            round(node_latency_ms, 3),
            self.ap_state,
            rpn_result,
            self.consecutive_fail,
            "|".join(trigger_channels),
            stats["gpu"],
            round(stats["cpu"], 1),
            stats["power"],
            stats["ram"],
        ])
        self.csv.flush()

    def destroy_node(self):
        self.monitor.stop()
        self.csv.close()
        self.det_csv.close()
        super().destroy_node()


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="fsm_node (AP) — RPN voting + persistence, cFS LC_action"
    )
    p.add_argument("--persistence", type=int, default=PERSISTENCE_N,
                   help="연속 N슬롯 TRIGGERED → ALERT. N=2(30min)/N=3(1h @15min).")
    p.add_argument("--window", type=int, default=BG_WINDOW_DAYS_PROV,
                   help="HK 로그 provenance 용 window(days). ksem 와 동일 값 전달.")
    p.add_argument("--k", type=int, default=K_PROV,
                   help="HK 로그 provenance 용 K. ksem 와 동일 값 전달.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--active", dest="active", action="store_true",
                   help="실제 발동(ALERT latch). 기본.")
    g.add_argument("--dry-run", dest="active", action="store_false",
                   help="dry-run(PASSIVE): 판정만, latch/액추에이션 억제.")
    p.set_defaults(active=AP_ACTIVE)
    return p.parse_known_args(argv)[0]


def main(argv=None):
    args = _parse_args(argv)

    global PERSISTENCE_N, AP_ACTIVE, BG_WINDOW_DAYS_PROV, K_PROV
    PERSISTENCE_N       = args.persistence
    AP_ACTIVE           = args.active
    BG_WINDOW_DAYS_PROV = args.window   # provenance only — 판정에 미사용
    K_PROV              = args.k        # provenance only

    rclpy.init()
    node = FsmNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
