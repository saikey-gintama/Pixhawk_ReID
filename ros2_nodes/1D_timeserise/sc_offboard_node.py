"""
offboard_node.py  —  cFS SC (RTS 층, 최소 수정판)
=================================================
fsm_node(AP)의 /sep_alert 를 받아 드론 운용 모드를 전환한다.

cFS 매핑: SC (Stored Command / RTS).
  AP(fsm_node)가 발동 조건을 판정하면, 그에 대응하는 명령 시퀀스를 실행.

기존 우주과학회 offboard_node 대비 변경점 (최소):
  - 구독: /target_bbox(detection) → /sep_alert(SEP onset alert)
  - 콜백: detection 리스트 → ap_state(NOMINAL/PRE_ALERT/ALERT) 파싱
  - PRE_ALERT: setpoint pre-streaming 시작 (Offboard 전환 선행조건 충족)
               ← 우주과학회 교훈: ALERT 직전 setpoint 스트림이 2Hz↑ 흐르고 있어야
                 PX4 Offboard 전환이 거부되지 않음
  - ALERT: Offboard 전환 + ARM
  - NOMINAL: AP가 명시적으로 보냄 → HOLD 복귀 (시간추정 불필요)
  - 큰 구조(IDLE/TRACKING/HOLD FSM, 10Hz 루프, PX4 헬퍼)는 그대로 유지

상태 매핑:
  IDLE      ← ap_state NOMINAL (또는 미수신)
  TRACKING  ← ap_state ALERT   (Offboard+ARM, active 관측 모드)
  HOLD      ← ALERT→NOMINAL 복귀 후 일정시간 유지
  PRE_ALERT 는 별도 state 가 아니라 "pre-streaming 플래그"로 처리
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import String
from px4_msgs.msg import (
    VehicleCommand,
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleStatus,
)
import time
import csv
import json
import os
import argparse

RESULT_DIR = os.environ.get("RESULT_DIR", ".")

# ══════════════════════════════════════════════════════
# SC 파라미터 — argparse 로 런타임 덮어쓰기 (cFS TBL 철학)
# ══════════════════════════════════════════════════════
# PASSIVE(dry-run): PX4 VehicleCommand 를 실제로 발행하지 않는다.
#   True  = 측정 전용 (action latency 만, PX4 SITL 없이) ← 현재 마감 전 기본
#   False = ACTIVE (실제 Offboard+ARM, PX4 SITL/실기 연결 시)
SC_PASSIVE = True
# ══════════════════════════════════════════════════════

# -------------------------------------------------------
# 상태 정의
# -------------------------------------------------------
STATE_IDLE     = "IDLE"      # NOMINAL → Offboard 진입 대기
STATE_TRACKING = "TRACKING"  # ALERT → Offboard + ARM (active 관측)
STATE_HOLD     = "HOLD"      # ALERT 해제 → 일정 시간 유지 후 IDLE 복귀

# AP alert 상태 (fsm_node 와 일치)
AP_NOMINAL   = "NOMINAL"
AP_PRE_ALERT = "PRE_ALERT"
AP_ALERT     = "ALERT"


class OffboardNode(Node):
    def __init__(self):
        super().__init__('offboard_node')

        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ── Subscribers ──
        # /target_bbox → /sep_alert 로 교체
        self.alert_sub = self.create_subscription(
            String, '/sep_alert', self.alert_callback, 10
        )
        self.status_sub = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status_v1', self.status_callback, px4_qos
        )

        # ── Publishers ──
        self.cmd_pub           = self.create_publisher(VehicleCommand,      '/fmu/in/vehicle_command',       px4_qos)
        self.offboard_mode_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', px4_qos)
        self.setpoint_pub      = self.create_publisher(TrajectorySetpoint,  '/fmu/in/trajectory_setpoint',   px4_qos)

        # ── 상태 변수 ──
        self.state                     = STATE_IDLE
        self.arming_state              = 0
        self.nav_state                 = 0
        self.last_alert_time           = 0.0
        self.last_action_ms            = float("nan")   # action 구간 측정값 (telemetry)
        self.offboard_setpoint_counter = 0
        self.HOLD_DURATION_SEC         = 3.0

        # PASSIVE(dry-run): PX4 명령을 실제로 쏘지 않고 수신/전이 시각만 기록.
        # cFS LC ACTIVE/PASSIVE 의 PASSIVE 모드 — RTS 미발사, 로그만.
        # PX4 SITL 은 마감 뒤로 미뤘으므로 현재는 PASSIVE 로 action latency 만 측정.
        self.passive = SC_PASSIVE

        # PRE_ALERT pre-streaming 플래그
        # (ALERT 전에 setpoint 스트림을 미리 흘려 Offboard 전환 선행조건 충족)
        self.pre_streaming = False

        # ── CSV 이벤트 로그 (action_ms = fsm 발행→offboard 수신 transport) ──
        log_path    = os.path.join(RESULT_DIR, "event_log.csv")
        self.csv    = open(log_path, "w", newline="")
        self.writer = csv.writer(self.csv)
        self.writer.writerow([
            "timestamp", "event", "state", "ap_state", "nav_state",
            "action_ms", "passive"
        ])
        self.get_logger().info(f"Logging to {log_path}")

        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("offboard_node ready. Waiting for /sep_alert ...")

    # ────────────────────────────────────────
    # Callbacks
    # ────────────────────────────────────────
    def alert_callback(self, msg: String):
        try:
            alert = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        ap_state = alert.get("ap_state", AP_NOMINAL)
        self.last_alert_time = time.time()

        # ── action 구간 측정 (telemetry 전용, 제어 로직 미사용) ──
        # fsm_pub_ts: fsm_node 의 /sep_alert 발행시각. action = 수신 - 발행.
        # 이 값은 로깅에만 쓰고 state 전이 판단엔 절대 쓰지 않는다.
        fsm_pub_ts = alert.get("fsm_pub_ts", None)
        self.last_action_ms = (
            (self.last_alert_time - fsm_pub_ts) * 1000.0 if fsm_pub_ts else float("nan")
        )

        if ap_state == AP_PRE_ALERT:
            # 아직 발동 전 — setpoint pre-streaming 시작 (전환 선행조건 준비)
            if not self.pre_streaming:
                self.pre_streaming = True
                self.get_logger().info("PRE_ALERT: start setpoint pre-streaming")
                self._log_event("PRE_ALERT_PRESTREAM", ap_state)

        elif ap_state == AP_ALERT:
            self.pre_streaming = True   # ALERT 면 당연히 스트리밍 중
            if self.state == STATE_IDLE:
                self.get_logger().info("ALERT! IDLE → TRACKING (Offboard+ARM)")
                self._log_event("ALERT_TRIGGER", ap_state)
                self.state = STATE_TRACKING

        elif ap_state == AP_NOMINAL:
            # AP 가 명시적으로 정상 복귀 신호를 보냄
            self.pre_streaming = False
            if self.state == STATE_TRACKING:
                self.get_logger().info("NOMINAL: TRACKING → HOLD")
                self._log_event("ALERT_CLEARED", ap_state)
                self.state = STATE_HOLD

    def status_callback(self, msg: VehicleStatus):
        self.arming_state = msg.arming_state
        self.nav_state    = msg.nav_state

    # ────────────────────────────────────────
    # 10Hz 제어 루프
    # ────────────────────────────────────────
    def control_loop(self):
        now = time.time()

        if self.state == STATE_IDLE:
            # PRE_ALERT 면 pre-streaming (선행조건 충족), 아니면 대기 스트림
            self._publish_offboard_control_mode()
            self._publish_hover_setpoint()
            if self.pre_streaming:
                self.offboard_setpoint_counter += 1
            else:
                self.offboard_setpoint_counter = 0

        elif self.state == STATE_TRACKING:
            self._publish_offboard_control_mode()
            self._publish_hover_setpoint()

            # setpoint 가 충분히 쌓였을 때만 Offboard 전환 + ARM
            # (우주과학회 교훈: 스트림 선행 없이 전환하면 PX4 가 거부)
            if self.offboard_setpoint_counter >= 10:
                self._engage_offboard_mode()
                self._arm()
            self.offboard_setpoint_counter += 1

        elif self.state == STATE_HOLD:
            self._publish_offboard_control_mode()
            self._publish_hover_setpoint()

            if now - self.last_alert_time > self.HOLD_DURATION_SEC * 2:
                self.get_logger().info("Hold timeout. HOLD → IDLE")
                self._log_event("HOLD_TIMEOUT", AP_NOMINAL)
                self.state = STATE_IDLE
                self.offboard_setpoint_counter = 0
                self.pre_streaming = False

    # ────────────────────────────────────────
    # PX4 퍼블리시 헬퍼 (변경 없음)
    # ────────────────────────────────────────
    def _publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.timestamp    = self._now_us()
        msg.position     = True
        msg.velocity     = False
        msg.acceleration = False
        msg.attitude     = False
        msg.body_rate    = False
        self.offboard_mode_pub.publish(msg)

    def _publish_hover_setpoint(self):
        msg = TrajectorySetpoint()
        msg.timestamp = self._now_us()
        msg.position  = [0.0, 0.0, -1.5]
        msg.yaw       = 0.0
        self.setpoint_pub.publish(msg)

    def _engage_offboard_mode(self):
        self._publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            param1=1.0, param2=6.0
        )
        self.get_logger().info("OFFBOARD mode command sent → check QGC")
        self._log_event("OFFBOARD_MODE_CMD", AP_ALERT)

    def _arm(self):
        self._publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=1.0
        )
        self.get_logger().info("ARM command sent")
        self._log_event("ARM_CMD", AP_ALERT)

    def _publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        # PASSIVE(dry-run): 실제 명령 미발행. cFS LC PASSIVE = RTS 미발사.
        # action latency 측정은 alert 수신 시점에 이미 끝났으므로 여기서 막아도
        # 측정엔 영향 없다 (측정과 액추에이션 분리).
        if self.passive:
            self._log_event(f"PASSIVE_SUPPRESS_CMD_{int(command)}", AP_ALERT)
            return
        msg = VehicleCommand()
        msg.timestamp        = self._now_us()
        msg.command          = command
        msg.param1           = param1
        msg.param2           = param2
        msg.target_system    = 1
        msg.target_component = 1
        msg.source_system    = 1
        msg.source_component = 1
        msg.from_external    = True
        self.cmd_pub.publish(msg)

    def _now_us(self) -> int:
        return int(self.get_clock().now().nanoseconds / 1000)

    def _log_event(self, event: str, ap_state: str):
        am = self.last_action_ms
        self.writer.writerow([
            time.time(), event, self.state,
            ap_state, self.nav_state,
            round(am, 3) if am == am else "",   # NaN→빈칸
            int(self.passive),
        ])
        self.csv.flush()

    def destroy_node(self):
        self.csv.close()
        super().destroy_node()


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="offboard_node (SC) — alert→action, cFS PASSIVE dry-run"
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--passive", dest="passive", action="store_true",
                   help="dry-run: PX4 명령 미발행, action latency 만 측정. 기본.")
    g.add_argument("--active", dest="passive", action="store_false",
                   help="ACTIVE: 실제 Offboard+ARM (PX4 SITL/실기 연결 시).")
    p.set_defaults(passive=SC_PASSIVE)
    return p.parse_known_args(argv)[0]


def main(argv=None):
    args = _parse_args(argv)
    global SC_PASSIVE
    SC_PASSIVE = args.passive

    rclpy.init()
    node = OffboardNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
