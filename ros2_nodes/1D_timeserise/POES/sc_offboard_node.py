"""
sc_offboard_node.py  --  cFS SC(RTS 층), POES 온보드 파이프라인 S4
======================================================================
/sep_alert(ap_fsm_node) 를 받아 IDLE/TRACKING/HOLD 로 드론 운용 모드를 전환한다.
KSEM/sc_offboard_node.py 승계: PX4 헬퍼, PASSIVE dry-run, 10Hz 제어루프,
offboard_setpoint_counter>=10 선행 스트림 규약을 그대로 유지한다.

상태 매핑(md 6절 S4, KSEM 과 동일):
  ap_state PRE_ALERT -> pre_streaming=True. setpoint pre-streaming 시작.
      **상태 전이는 없다**(가역 구간) -- PX4 는 Offboard 전환 직전 선행 스트림이
      2Hz 이상 흐르고 있어야 전환을 거부하지 않는다(우주과학회 실증 교훈). 이것이
      2단계 경보(PRE_ALERT/ALERT)의 운용상 근거다: PRE_ALERT 에서 미리 흘려두지
      않으면 ALERT 순간 Offboard 전환 자체가 거부될 수 있다.
  ap_state ALERT     -> IDLE -> TRACKING. setpoint 가 충분히(counter>=10) 쌓였을
      때만 실제 Offboard 전환 + ARM(KSEM 규약 그대로, 재구현 아님).
  ap_state NOMINAL   -> TRACKING -> HOLD. pre_streaming=False.
  HOLD 는 HOLD_DURATION_SEC*2 경과 후 IDLE 로 자동 복귀(KSEM 그대로).

판정 로직(ScFsmCore)은 Node 클래스 밖 -- rclpy/px4_msgs 없이 import 가능(md 2절).

--ai-gate {off, shadow, gate}  기본 off:
  off    : /ai_verdict 구독 안 함. 현행 KSEM 동작 그대로(시나리오 b).
  shadow : /ai_verdict 구독해 로그에만 기록. **제어 경로는 off 와 완전 동일**
           (시나리오 c/d 기본 -- ΔPower 가 TCN 비용만 분리되려면 제어 동작이
           같아야 한다). ScFsmCore.on_ai_verdict() 는 last_p_event 만 갱신하고
           state/pre_streaming/counter 는 절대 건드리지 않는다.
  gate   : ALERT 이면서 최근 /ai_verdict.p_event >= --ai-threshold 일 때만
           IDLE->TRACKING 전이. 데모용, 비용 측정에는 쓰지 않는다.

배속 주의: SC 의 10Hz 루프와 HOLD_DURATION_SEC 은 벽시계 기준이고 리플레이 배속과
무관하다. 1800배 리플레이면 물리 15분이 벽시계 0.5초라 HOLD 타임아웃(기본 6초)보다
먼저 다음 alert 가 온다 -- 가속 리플레이에서 IDLE/TRACKING/HOLD **순환 횟수 자체는
물리적 대표성이 없다**. 유의미한 것은 transport/action 지연(M6, M10, action_ms)뿐.
--hold-scale 로 필요 시 HOLD_DURATION 을 배속에 맞춰 축소할 수 있다. 논문 4.4절용
SITL 실증 1회는 1배속 근처에서 별도로 돌린다(--hold-scale 1.0, 기본값).

측정:
  action_ms  = SC 수신시각 - alert 의 fsm_pub_ts        (KSEM 기존 규약, 유지)
  M10 e2e_ms = SC 명령 발행시각 - t_sample_rx            (신규, S4)
      t_sample_rx 는 WP 가 원 샘플 처리를 시작한 벽시계(S1에서 붙임) -- AP/AI 를
      거치며 변경 없이 그대로 전달된 값을 SC 가 받아 쓴다. 판정에는 쓰지 않는다
      (물리시각 ts 와 분리 유지, KSEM 측정/판정 분리 원칙).
  전이 시각 기록: PRE_ALERT 수신 / ALERT 수신 / Offboard 명령 / ARM 명령.
"""
from __future__ import annotations

import json
import os
import time
from time import perf_counter

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
    from std_msgs.msg import String
    _HAVE_RCLPY = True
except ImportError:                       # Windows 개발 환경 -- 로직 단위 검증용
    _HAVE_RCLPY = False
    rclpy = None
    String = None
    QoSProfile = ReliabilityPolicy = HistoryPolicy = DurabilityPolicy = None

    class Node:                            # type: ignore -- 더미 베이스, 인스턴스화 안 함
        pass

try:
    from px4_msgs.msg import VehicleCommand, OffboardControlMode, TrajectorySetpoint, VehicleStatus
    _HAVE_PX4 = True
except ImportError:                       # Windows 개발 환경 -- px4_msgs 도 rclpy 와 동일하게 guard
    _HAVE_PX4 = False
    VehicleCommand = OffboardControlMode = TrajectorySetpoint = VehicleStatus = None

RESULT_DIR = os.environ.get("RESULT_DIR", ".")

# 실 PX4/MAVLink 명령 ID(px4_msgs.VehicleCommand 와 동일 값) -- ScFsmCore 가 px4_msgs
# 없이도 로그/이벤트에 실제 명령 코드를 남길 수 있도록 여기서 상수로 고정.
CMD_DO_SET_MODE       = 176   # VEHICLE_CMD_DO_SET_MODE
CMD_COMPONENT_ARM_DISARM = 400  # VEHICLE_CMD_COMPONENT_ARM_DISARM

# ══════════════════════════════════════════════════════
# 파라미터 블록 -- 기본값(argparse 로 런타임 덮어쓰기, cFS TBL 철학)
# ══════════════════════════════════════════════════════
PASSIVE          = True    # dry-run: PX4 명령 미발행, action/e2e latency 만 측정. 기본.
AI_GATE_MODE      = "off"   # {off, shadow, gate}
AI_THRESHOLD      = 0.5
HOLD_DURATION_SEC_BASE = 3.0   # KSEM 원값(초, 실시간/1배속 기준)
HOLD_SCALE        = 1.0        # 가속 리플레이용 축소 배수. 1.0=KSEM 그대로.

# 상태 정의 (KSEM 과 동일)
STATE_IDLE, STATE_TRACKING, STATE_HOLD = "IDLE", "TRACKING", "HOLD"
AP_NOMINAL, AP_PRE_ALERT, AP_ALERT = "NOMINAL", "PRE_ALERT", "ALERT"


# ══════════════════════════════════════════════════════
# SC 핵심 로직 -- rclpy/px4_msgs 무관. Node 는 이 클래스를 감쌀 뿐(md 2절).
# ══════════════════════════════════════════════════════
class ScFsmCore:
    """IDLE/TRACKING/HOLD 상태기계 + pre_streaming + offboard_setpoint_counter.
    PASSIVE 여부까지 여기서 판정(진짜 PX4 발행이 필요한 부분은 Node 가 명령 목록을
    보고 실행만 한다) -- 그래야 rclpy/px4_msgs 없이도 PASSIVE_SUPPRESS 를 검증할 수 있다."""

    def __init__(self, ai_gate_mode: str = AI_GATE_MODE, ai_threshold: float = AI_THRESHOLD,
                hold_duration_sec: float = HOLD_DURATION_SEC_BASE, passive: bool = PASSIVE):
        self.ai_gate_mode = ai_gate_mode
        self.ai_threshold = ai_threshold
        self.hold_duration_sec = hold_duration_sec
        self.passive = passive

        self.state = STATE_IDLE
        self.pre_streaming = False
        self.offboard_setpoint_counter = 0
        self.last_alert_time = 0.0
        self.last_action_ms = float("nan")
        self.last_t_sample_rx: float | None = None
        self.last_p_event: float | None = None

        self.commands_sent: list[str] = []         # ACTIVE 로 실제 "발행"된 명령(시뮬레이션 기록)
        self.commands_suppressed: list[str] = []   # PASSIVE 로 억제된 명령

    def on_alert(self, ap_state: str, now: float, fsm_pub_ts, t_sample_rx) -> list[str]:
        """/sep_alert 수신 1회. 발생한 이벤트 이름 목록(0개 이상) 반환.
        gate 모드가 아니면(off/shadow) 항상 allowed=True -- 두 모드의 전이 시퀀스가
        바이트 단위로 같아야 한다(검증 ③).

        last_alert_time 은 PRE_ALERT/ALERT 일 때만 갱신한다(NOMINAL 이면 안 건드림).
        AP 는 매 틱(대부분 NOMINAL)마다 /sep_alert 를 발행하므로, KSEM 원본처럼
        무조건 갱신하면 이 값이 끊임없이 새로고침돼 HOLD 타임아웃이 영원히 발동하지
        않는다 -- 전 구간 리플레이로 실측된 문제(변수명 자체가 'last_ALERT_time'이라
        원 의도도 이쪽이었을 것). action_ms(전송지연 로그용)는 메시지 종류와 무관하게
        그대로 매번 갱신한다(상태기계와 무관한 순수 진단값)."""
        if ap_state in (AP_PRE_ALERT, AP_ALERT):
            self.last_alert_time = now
        self.last_action_ms = (now - fsm_pub_ts) * 1000.0 if fsm_pub_ts else float("nan")
        self.last_t_sample_rx = t_sample_rx
        events: list[str] = []

        if ap_state == AP_PRE_ALERT:
            if not self.pre_streaming:
                self.pre_streaming = True
                events.append("PRE_ALERT_PRESTREAM")

        elif ap_state == AP_ALERT:
            self.pre_streaming = True   # ALERT 면 당연히 스트리밍 중
            allowed = True
            if self.ai_gate_mode == "gate":
                allowed = (self.last_p_event is not None and self.last_p_event >= self.ai_threshold)
            if self.state == STATE_IDLE:
                if allowed:
                    self.state = STATE_TRACKING
                    events.append("ALERT_TRIGGER")
                else:
                    events.append("ALERT_GATED_SUPPRESS")   # gate 모드 전용 이벤트

        elif ap_state == AP_NOMINAL:
            self.pre_streaming = False
            if self.state == STATE_TRACKING:
                self.state = STATE_HOLD
                events.append("ALERT_CLEARED")

        return events

    def on_ai_verdict(self, p_event) -> None:
        """shadow/gate 모드에서만 호출된다(off 는 구독 자체를 안 함). last_p_event 만
        갱신 -- state/pre_streaming/offboard_setpoint_counter 는 절대 건드리지 않는다.
        그래야 shadow 의 제어 경로가 off 와 완전히 같다(검증 ③)."""
        self.last_p_event = p_event

    def tick_control(self, now: float) -> list[str]:
        """10Hz 제어루프 1회(KSEM 구조 그대로: TRACKING 이고 counter>=10 이면 매 틱
        Offboard+ARM 명령을 다시 낸다 -- 원본에 재발사 방지 가드가 없어 그대로 승계).
        반환: 이번 틱 발생 이벤트."""
        events: list[str] = []
        if self.state == STATE_IDLE:
            if self.pre_streaming:
                self.offboard_setpoint_counter += 1
            else:
                self.offboard_setpoint_counter = 0

        elif self.state == STATE_TRACKING:
            if self.offboard_setpoint_counter >= 10:
                for name, cmd_id in (("OFFBOARD_MODE_CMD", CMD_DO_SET_MODE),
                                     ("ARM_CMD", CMD_COMPONENT_ARM_DISARM)):
                    if self.passive:
                        self.commands_suppressed.append(name)
                    else:
                        self.commands_sent.append(name)
                    events.append(name)
            self.offboard_setpoint_counter += 1

        elif self.state == STATE_HOLD:
            if now - self.last_alert_time > self.hold_duration_sec * 2:
                self.state = STATE_IDLE
                self.offboard_setpoint_counter = 0
                self.pre_streaming = False
                events.append("HOLD_TIMEOUT")

        return events


# ──────────────────────────────────────────────────────
# ScOffboardNode -- rclpy/px4_msgs 없으면 인스턴스화만 못 함
# ──────────────────────────────────────────────────────
class ScOffboardNode(Node):
    def __init__(self):
        super().__init__("sc_offboard_node")
        if String is None:
            raise RuntimeError("rclpy/std_msgs 없음 -- Jetson(ROS2 환경)에서 실행할 것")
        if not _HAVE_PX4:
            raise RuntimeError("px4_msgs 없음 -- PX4 SITL/실기 연결 환경에서만 실행 가능")

        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.alert_sub = self.create_subscription(String, "/sep_alert", self.on_alert, 10)
        self.status_sub = self.create_subscription(
            VehicleStatus, "/fmu/out/vehicle_status_v1", self.on_status, px4_qos)
        self.verdict_sub = None
        if AI_GATE_MODE != "off":
            self.verdict_sub = self.create_subscription(String, "/ai_verdict", self.on_verdict, 10)

        self.cmd_pub = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", px4_qos)
        self.offboard_mode_pub = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", px4_qos)
        self.setpoint_pub = self.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", px4_qos)

        self.core = ScFsmCore(AI_GATE_MODE, AI_THRESHOLD,
                              HOLD_DURATION_SEC_BASE * HOLD_SCALE, PASSIVE)
        self.nav_state = 0
        self.arming_state = 0

        log_path = os.path.join(RESULT_DIR, "event_log.csv")
        import csv
        self.csv = open(log_path, "w", newline="")
        self.writer = csv.writer(self.csv)
        self.writer.writerow([
            "timestamp", "event", "state", "ap_state", "nav_state",
            "action_ms", "e2e_ms", "passive", "ai_gate_mode",
        ])
        self.get_logger().info(f"[SC] Logging to {log_path}")
        self.get_logger().info(
            f"[SC] ai_gate={AI_GATE_MODE} passive={PASSIVE} "
            f"hold_duration={self.core.hold_duration_sec:.2f}s(scale={HOLD_SCALE}) -- "
            f"가속 리플레이에서 HOLD/IDLE 순환 자체는 물리적 대표성 없음. "
            f"transport/action/M10 만 유의미.")

        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("[SC] ready. Waiting for /sep_alert ...")

    # ── Callbacks ──
    def on_alert(self, msg):
        try:
            alert = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        ap_state = alert.get("ap_state", AP_NOMINAL)
        now = time.time()
        events = self.core.on_alert(ap_state, now, alert.get("fsm_pub_ts"), alert.get("t_sample_rx"))
        for ev in events:
            self._log_event(ev, ap_state)

    def on_verdict(self, msg):
        try:
            v = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self.core.on_ai_verdict(v.get("p_event"))

    def on_status(self, msg):
        self.arming_state = msg.arming_state
        self.nav_state = msg.nav_state

    # ── 10Hz 제어 루프 ──
    def control_loop(self):
        now = time.time()
        self._publish_offboard_control_mode()
        self._publish_hover_setpoint()

        events = self.core.tick_control(now)
        for ev in events:
            if ev == "OFFBOARD_MODE_CMD":
                self._engage_offboard_mode()
            elif ev == "ARM_CMD":
                self._arm()
            elif ev == "HOLD_TIMEOUT":
                self.get_logger().info("[SC] Hold timeout. HOLD -> IDLE")
                self._log_event("HOLD_TIMEOUT", AP_NOMINAL)

    # ── PX4 퍼블리시 헬퍼 (KSEM 과 동일, 변경 없음) ──
    def _publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = self._now_us()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        self.offboard_mode_pub.publish(msg)

    def _publish_hover_setpoint(self):
        msg = TrajectorySetpoint()
        msg.timestamp = self._now_us()
        msg.position = [0.0, 0.0, -1.5]
        msg.yaw = 0.0
        self.setpoint_pub.publish(msg)

    def _compute_e2e_ms(self) -> float:
        t_rx = self.core.last_t_sample_rx
        return (time.time() - t_rx) * 1000.0 if t_rx else float("nan")

    def _engage_offboard_mode(self):
        e2e_ms = self._compute_e2e_ms()
        self._publish_vehicle_command(CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        if not self.core.passive:
            self.get_logger().info("[SC] OFFBOARD mode command sent -> check QGC")
        self._log_event("OFFBOARD_MODE_CMD" if not self.core.passive else "PASSIVE_SUPPRESS_CMD_OFFBOARD_MODE",
                        AP_ALERT, e2e_ms)

    def _arm(self):
        e2e_ms = self._compute_e2e_ms()
        self._publish_vehicle_command(CMD_COMPONENT_ARM_DISARM, param1=1.0)
        if not self.core.passive:
            self.get_logger().info("[SC] ARM command sent")
        self._log_event("ARM_CMD" if not self.core.passive else "PASSIVE_SUPPRESS_CMD_ARM",
                        AP_ALERT, e2e_ms)

    def _publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        # PASSIVE(dry-run): 실제 명령 미발행. cFS LC PASSIVE = RTS 미발사.
        # action/e2e latency 측정은 alert 수신/명령 판단 시점에 이미 끝났으므로
        # 여기서 막아도 측정엔 영향 없다(측정과 액추에이션 분리, KSEM 원칙 승계).
        if self.core.passive:
            return
        msg = VehicleCommand()
        msg.timestamp = self._now_us()
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.cmd_pub.publish(msg)

    def _now_us(self) -> int:
        return int(self.get_clock().now().nanoseconds / 1000)

    def _log_event(self, event: str, ap_state: str, e2e_ms: float | None = None):
        am = self.core.last_action_ms
        self.writer.writerow([
            time.time(), event, self.core.state, ap_state, self.nav_state,
            round(am, 3) if am == am else "",
            round(e2e_ms, 3) if (e2e_ms is not None and e2e_ms == e2e_ms) else "",
            int(self.core.passive), self.core.ai_gate_mode,
        ])
        self.csv.flush()

    def destroy_node(self):
        if not self.csv.closed:
            self.csv.close()
        super().destroy_node()


def _parse_args(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="sc_offboard_node (SC) -- alert->action, cFS PASSIVE dry-run")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--passive", dest="passive", action="store_true", help="dry-run. 기본.")
    g.add_argument("--active", dest="passive", action="store_false", help="ACTIVE: 실제 Offboard+ARM.")
    p.set_defaults(passive=PASSIVE)
    p.add_argument("--ai-gate", choices=["off", "shadow", "gate"], default=AI_GATE_MODE)
    p.add_argument("--ai-threshold", type=float, default=AI_THRESHOLD)
    p.add_argument("--hold-scale", type=float, default=HOLD_SCALE,
                   help="HOLD_DURATION_SEC 배수. 가속 리플레이용 축소. 1.0=KSEM 원값(3.0s).")
    return p.parse_known_args(argv)[0]


def main(argv=None):
    args = _parse_args(argv)
    global PASSIVE, AI_GATE_MODE, AI_THRESHOLD, HOLD_SCALE
    PASSIVE       = args.passive
    AI_GATE_MODE  = args.ai_gate
    AI_THRESHOLD  = args.ai_threshold
    HOLD_SCALE    = args.hold_scale

    if not _HAVE_RCLPY:
        raise SystemExit("[SC] rclpy 없음 -- 이 노드 실행은 Jetson(ROS2 환경)에서만 가능. "
                         "로직 단위 검증은 ScFsmCore 를 직접 import 해서 할 것.")
    if not _HAVE_PX4:
        raise SystemExit("[SC] px4_msgs 없음 -- PX4 SITL/실기 연결 환경에서만 실행 가능.")

    rclpy.init()
    node = ScOffboardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
