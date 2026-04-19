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

# -------------------------------------------------------
# 상태 정의
# -------------------------------------------------------
STATE_IDLE      = "IDLE"       # 탐지 없음 → Offboard 진입 대기
STATE_TRACKING  = "TRACKING"   # 탐지 있음 → Offboard + hover setpoint 유지
STATE_HOLD      = "HOLD"       # 탐지 소실 → 일정 시간 위치 유지 후 IDLE 복귀


class OffboardNode(Node):
    def __init__(self):
        super().__init__('offboard_node')

        # ── QoS: PX4 토픽은 BEST_EFFORT 필수 ──
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ── Subscribers ──
        self.bbox_sub = self.create_subscription(
            String,
            '/target_bbox',
            self.bbox_callback,
            10
        )
        self.status_sub = self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status',
            self.status_callback,
            px4_qos
        )

        # ── Publishers ──
        self.cmd_pub = self.create_publisher(
            VehicleCommand,
            '/fmu/in/vehicle_command',
            px4_qos
        )
        self.offboard_mode_pub = self.create_publisher(
            OffboardControlMode,
            '/fmu/in/offboard_control_mode',
            px4_qos
        )
        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint,
            '/fmu/in/trajectory_setpoint',
            px4_qos
        )

        # ── 상태 변수 ──
        self.state = STATE_IDLE
        self.arming_state = 0          # VehicleStatus.arming_state
        self.nav_state = 0             # VehicleStatus.nav_state
        self.last_detection_time = 0.0
        self.offboard_setpoint_counter = 0

        # HOLD 유지 시간 (탐지 소실 후 N초간 setpoint 유지)
        self.HOLD_DURATION_SEC = 3.0

        # ── CSV 이벤트 로그 ──
        self.csv = open("event_log.csv", "w", newline="")
        self.writer = csv.writer(self.csv)
        self.writer.writerow([
            "timestamp", "event", "state", "detections", "nav_state"
        ])

        # ── 10Hz 제어 루프 (Offboard는 setpoint를 2Hz 이상 계속 보내야 함) ──
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info("offboard_node ready. Waiting for detections on /target_bbox ...")

    # ────────────────────────────────────────
    # Callbacks
    # ────────────────────────────────────────
    def bbox_callback(self, msg: String):
        detections = json.loads(msg.data)
        if len(detections) == 0:
            return

        self.last_detection_time = time.time()

        if self.state == STATE_IDLE:
            self.get_logger().info(f"Detection! Transitioning IDLE → TRACKING")
            self._log_event("DETECTION_TRIGGER", detections)
            self.state = STATE_TRACKING

    def status_callback(self, msg: VehicleStatus):
        self.arming_state = msg.arming_state
        self.nav_state    = msg.nav_state

    # ────────────────────────────────────────
    # 10Hz 제어 루프
    # ────────────────────────────────────────
    def control_loop(self):
        now = time.time()

        if self.state == STATE_IDLE:
            # Offboard 진입 전 setpoint를 미리 보내둬야 모드 전환 허용됨 (PX4 요구사항)
            # → 탐지 전에도 10Hz로 OffboardControlMode + setpoint를 계속 발행
            self._publish_offboard_control_mode()
            self._publish_hover_setpoint()
            self.offboard_setpoint_counter += 1

        elif self.state == STATE_TRACKING:
            self._publish_offboard_control_mode()
            self._publish_hover_setpoint()

            # Offboard 모드 + ARM 명령 (처음 TRACKING 진입 시 한 번)
            if self.offboard_setpoint_counter >= 10:
                # 10개 setpoint 쌓인 후 Offboard 전환 (PX4 요구사항)
                self._engage_offboard_mode()
                self._arm()

            self.offboard_setpoint_counter += 1

            # 탐지 소실 감지
            if now - self.last_detection_time > self.HOLD_DURATION_SEC:
                self.get_logger().info("Detection lost. TRACKING → HOLD")
                self._log_event("DETECTION_LOST", [])
                self.state = STATE_HOLD

        elif self.state == STATE_HOLD:
            self._publish_offboard_control_mode()
            self._publish_hover_setpoint()

            # HOLD 시간 초과 → IDLE 복귀 (Disarm은 하지 않음, 안전 이유)
            if now - self.last_detection_time > self.HOLD_DURATION_SEC * 2:
                self.get_logger().info("Hold timeout. HOLD → IDLE")
                self._log_event("HOLD_TIMEOUT", [])
                self.state = STATE_IDLE
                self.offboard_setpoint_counter = 0

    # ────────────────────────────────────────
    # PX4 퍼블리시 헬퍼
    # ────────────────────────────────────────
    def _publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.timestamp      = self._now_us()
        msg.position       = True   # position setpoint 사용
        msg.velocity       = False
        msg.acceleration   = False
        msg.attitude       = False
        msg.body_rate      = False
        self.offboard_mode_pub.publish(msg)

    def _publish_hover_setpoint(self):
        """현재 위치에서 수평 hover (NED 기준, z=-1.5m = 지상 1.5m)"""
        msg = TrajectorySetpoint()
        msg.timestamp = self._now_us()
        msg.position  = [0.0, 0.0, -1.5]   # NED: z 음수가 위쪽
        msg.yaw       = 0.0
        self.setpoint_pub.publish(msg)

    def _engage_offboard_mode(self):
        """OFFBOARD 모드 전환 명령"""
        self._publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            param1=1.0,   # custom mode 사용
            param2=6.0    # PX4_CUSTOM_MAIN_MODE_OFFBOARD
        )
        self.get_logger().info("OFFBOARD mode command sent → check QGC")
        self._log_event("OFFBOARD_MODE_CMD", [])

    def _arm(self):
        """ARM 명령"""
        self._publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=1.0
        )
        self.get_logger().info("ARM command sent")
        self._log_event("ARM_CMD", [])

    def _publish_vehicle_command(self, command, param1=0.0, param2=0.0):
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

    def _log_event(self, event: str, detections: list):
        self.writer.writerow([
            time.time(), event, self.state,
            len(detections), self.nav_state
        ])
        self.csv.flush()

    def destroy_node(self):
        self.csv.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = OffboardNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()