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
import math

RESULT_DIR = os.environ.get("RESULT_DIR", ".")

# ═══════════════════════════════════════════════════════
# ▶ 제어 파라미터  ← 여기만 바꾸면 됩니다
# ═══════════════════════════════════════════════════════

# PID (픽셀 오차 → 미터 이동)
PID_KP, PID_KI, PID_KD   = 0.003, 0.0005, 0.001
PID_OUTPUT_LIMIT          = 1.5   # 한 틱 최대 이동량 [m]

# setpoint 절대 안전 범위 [m, NED]
SP_X_LIMIT  = 5.0     # 앞뒤 ±5m
SP_Y_LIMIT  = 5.0     # 좌우 ±5m
SP_Z_MIN    = -3.0    # 최대 고도 (NED 음수)
SP_Z_MAX    = -0.5    # 최소 고도 (너무 낮지 않도록)

# yaw 제어 (bbox 중심 x오차 → yaw 조정)
YAW_ENABLE      = True
YAW_KP          = 0.003        # 픽셀 오차 → rad
YAW_MAX_RATE    = math.radians(30)  # 최대 yaw 변화량 [rad/tick]
FRAME_WIDTH     = 640.0

# bbox 면적 기반 z 고도 정책
# yolo_bbox_node.py 의 AREA_ZONES 와 동일하게 유지하세요
AREA_ZONES = [
    (5_000,         -2.5),
    (20_000,        -1.5),
    (60_000,        -0.8),
    (float("inf"),  -0.5),
]

# HOLD 타임아웃
HOLD_TIMEOUT_SEC = 3.0   # HOLD 상태에서 이 시간 지나면 IDLE 복귀

# ═══════════════════════════════════════════════════════


# -------------------------------------------------------
# 상태 정의
# -------------------------------------------------------
STATE_IDLE     = "IDLE"
STATE_TRACKING = "TRACKING"
STATE_HOLD     = "HOLD"


# -------------------------------------------------------
# PID 제어기
# -------------------------------------------------------
class PIDController:
    def __init__(self, kp, ki, kd, output_limit):
        self.kp = kp; self.ki = ki; self.kd = kd
        self.output_limit = output_limit
        self._integral    = 0.0
        self._prev_error  = 0.0
        self._prev_time   = None

    def update(self, error: float) -> float:
        now = time.time()
        dt  = 0.1 if self._prev_time is None else max(now - self._prev_time, 1e-4)
        self._prev_time  = now
        self._integral  += error * dt
        self._integral   = max(-10.0, min(10.0, self._integral))
        derivative       = (error - self._prev_error) / dt
        self._prev_error = error
        out = self.kp * error + self.ki * self._integral + self.kd * derivative
        return max(-self.output_limit, min(self.output_limit, out))

    def reset(self):
        self._integral = 0.0; self._prev_error = 0.0; self._prev_time = None


# -------------------------------------------------------
# Offboard BBox Node
# -------------------------------------------------------
class OffboardBboxNode(Node):
    def __init__(self):
        super().__init__('offboard_bbox_node')

        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ── Subscribers ──
        self.bbox_sub   = self.create_subscription(
            String, '/target_bbox', self.bbox_callback, 10)
        self.status_sub = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.status_callback, px4_qos)

        # ── Publishers ──
        self.cmd_pub           = self.create_publisher(
            VehicleCommand,      '/fmu/in/vehicle_command',       px4_qos)
        self.offboard_mode_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', px4_qos)
        self.setpoint_pub      = self.create_publisher(
            TrajectorySetpoint,  '/fmu/in/trajectory_setpoint',   px4_qos)

        # ── 프레임 중심 ──
        self.cx_center = FRAME_WIDTH / 2.0
        self.cy_center = FRAME_WIDTH / 2.0  # 640×640 기준

        # ── PID ──
        self.pid_x = PIDController(PID_KP, PID_KI, PID_KD, PID_OUTPUT_LIMIT)
        self.pid_y = PIDController(PID_KP, PID_KI, PID_KD, PID_OUTPUT_LIMIT)

        # ── 현재 setpoint ──
        self.sp_x   = 0.0
        self.sp_y   = 0.0
        self.sp_z   = -1.5   # 초기 hover z
        self.sp_yaw = 0.0    # [rad]

        # ── 상태 ──
        self.state                     = STATE_IDLE
        self.arming_state              = 0
        self.nav_state                 = 0
        self.last_detection_time       = 0.0
        self.offboard_setpoint_counter = 0

        # ── CSV 이벤트 로그 ──
        log_path    = os.path.join(RESULT_DIR, "offboard_event_log.csv")
        self.csv    = open(log_path, "w", newline="")
        self.writer = csv.writer(self.csv)
        self.writer.writerow([
            "timestamp", "event", "state",
            "err_x_px", "err_y_px",
            "sp_x", "sp_y", "sp_z", "sp_yaw_deg",
            "nav_state"
        ])
        self.get_logger().info(f"Logging to {log_path}")

        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info(
            "offboard_bbox_node ready. "
            f"Safety bounds: x±{SP_X_LIMIT}m y±{SP_Y_LIMIT}m z[{SP_Z_MAX},{SP_Z_MIN}]m"
        )

    # ──────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────
    def bbox_callback(self, msg: String):
        detections = json.loads(msg.data)
        if not detections:
            return

        self.last_detection_time = time.time()

        # 최고 confidence 선택 (yolo_bbox_node에서 이미 필터됨)
        best   = max(detections, key=lambda d: d["conf"])
        x1, y1, x2, y2 = best["bbox"]
        area   = (x2 - x1) * (y2 - y1)

        # smooth_cx/cy가 있으면 사용, 없으면 raw 계산
        bbox_cx = best.get("smooth_cx", (x1 + x2) / 2.0)
        bbox_cy = best.get("smooth_cy", (y1 + y2) / 2.0)

        err_x = bbox_cx - self.cx_center   # 양수 = 오른쪽 → NED y+
        err_y = bbox_cy - self.cy_center   # 양수 = 아래   → NED x+

        # ── PID → setpoint 누적 ──
        self.sp_y += self.pid_y.update(err_x)
        self.sp_x += self.pid_x.update(err_y)

        # ── bbox 면적 기반 z 고도 결정 ──
        self.sp_z = self._area_to_z(area)

        # ── yaw 제어 ──
        if YAW_ENABLE:
            yaw_delta = YAW_KP * err_x
            yaw_delta = max(-YAW_MAX_RATE, min(YAW_MAX_RATE, yaw_delta))
            self.sp_yaw += yaw_delta
            # yaw를 -π ~ +π 범위로 wrap
            self.sp_yaw = (self.sp_yaw + math.pi) % (2 * math.pi) - math.pi

        # ── setpoint 안전 범위 클램핑 ──
        self.sp_x = max(-SP_X_LIMIT, min(SP_X_LIMIT, self.sp_x))
        self.sp_y = max(-SP_Y_LIMIT, min(SP_Y_LIMIT, self.sp_y))
        self.sp_z = max(SP_Z_MIN,    min(SP_Z_MAX,    self.sp_z))

        self._log_event("BBOX_UPDATE", err_x, err_y)

        if self.state == STATE_IDLE:
            self.get_logger().info("Detection confirmed! IDLE → TRACKING")
            self._log_event("DETECTION_TRIGGER", err_x, err_y)
            self.state = STATE_TRACKING

        elif self.state == STATE_HOLD:
            self.get_logger().info("Detection recovered. HOLD → TRACKING")
            self.pid_x.reset()
            self.pid_y.reset()
            self._log_event("DETECTION_RECOVER", err_x, err_y)
            self.state = STATE_TRACKING

    def status_callback(self, msg: VehicleStatus):
        self.arming_state = msg.arming_state
        self.nav_state    = msg.nav_state

    # ──────────────────────────────────────────
    # 10Hz 제어 루프
    # ──────────────────────────────────────────
    def control_loop(self):
        now = time.time()

        if self.state == STATE_IDLE:
            # setpoint를 계속 발행해야 Offboard mode 진입 가능
            self._publish_offboard_control_mode()
            self._publish_setpoint()
            self.offboard_setpoint_counter += 1

        elif self.state == STATE_TRACKING:
            self._publish_offboard_control_mode()
            self._publish_setpoint()

            if self.offboard_setpoint_counter >= 10:
                self._engage_offboard_mode()
                self._arm()
            self.offboard_setpoint_counter += 1

            # 탐지 소실 0.5초 → HOLD
            if now - self.last_detection_time > 0.5:
                self.get_logger().info("Detection lost. TRACKING → HOLD")
                self._log_event("DETECTION_LOST", 0.0, 0.0)
                self.pid_x.reset()
                self.pid_y.reset()
                self.state = STATE_HOLD

        elif self.state == STATE_HOLD:
            # 마지막 sp_x, sp_y, sp_z 유지
            self._publish_offboard_control_mode()
            self._publish_setpoint()

            if now - self.last_detection_time > HOLD_TIMEOUT_SEC:
                self.get_logger().info("Hold timeout. HOLD → IDLE")
                self._log_event("HOLD_TIMEOUT", 0.0, 0.0)
                self.state = STATE_IDLE
                self.offboard_setpoint_counter = 0

    # ──────────────────────────────────────────
    # 헬퍼
    # ──────────────────────────────────────────
    def _area_to_z(self, area: float) -> float:
        """bbox 면적(픽셀²) → 목표 NED z 고도"""
        for area_limit, z_target in AREA_ZONES:
            if area < area_limit:
                return z_target
        return AREA_ZONES[-1][1]

    def _publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.timestamp    = self._now_us()
        msg.position     = True
        msg.velocity     = False
        msg.acceleration = False
        msg.attitude     = False
        msg.body_rate    = False
        self.offboard_mode_pub.publish(msg)

    def _publish_setpoint(self):
        msg = TrajectorySetpoint()
        msg.timestamp = self._now_us()
        msg.position  = [self.sp_x, self.sp_y, self.sp_z]
        msg.yaw       = self.sp_yaw if YAW_ENABLE else 0.0
        self.setpoint_pub.publish(msg)

    def _engage_offboard_mode(self):
        self._publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("OFFBOARD mode command sent")
        self._log_event("OFFBOARD_MODE_CMD", 0.0, 0.0)

    def _arm(self):
        self._publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info("ARM command sent")
        self._log_event("ARM_CMD", 0.0, 0.0)

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

    def _log_event(self, event: str, err_x: float, err_y: float):
        self.writer.writerow([
            time.time(), event, self.state,
            round(err_x, 1), round(err_y, 1),
            round(self.sp_x,  4), round(self.sp_y,  4),
            round(self.sp_z,  4), round(math.degrees(self.sp_yaw), 2),
            self.nav_state
        ])
        self.csv.flush()

    def destroy_node(self):
        self.csv.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = OffboardBboxNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()