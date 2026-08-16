"""
sc_offboard_node.py  --  cFS SC(RTS 층), POES 온보드 파이프라인 S4
======================================================================
/sep_alert(ap_fsm_node) 를 받아 IDLE/TRACKING/HOLD(/LANDING, D3) 로 드론 운용 모드를 전환한다.
KSEM/sc_offboard_node.py 승계: PX4 헬퍼, PASSIVE dry-run, 10Hz 제어루프,
offboard_setpoint_counter>=10 선행 스트림 규약을 그대로 유지한다.

상태 매핑(md 6절 S4 기반, 재발사 가드/DISARM 도입으로 KSEM 원본에서 갱신됨):
  ap_state PRE_ALERT -> pre_streaming=True. setpoint pre-streaming 시작.
      **상태 전이는 없다**(가역 구간) -- PX4 는 Offboard 전환 직전 선행 스트림이
      2Hz 이상 흐르고 있어야 전환을 거부하지 않는다(우주과학회 실증 교훈). 이것이
      2단계 경보(PRE_ALERT/ALERT)의 운용상 근거다: PRE_ALERT 에서 미리 흘려두지
      않으면 ALERT 순간 Offboard 전환 자체가 거부될 수 있다.
  ap_state ALERT     -> IDLE 또는 HOLD -> TRACKING. setpoint 가 충분히(counter>=10)
      쌓였을 때 에피소드당 **정확히 1회만** Offboard 전환 + ARM(재발사 가드,
      command_sent_this_episode). HOLD 중에도 ALERT 가 재도착하면 TRACKING 으로
      복귀해 재ARM 한다(D2, 클러스터 이벤트 대응 -- offboard_setpoint_counter 는
      리셋하지 않는다. HOLD 동안에도 control_loop 가 setpoint 를 끊김 없이
      계속 발행하므로 PX4 의 2Hz 선행 스트림 요건은 이미 충족된 상태다).
      실측 확인(전 구간 causal 리플레이): 재ALERT 는 AP 지속성 카운터가 NOMINAL 에서
      0 으로 완전히 리셋된 뒤 새 연속 4틱(3600/S초, S=배속)이 필요해 HOLD_TIMEOUT
      (배속 무관 실 벽시계 6초)보다 항상 늦게 온다(S<600) -- 이 경로는 실 데이터
      리플레이로는 도달하지 않고 코드/단위테스트로만 방어적으로 검증된다.
  ap_state NOMINAL   -> TRACKING -> HOLD(또는 LANDING), pre_streaming=False. 이
      에피소드에서 실제로 ARM 했었다면(command_sent_this_episode):
        --land-before-disarm(기본 on) : 바로 DISARM 하지 않고 STATE_LANDING 으로
            전이 + VEHICLE_CMD_NAV_LAND(21) 발행(D3). /fmu/out/vehicle_land_detected
            의 landed==true 를 보면 그 자리에서 DISARM 하고 HOLD 로 복귀한다.
            --land-timeout-sec(기본 20초) 안에 착륙이 확인되지 않으면 착륙 미확인
            상태로 강제 DISARM 하고 LAND_TIMEOUT 이벤트를 event_log.csv 에 남긴다.
            착륙-확인 경로와 타임아웃 경로 둘 다 STATE_LANDING 자체를 가드로 써서
            서로 중복 발행하지 않는다(먼저 상태를 HOLD 로 바꾸는 쪽만 발행, 아래
            ScFsmCore.on_land_detected()/tick_control() 참고).
        --no-land-before-disarm       : 구 동작 그대로 -- DISARM 을 이 순간
            (ALERT_CLEARED 직후) 바로 발행하고 HOLD 로 전이한다(D1, 불필요한 ARM
            상태 유지 시간 최소화). HOLD_TIMEOUT 까지 기다리지 않는다.
      ARM 을 이 에피소드에서 한 번도 못 냈으면(가드에 막혔거나 ALERT 가 너무
      짧았음) LANDING/DISARM 없이 곧장 HOLD 로 전이한다(내릴 것도 없다).
  HOLD 는 HOLD_DURATION_SEC*2 경과 후 IDLE 로 자동 복귀(KSEM 그대로).
  재ARM 채터링 가드: 마지막 DISARM 후 MIN_REARM_INTERVAL_SEC(기본 1.0초) 이내에는
      재ARM 하지 않는다(D4). 이 값은 합법적 재ALERT 소요시간(최소 수 초, 위 참고)
      보다 훨씬 짧아 정상 재개는 막지 않고, 글리치/중복메시지 수준의 즉시 재발사만
      억제한다.

  **D3(착륙 시퀀스) 구현 완료**: --land-before-disarm 기본 on. 켜져 있으면 고도
  1.5m 호버에서 그대로 DISARM 하지 않고 위 STATE_LANDING 경로를 거친다.
  --no-land-before-disarm 로 끄면 과거처럼 착륙 없이 즉시 DISARM 하므로 프로펠러
  장착 상태에서 --active 실행 금지 -- 자유낙하한다.
  **주의**: ScFsmCore(...) 를 land_before_disarm 인자 없이 직접 생성하면(예:
  verify_sc_node.py 의 기존 T2~T6) 기본값은 여전히 False(구 동작)다 -- 실행 경로
  (main()/ScOffboardNode)만 모듈 상수 LAND_BEFORE_DISARM(=True)을 항상 명시적으로
  넘긴다. verify_sc_node.py 는 이번 작업 범위(파일 하나만 수정) 밖이라 그 기존
  테스트들은 계속 "즉시 DISARM" 구 경로를 검증하며, 이 값을 명시적으로 넘기지
  않는 한 그대로 통과한다.

  **D3 후속 버그 수정 3건(실측/코드리뷰로 발견)**:
    1) NAV_LAND 의 param4(yaw) 도 param5/6/7 처럼 NaN 이어야 한다 -- 0.0 이면 "정북을
       향해 회전"으로 해석돼 착륙 중 불필요한 요 회전이 생긴다.
    2) on_land_detected()/tick_control() 의 LANDING->HOLD 전이에서 last_alert_time=now
       를 갱신하지 않으면, 착륙이 10~20초 걸리는 동안 HOLD_TIMEOUT 조건(now-
       last_alert_time > hold_duration*2, 기본 6초)이 HOLD 진입 시점에 이미 참이라
       HOLD 단계가 로그에서 사실상 사라진다(즉시 IDLE 로 넘어감). 이제 두 전이 지점
       모두에서 갱신해 착륙 이후에도 HOLD 가 정상적으로 유지된다.
    3) STATE_LANDING 중 재ALERT 는 안전상 무시(명령 없음)하는 게 맞지만, 이벤트를
       하나도 안 남기면 클러스터 이벤트 구간에서 착륙 중 alert 가 왔었다는 사실을
       event_log.csv 로 추적할 수 없었다 -- "ALERT_DURING_LANDING_IGNORED" 관측용
       이벤트를 추가했다(명령은 여전히 내지 않는다).

--setpoint-mode {position, attitude}  기본 position(SITL 은 이 모드로 실증):
  position : 현행. OffboardControlMode.position=True + TrajectorySetpoint(고도
             1.5m 호버). 유효한 로컬 위치 추정(GPS/광류/VIO 등)이 있어야 PX4 가
             Offboard 진입/ARM 을 거부하지 않는다.
  attitude : OffboardControlMode.attitude=True + VehicleAttitudeSetpoint(수평
             자세 + thrust~0). 위치 추정을 요구하지 않아 실내(GPS/광류/VIO 없음)
             프로펠러-탈거 벤치 테스트에 쓴다. SITL 실증 다음 단계인 실내 실기
             테스트에서 사용 예정.

--status-topic(기본 /fmu/out/vehicle_status_v1, 젯슨 ros2 topic list 실측 확인됨 -- 바꾸지
말 것), --status-timeout-sec(기본 5): PX4/px4_msgs 버전에 따라 vehicle_status_v1 대신
vehicle_status 인 경우가 있다. 틀리면 조용히 메시지가 안 와 arming_state 가 0 에
고정되고(진단용 컬럼, 판정에는 안 씀) LANDING 의 착륙확인 대기(vehicle_land_detected 는
별도 토픽이라 직접 영향은 없지만 진단이 막힌다)도 파악하기 어려워진다 -- 기동 후
--status-timeout-sec 안에 이 토픽에서 메시지가 없으면 경고 로그를 남긴다(죽이지 않음).

--bench-thrust(기본 0.0=현행): --setpoint-mode attitude 전용. thrust_body=[0,0,-|값|]
(PX4 body NED 에서 위쪽 추력은 z 음수). 0.0 이면 모터가 안 돌아 "모터가 실제로 도는"
프로펠러-탈거 벤치 시연이 안 보인다 -- 0.1 근처 권장. **프로펠러 장착 상태에서 0 이 아닌
값 사용 금지**(기동 시 경고 로그 남김).

--no-px4(기본 off, 즉 현행처럼 px4_msgs 필수): 켜면 px4_msgs 관련 퍼블리셔/구독자를 아예
만들지 않고 _publish_* 는 전부 no-op 이 된다 -- ScFsmCore 판정과 event_log.csv 기록
(action_ms/e2e_ms 포함)은 그대로다. /sep_alert 구독은 유지(rclpy 만 있으면 됨). 드론/PX4
가 없는 개발 PC 에서 파이프라인 전체(WP/AP/[AI]/SC)를 리허설하기 위한 용도 -- px4_msgs 가
아예 설치 안 돼 있어도(_HAVE_PX4=False) 이 플래그면 죽지 않는다.

판정 로직(ScFsmCore)은 Node 클래스 밖 -- rclpy/px4_msgs 없이 import 가능(md 2절).

--ai-gate {off, shadow, gate}  기본 off:
  off    : /ai_verdict 구독 안 함. 현행 KSEM 동작 그대로(시나리오 b).
  shadow : /ai_verdict 구독해 로그에만 기록. **제어 경로는 off 와 완전 동일**
           (시나리오 c/d 기본 -- ΔPower 가 TCN 비용만 분리되려면 제어 동작이
           같아야 한다). ScFsmCore.on_ai_verdict() 는 last_p_event 만 갱신하고
           state/pre_streaming/counter 는 절대 건드리지 않는다. p_event 가 None
           (ai_tcn_node 의 INSUFFICIENT_DATA)이면 갱신 자체를 건너뛴다 -- 직전
           유효 확률을 지우면 gate 모드의 ARM 조건(last_p_event>=threshold)이
           결측 틱 때문에 부당하게 막힐 수 있다.
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
      (물리시각 ts 와 분리 유지, KSEM 측정/판정 분리 원칙). 실제 PX4 커맨드가
      나가는 모든 이벤트(OFFBOARD_MODE_CMD/ARM_CMD/LAND_CMD/DISARM_CMD)마다
      기록된다(PASSIVE 여도 측정은 그대로, 위 원칙 참고).
      (조사 결과, 2026-08) results/*/event_log.csv 에서 M10 이 n=0 으로 비어
      있던 기존 리소스-벤치 런(e_offboard_*)은 기록 경로 자체의 버그가 아니었다
      -- 그 런들이 가속 리플레이 배속으로 도는 바람에 ALERT 지속시간이, 10Hz
      제어루프가 offboard_setpoint_counter>=10 을 채우는 데 필요한 실 벽시계
      1초보다 짧아서 OFFBOARD_MODE_CMD/ARM_CMD 자체가 한 번도 안 나갔기 때문
      이다(해당 event_log.csv 에 그 두 이벤트 행이 아예 없음, 실측 확인). 1배속
      근처 SITL 실증(--hold-scale 1.0, 기본값)에서는 ALERT 지속시간이 1초를
      넉넉히 넘으므로 이 문제가 없다.
  전이 시각 기록: PRE_ALERT 수신 / ALERT 수신 / Offboard 명령 / ARM 명령 / LAND
      명령 / 착륙확인-또는-타임아웃 DISARM.
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
    from px4_msgs.msg import (VehicleCommand, OffboardControlMode, TrajectorySetpoint, VehicleStatus,
                              VehicleAttitudeSetpoint, VehicleLandDetected)
    _HAVE_PX4 = True
except ImportError:                       # Windows 개발 환경 -- px4_msgs 도 rclpy 와 동일하게 guard
    _HAVE_PX4 = False
    VehicleCommand = OffboardControlMode = TrajectorySetpoint = VehicleStatus = None
    VehicleAttitudeSetpoint = VehicleLandDetected = None

RESULT_DIR = os.environ.get("RESULT_DIR", ".")

# 실 PX4/MAVLink 명령 ID(px4_msgs.VehicleCommand 와 동일 값) -- ScFsmCore 가 px4_msgs
# 없이도 로그/이벤트에 실제 명령 코드를 남길 수 있도록 여기서 상수로 고정.
CMD_DO_SET_MODE       = 176   # VEHICLE_CMD_DO_SET_MODE
CMD_COMPONENT_ARM_DISARM = 400  # VEHICLE_CMD_COMPONENT_ARM_DISARM
CMD_NAV_LAND          = 21    # VEHICLE_CMD_NAV_LAND (D3)

# ══════════════════════════════════════════════════════
# 파라미터 블록 -- 기본값(argparse 로 런타임 덮어쓰기, cFS TBL 철학)
# ══════════════════════════════════════════════════════
PASSIVE          = True    # dry-run: PX4 명령 미발행, action/e2e latency 만 측정. 기본.
AI_GATE_MODE      = "off"   # {off, shadow, gate}
AI_THRESHOLD      = 0.5
HOLD_DURATION_SEC_BASE = 3.0   # KSEM 원값(초, 실시간/1배속 기준)
HOLD_SCALE        = 1.0        # 가속 리플레이용 축소 배수. 1.0=KSEM 그대로.
MIN_REARM_INTERVAL_SEC = 1.0   # D4=(a): 마지막 DISARM 후 이 시간(실 벽시계) 이내 재ARM 금지
                                 # (채터링 가드). 근거는 모듈 docstring 참고 -- 합법적 재ALERT
                                 # 소요시간(수 초 이상)보다 훨씬 짧아 정상 재개는 막지 않는다.
LAND_BEFORE_DISARM = True      # D3: ALERT_CLEARED 시 즉시 DISARM 대신 NAV_LAND -> 착륙확인/
                                 # 타임아웃 후 DISARM. 기본 on(SITL 에서 자유낙하 방지).
                                 # ScFsmCore 자체 기본값은 False -- 모듈 docstring 의 "주의" 참고.
LAND_TIMEOUT_SEC   = 20.0      # LANDING 상태에서 착륙확인(vehicle_land_detected) 대기 최대
                                 # 시간(초, 실 벽시계). 초과 시 강제 DISARM + LAND_TIMEOUT 기록.
SETPOINT_MODE      = "position"  # {position, attitude}
STATUS_TOPIC       = "/fmu/out/vehicle_status_v1"   # 젯슨 ros2 topic list 실측 확인됨 -- 바꾸지 말 것
STATUS_TIMEOUT_SEC = 5.0       # 기동 후 STATUS_TOPIC 에서 메시지가 없으면 경고를 남기기까지 대기(초)
BENCH_THRUST       = 0.0       # attitude 모드 thrust_body z 크기(부호는 코드가 처리). 기본 0.0=현행
                                 # (모터 안 돎). 프로펠러 장착 상태에서 0 이 아닌 값 금지.
NO_PX4             = False     # True 면 px4_msgs 퍼블리셔/구독자를 만들지 않음(_publish_* no-op).
                                 # 판정/로깅은 그대로 -- 드론 없는 개발 PC 리허설용.

# 상태 정의 (KSEM 과 동일 + LANDING(D3) 신규)
STATE_IDLE, STATE_TRACKING, STATE_HOLD, STATE_LANDING = "IDLE", "TRACKING", "HOLD", "LANDING"
AP_NOMINAL, AP_PRE_ALERT, AP_ALERT = "NOMINAL", "PRE_ALERT", "ALERT"


# ══════════════════════════════════════════════════════
# SC 핵심 로직 -- rclpy/px4_msgs 무관. Node 는 이 클래스를 감쌀 뿐(md 2절).
# ══════════════════════════════════════════════════════
class ScFsmCore:
    """IDLE/TRACKING/HOLD/LANDING(D3) 상태기계 + pre_streaming + offboard_setpoint_counter.
    PASSIVE 여부까지 여기서 판정(진짜 PX4 발행이 필요한 부분은 Node 가 명령 목록을
    보고 실행만 한다) -- 그래야 rclpy/px4_msgs 없이도 PASSIVE_SUPPRESS 를 검증할 수 있다."""

    def __init__(self, ai_gate_mode: str = AI_GATE_MODE, ai_threshold: float = AI_THRESHOLD,
                hold_duration_sec: float = HOLD_DURATION_SEC_BASE, passive: bool = PASSIVE,
                min_rearm_interval_sec: float = MIN_REARM_INTERVAL_SEC,
                land_before_disarm: bool = False,   # 주의: 모듈 상수 LAND_BEFORE_DISARM(True)과
                                                     # 다르다. verify_sc_node.py 의 기존
                                                     # T2~T6(이 인자 생략)가 "즉시 DISARM" 구
                                                     # 경로를 계속 검증하도록 여기 기본은 False 로
                                                     # 남긴다 -- 실행 경로(main()/ScOffboardNode)만
                                                     # 항상 명시적으로 True 를 넘긴다(모듈 docstring
                                                     # "주의" 항목 참고).
                land_timeout_sec: float = LAND_TIMEOUT_SEC):
        self.ai_gate_mode = ai_gate_mode
        self.ai_threshold = ai_threshold
        self.hold_duration_sec = hold_duration_sec
        self.passive = passive
        self.min_rearm_interval_sec = min_rearm_interval_sec
        self.land_before_disarm = land_before_disarm
        self.land_timeout_sec = land_timeout_sec

        self.state = STATE_IDLE
        self.pre_streaming = False
        self.offboard_setpoint_counter = 0
        self.last_alert_time = 0.0
        self.last_action_ms = float("nan")
        self.last_t_sample_rx: float | None = None
        self.last_p_event: float | None = None

        # 재발사 가드(2-1) + DISARM(2-2) 상태
        self.command_sent_this_episode = False   # 이 TRACKING 에피소드에서 OFFBOARD+ARM 을 이미 냈는가
        self.last_disarm_time: float | None = None   # D4 재ARM 채터링 가드용
        self.land_deadline: float | None = None   # D3: STATE_LANDING 진입 시각 + land_timeout_sec

        self.commands_sent: list[str] = []         # ACTIVE 로 실제 "발행"된 명령(시뮬레이션 기록)
        self.commands_suppressed: list[str] = []   # PASSIVE 로 억제된 명령

    def _emit_cmd(self, name: str, events: list[str]) -> None:
        """PASSIVE 여부에 따라 commands_sent/commands_suppressed 중 한쪽에 기록하고 events 에도
        추가한다(OFFBOARD/ARM/DISARM/LAND 4곳에서 반복되던 패턴을 한 곳으로 모음). 실제 PX4
        발행 여부는 Node 가 self.core.passive 를 보고 독립적으로 다시 판단한다 -- 여기 기록은
        검증/관측용(commands_sent/suppressed)일 뿐이다."""
        if self.passive:
            self.commands_suppressed.append(name)
        else:
            self.commands_sent.append(name)
        events.append(name)

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
            # D2=(b): HOLD 중에도 ALERT 재도착 시 TRACKING 복귀(클러스터 이벤트 대응).
            # IDLE 과 HOLD 둘 다 허용 -- TRACKING 자체(이미 추적 중)는 재진입 대상이 아니다.
            if self.state in (STATE_IDLE, STATE_HOLD):
                if allowed:
                    self.state = STATE_TRACKING
                    events.append("ALERT_TRIGGER")
                    # 새 에피소드 시작 -- 재발사 가드 리셋(counter 는 건드리지 않는다.
                    # HOLD 동안에도 setpoint 스트리밍이 안 끊겼으므로 PX4 의 2Hz 선행
                    # 스트림 요건은 이미 충족돼 있다 -- 0 으로 되돌리면 불필요하게
                    # 1초를 더 기다리게 만들 뿐이다).
                    self.command_sent_this_episode = False
                else:
                    events.append("ALERT_GATED_SUPPRESS")   # gate 모드 전용 이벤트
            elif self.state == STATE_LANDING:
                # 버그4: 착륙 중 재ALERT 는 안전상 무시(명령 없음)하지만 관측용 이벤트는
                # 남긴다 -- 그래야 클러스터 이벤트 구간에서 착륙 중 alert 가 왔었다는 사실이
                # event_log.csv 에 남는다.
                events.append("ALERT_DURING_LANDING_IGNORED")

        elif ap_state == AP_NOMINAL:
            self.pre_streaming = False
            if self.state == STATE_TRACKING:
                events.append("ALERT_CLEARED")
                # 이 에피소드에서 실제로 ARM 했을 때만(가드에 막혀 한 번도 못 냈으면
                # 내릴 것도 없다) LANDING/DISARM 을 발행 + 중복 방지.
                if self.command_sent_this_episode:
                    if self.land_before_disarm:
                        # D3: 즉시 DISARM 대신 NAV_LAND 발행 후 착륙확인/타임아웃을 기다린다
                        # (on_land_detected()/tick_control() 의 STATE_LANDING 분기가 이어받음).
                        self.state = STATE_LANDING
                        self.land_deadline = now + self.land_timeout_sec
                        self._emit_cmd("LAND_CMD", events)
                    else:
                        # 구 동작: HOLD_TIMEOUT 까지 기다리지 않고 이 순간 DISARM(D1=(a),
                        # 불필요한 ARM 유지시간 최소화).
                        self._emit_cmd("DISARM_CMD", events)
                        self.last_disarm_time = now
                        self.state = STATE_HOLD
                else:
                    self.state = STATE_HOLD
                self.command_sent_this_episode = False

        return events

    def on_ai_verdict(self, p_event) -> None:
        """shadow/gate 모드에서만 호출된다(off 는 구독 자체를 안 함). last_p_event 만
        갱신 -- state/pre_streaming/offboard_setpoint_counter 는 절대 건드리지 않는다.
        그래야 shadow 의 제어 경로가 off 와 완전히 같다(검증 ③).
        p_event 가 None(ai_tcn_node 가 INSUFFICIENT_DATA 로 발행)이면 갱신을 건너뛴다 --
        그대로 저장하면 직전 유효 확률이 지워져, gate 모드의 ARM 조건(last_p_event>=
        threshold)이 결측 틱 때문에 부당하게 막힐 수 있다."""
        if p_event is not None:
            self.last_p_event = p_event

    def on_land_detected(self, landed: bool, now: float) -> list[str]:
        """/fmu/out/vehicle_land_detected 콜백(D3). landed=True 이고 아직 STATE_LANDING
        이면 DISARM 후 HOLD 로 복귀한다. state==STATE_LANDING 자체가 가드 -- 이 경로와
        tick_control() 의 타임아웃 경로 중 먼저 도달해 state 를 HOLD 로 바꾸는 쪽만
        발행하므로 중복 DISARM 이 나지 않는다."""
        events: list[str] = []
        if landed and self.state == STATE_LANDING:
            events.append("LANDED_CONFIRMED")
            self._emit_cmd("DISARM_CMD", events)
            self.last_disarm_time = now
            self.state = STATE_HOLD
            # 버그3: 착륙에 10~20초가 걸리면 HOLD 진입 시점에 이미 now-last_alert_time 이
            # hold_duration*2 를 넘어 있어 HOLD 가 로그에서 사라진다 -- HOLD 진입을 새
            # 기준시각으로 갱신해 착륙 이후에도 HOLD 가 정상적으로 유지되게 한다.
            self.last_alert_time = now
        return events

    def tick_control(self, now: float) -> list[str]:
        """10Hz 제어루프 1회. TRACKING 이고 counter>=10 이면 **에피소드당 1회만**
        Offboard+ARM 명령을 낸다(command_sent_this_episode 가드, 2-1 -- KSEM 원본은
        재발사 방지 가드가 없어 매 틱 재발사했으나 이제 막는다). 재ARM 은 추가로
        D4 채터링 가드(min_rearm_interval_sec)도 통과해야 한다.
        반환: 이번 틱 발생 이벤트."""
        events: list[str] = []
        if self.state == STATE_IDLE:
            if self.pre_streaming:
                self.offboard_setpoint_counter += 1
            else:
                self.offboard_setpoint_counter = 0

        elif self.state == STATE_TRACKING:
            if self.offboard_setpoint_counter >= 10 and not self.command_sent_this_episode:
                guard_ok = (self.last_disarm_time is None
                           or (now - self.last_disarm_time) >= self.min_rearm_interval_sec)
                if guard_ok:
                    for name in ("OFFBOARD_MODE_CMD", "ARM_CMD"):
                        self._emit_cmd(name, events)
                    self.command_sent_this_episode = True
                else:
                    events.append("ARM_REARM_GUARD_SUPPRESS")   # D4 -- 관측용(명령 미발행)
            self.offboard_setpoint_counter += 1

        elif self.state == STATE_HOLD:
            if now - self.last_alert_time > self.hold_duration_sec * 2:
                self.state = STATE_IDLE
                self.offboard_setpoint_counter = 0
                self.pre_streaming = False
                self.command_sent_this_episode = False   # 에피소드 종료 시점 명시(이미 False 인 게 정상)
                events.append("HOLD_TIMEOUT")

        elif self.state == STATE_LANDING:
            # D3: 착륙확인(on_land_detected)이 --land-timeout-sec 안에 안 오면 착륙 미확인
            # 상태로 강제 DISARM(state==STATE_LANDING 가드로 on_land_detected() 와의 중복
            # 발행을 막는다 -- 자세한 설명은 on_land_detected() 참고).
            if self.land_deadline is not None and now >= self.land_deadline:
                events.append("LAND_TIMEOUT")
                self._emit_cmd("DISARM_CMD", events)
                self.last_disarm_time = now
                self.state = STATE_HOLD
                self.last_alert_time = now   # 버그3 -- on_land_detected() 와 동일 이유

        return events


# ──────────────────────────────────────────────────────
# ScOffboardNode -- rclpy 없으면 인스턴스화 자체가 안 됨. px4_msgs 는 --no-px4 없이만 필수.
# ──────────────────────────────────────────────────────
class ScOffboardNode(Node):
    def __init__(self):
        super().__init__("sc_offboard_node")
        if String is None:
            raise RuntimeError("rclpy/std_msgs 없음 -- Jetson(ROS2 환경)에서 실행할 것")

        self.no_px4 = NO_PX4
        if not self.no_px4 and not _HAVE_PX4:
            raise RuntimeError("px4_msgs 없음 -- PX4 SITL/실기 연결 환경에서만 실행 가능"
                               "(px4_msgs 없이 파이프라인만 리허설하려면 --no-px4)")

        self.status_topic = STATUS_TOPIC
        self.setpoint_mode = SETPOINT_MODE
        self.bench_thrust = BENCH_THRUST
        if self.setpoint_mode == "attitude" and self.bench_thrust != 0.0:
            self.get_logger().warn(
                f"[SC] --bench-thrust={self.bench_thrust} != 0 -- 모터가 실제로 돕니다. "
                f"!! 프로펠러 장착 상태에서 이 옵션으로 실행 금지 !! 탈거 벤치 테스트 전용.")

        self.alert_sub = self.create_subscription(String, "/sep_alert", self.on_alert, 10)
        self.verdict_sub = None
        if AI_GATE_MODE != "off":   # std_msgs/String 뿐 -- px4_msgs 무관, no_px4 와 독립
            self.verdict_sub = self.create_subscription(String, "/ai_verdict", self.on_verdict, 10)

        # px4_msgs 타입을 실제로 만지는 구독/발행은 전부 여기 안에서만 -- --no-px4 면
        # px4_msgs 가 아예 미설치(_HAVE_PX4=False)라도 이 블록에 안 들어가므로 안전하다.
        self.status_sub = None
        self.land_detected_sub = None
        self.cmd_pub = None
        self.offboard_mode_pub = None
        self.setpoint_pub = None
        self.attitude_setpoint_pub = None
        if not self.no_px4:
            px4_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )
            self.status_sub = self.create_subscription(
                VehicleStatus, self.status_topic, self.on_status, px4_qos)
            if LAND_BEFORE_DISARM:
                self.land_detected_sub = self.create_subscription(
                    VehicleLandDetected, "/fmu/out/vehicle_land_detected", self.on_land_detected, px4_qos)
            self.cmd_pub = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", px4_qos)
            self.offboard_mode_pub = self.create_publisher(
                OffboardControlMode, "/fmu/in/offboard_control_mode", px4_qos)
            if self.setpoint_mode == "attitude":
                self.attitude_setpoint_pub = self.create_publisher(
                    VehicleAttitudeSetpoint, "/fmu/in/vehicle_attitude_setpoint", px4_qos)
            else:
                self.setpoint_pub = self.create_publisher(
                    TrajectorySetpoint, "/fmu/in/trajectory_setpoint", px4_qos)

        # 모듈 상수(LAND_BEFORE_DISARM=True 기본)를 항상 명시적으로 넘긴다 -- ScFsmCore 자체
        # 기본값은 False(구 동작) 라서 여기서 넘기지 않으면 착륙 시퀀스가 켜지지 않는다
        # (모듈 docstring "주의" 항목, ScFsmCore.__init__ 주석 참고). --no-px4 여도 판정
        # 로직은 그대로 돈다(LANDING 도 진입하지만 landed 확인 콜백이 없으니 항상
        # --land-timeout-sec 경로로 HOLD 에 도달한다).
        self.core = ScFsmCore(AI_GATE_MODE, AI_THRESHOLD,
                              HOLD_DURATION_SEC_BASE * HOLD_SCALE, PASSIVE,
                              MIN_REARM_INTERVAL_SEC,
                              land_before_disarm=LAND_BEFORE_DISARM,
                              land_timeout_sec=LAND_TIMEOUT_SEC)
        self.nav_state = 0
        self.arming_state = 0

        # Q3: STATUS_TOPIC 버전 확인용 -- N초 안에 첫 메시지가 안 오면 경고(죽이지 않음).
        # --no-px4 면 애초에 구독이 없으므로 _check_status_topic_timeout() 안에서 스킵한다.
        self._status_received = False
        self._status_warned = False
        self._node_start_time = time.time()

        log_path = os.path.join(RESULT_DIR, "event_log.csv")
        import csv
        self.csv = open(log_path, "w", newline="")
        self.writer = csv.writer(self.csv)
        self.writer.writerow([
            "timestamp", "event", "state", "ap_state", "nav_state",
            "action_ms", "e2e_ms", "passive", "ai_gate_mode",
            "arming_state",   # 끝에 추가(Q5 확인: aggregate_onboard.py 는 열 이름으로 읽음,
                              # 진단용 -- ARM/DISARM 판정에는 쓰지 않는다(core 는 rclpy 모름)
            "no_px4",   # 끝에 추가(기존 열 순서/이름 불변) -- --no-px4 리허설 로그와 실제
                       # PX4 실기/SITL 로그를 results/ 에서 구분하기 위함. passive 와 같은 0/1.
        ])
        self.get_logger().info(f"[SC] Logging to {log_path}")
        self.get_logger().info(
            f"[SC] ai_gate={AI_GATE_MODE} passive={PASSIVE} setpoint_mode={self.setpoint_mode} "
            f"land_before_disarm={LAND_BEFORE_DISARM}(timeout={LAND_TIMEOUT_SEC:.0f}s) "
            f"status_topic={self.status_topic} no_px4={self.no_px4} "
            f"hold_duration={self.core.hold_duration_sec:.2f}s(scale={HOLD_SCALE}) -- "
            f"가속 리플레이에서 HOLD/IDLE 순환 자체는 물리적 대표성 없음. "
            f"transport/action/M10 만 유의미.")
        if self.no_px4:
            self.get_logger().warn("[SC] --no-px4: PX4 미연결 -- 명령은 기록만 됨"
                                   "(ScFsmCore 판정 + event_log.csv 는 정상 동작).")

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
            if ev == "DISARM_CMD":
                self._disarm()   # 내부에서 _log_event 도 함께 처리(_arm() 과 동일 패턴)
            elif ev == "LAND_CMD":
                self._land()     # D3 -- 마찬가지로 내부에서 _log_event 처리
            else:
                self._log_event(ev, ap_state)

    def on_verdict(self, msg):
        try:
            v = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self.core.on_ai_verdict(v.get("p_event"))

    def on_status(self, msg):
        self._status_received = True   # Q3: 첫 메시지 확인용
        self.arming_state = msg.arming_state
        self.nav_state = msg.nav_state

    def on_land_detected(self, msg):
        # LAND_BEFORE_DISARM=True 일 때만 구독(__init__ 참고).
        now = time.time()
        events = self.core.on_land_detected(bool(msg.landed), now)
        for ev in events:
            if ev == "DISARM_CMD":
                self._disarm()
            else:
                self._log_event(ev, AP_NOMINAL)

    # ── 10Hz 제어 루프 ──
    def control_loop(self):
        now = time.time()
        self._check_status_topic_timeout(now)
        self._publish_offboard_control_mode()
        self._publish_hover_setpoint()

        events = self.core.tick_control(now)
        for ev in events:
            if ev == "OFFBOARD_MODE_CMD":
                self._engage_offboard_mode()
            elif ev == "ARM_CMD":
                self._arm()
            elif ev == "DISARM_CMD":
                self._disarm()   # D3 -- LANDING 타임아웃 강제 DISARM(land-before-disarm off 면 안 씀)
            elif ev == "LAND_TIMEOUT":
                self.get_logger().warn(
                    f"[SC] Landing not confirmed within {self.core.land_timeout_sec:.1f}s -- forcing DISARM.")
                self._log_event("LAND_TIMEOUT", AP_NOMINAL)
            elif ev == "HOLD_TIMEOUT":
                self.get_logger().info("[SC] Hold timeout. HOLD -> IDLE")
                self._log_event("HOLD_TIMEOUT", AP_NOMINAL)
            elif ev == "ARM_REARM_GUARD_SUPPRESS":
                # D4 -- 명령 미발행, 관측용 로그만(채터링 가드가 실제로 얼마나 발동하는지 확인용)
                self._log_event("ARM_REARM_GUARD_SUPPRESS", AP_ALERT)

    def _check_status_topic_timeout(self, now: float):
        """Q3: status_topic 이름이 이 PX4/px4_msgs 버전과 안 맞으면 조용히 메시지가 안 와
        arming_state 가 0 에 고정된다 -- 기동 후 STATUS_TIMEOUT_SEC 안에 메시지가 없으면
        경고만 남기고 계속 진행한다(죽이지 않음). --no-px4 면 애초에 status_sub 자체가
        없어 이 경고가 항상 오탐이므로 스킵한다."""
        if self.no_px4:
            return
        if self._status_received or self._status_warned:
            return
        if now - self._node_start_time >= STATUS_TIMEOUT_SEC:
            self._status_warned = True
            self.get_logger().warn(
                f"[SC] '{self.status_topic}' 에서 {STATUS_TIMEOUT_SEC:.0f}초 안에 메시지가 오지 "
                f"않음 -- arming_state/nav_state 가 0 에 고정됨(진단용 컬럼, 판정에는 안 씀). "
                f"PX4/px4_msgs 버전에 따라 토픽명이 다를 수 있다(예: /fmu/out/vehicle_status). "
                f"--status-topic 으로 실제 토픽명을 지정할 것.")

    # ── PX4 퍼블리시 헬퍼 (--no-px4 면 전부 no-op) ──
    def _publish_offboard_control_mode(self):
        if self.no_px4:
            return
        msg = OffboardControlMode()
        msg.timestamp = self._now_us()
        is_attitude = (self.setpoint_mode == "attitude")
        msg.position = not is_attitude
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = is_attitude
        msg.body_rate = False
        self.offboard_mode_pub.publish(msg)

    def _publish_hover_setpoint(self):
        if self.setpoint_mode == "attitude":
            self._publish_attitude_setpoint()
        else:
            self._publish_position_setpoint()

    def _publish_position_setpoint(self):
        # D3 구현 후: LAND_BEFORE_DISARM=True(기본) 면 이 고도(1.5m)에서 그대로 DISARM 하지
        # 않고 NAV_LAND -> 착륙확인/타임아웃 경로를 거친다(_land()/on_land_detected() 참고).
        # --no-land-before-disarm 이면 과거처럼 이 고도에서 즉시 DISARM 하므로 프로펠러
        # 장착 상태에서 --active 실행 금지 -- 자유낙하한다.
        if self.no_px4:
            return
        msg = TrajectorySetpoint()
        msg.timestamp = self._now_us()
        msg.position = [0.0, 0.0, -1.5]
        msg.yaw = 0.0
        self.setpoint_pub.publish(msg)

    def _publish_attitude_setpoint(self):
        # --setpoint-mode attitude: 로컬 위치 추정이 없어도(실내, GPS/광류/VIO 없음) Offboard
        # 진입/ARM 이 거부되지 않도록 수평 자세만 낸다(프로펠러-탈거 벤치 테스트용).
        # 필드명(q_d/thrust_body)은 px4_msgs 표준 정의 기준 -- 실행 전 대상 머신에서
        # `ros2 interface show px4_msgs/msg/VehicleAttitudeSetpoint` 로 확인 권장.
        if self.no_px4:
            return
        msg = VehicleAttitudeSetpoint()
        msg.timestamp = self._now_us()
        msg.q_d = [1.0, 0.0, 0.0, 0.0]   # 단위 쿼터니언(w,x,y,z) -- 회전 없음, 수평 자세
        # 버그2: PX4 body NED 에서 위쪽 추력은 z 음수. 기본 --bench-thrust=0.0(현행, 모터
        # 안 돎) -- 0.1 근처를 주면 "모터가 실제로 도는" 벤치 시연이 보인다. abs() 는
        # 부호를 실수로 반대로 줘도 항상 위쪽(음수)만 나가게 하는 방어.
        # !! 프로펠러 장착 상태에서 0 이 아닌 값 금지(__init__ 기동 경고 참고) !!
        msg.thrust_body = [0.0, 0.0, -abs(self.bench_thrust)]
        self.attitude_setpoint_pub.publish(msg)

    def _compute_e2e_ms(self) -> float:
        t_rx = self.core.last_t_sample_rx
        return (time.time() - t_rx) * 1000.0 if t_rx is not None else float("nan")

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

    def _disarm(self):
        # D3 구현 후: land_before_disarm=True(기본) 면 이 호출은 착륙확인(on_land_detected)
        # 또는 --land-timeout-sec 타임아웃 뒤에만 온다. land_before_disarm=False 면 과거처럼
        # ALERT_CLEARED 직후(착륙 없이) 바로 온다 -- 이때 프로펠러 장착 상태에서 --active
        # 실행 금지, 자유낙하한다.
        e2e_ms = self._compute_e2e_ms()
        self._publish_vehicle_command(CMD_COMPONENT_ARM_DISARM, param1=0.0)
        if not self.core.passive:
            self.get_logger().info("[SC] DISARM command sent")
        self._log_event("DISARM_CMD" if not self.core.passive else "PASSIVE_SUPPRESS_CMD_DISARM",
                        AP_NOMINAL, e2e_ms)

    def _land(self):
        # D3: VEHICLE_CMD_NAV_LAND(21) -- PX4 를 AUTO_LAND 로 전환해 제자리 착륙시킨다.
        # 실제 DISARM 은 on_land_detected(landed=True) 또는 --land-timeout-sec 타임아웃에서.
        # 버그1: param4(yaw)/param5/6/7(lat/lon/alt) 은 반드시 NaN 이어야 "현재 헤딩 유지 +
        # 현재 위치에서 착륙"으로 해석된다 -- 기본값 0.0 을 그대로 두면 PX4 가 param4=0.0 을
        # "정북을 향해 회전"으로, param5/6=0.0 을 위경도 0,0(적도/본초자오선)으로 오인할 수 있다.
        e2e_ms = self._compute_e2e_ms()
        self._publish_vehicle_command(CMD_NAV_LAND, param4=float("nan"),
                                      param5=float("nan"), param6=float("nan"), param7=float("nan"))
        if not self.core.passive:
            self.get_logger().info("[SC] NAV_LAND command sent -- waiting for vehicle_land_detected")
        self._log_event("LAND_CMD" if not self.core.passive else "PASSIVE_SUPPRESS_CMD_LAND",
                        AP_NOMINAL, e2e_ms)

    def _publish_vehicle_command(self, command, param1=0.0, param2=0.0, param4=0.0,
                                 param5=0.0, param6=0.0, param7=0.0):
        # --no-px4: 명령 자체를 만들지 않는다(px4_msgs 가 미설치일 수도 있음).
        if self.no_px4:
            return
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
        msg.param4 = param4   # NAV_LAND 의 yaw -- _land() 가 NaN 으로 넘겨 "현재 헤딩 유지" 지정
        msg.param5 = param5   # NAV_LAND 의 lat -- _land() 가 NaN 으로 넘겨 "현재 위치" 지정
        msg.param6 = param6   # NAV_LAND 의 lon -- 위와 동일
        msg.param7 = param7   # NAV_LAND 의 alt -- 위와 동일(NaN=현재 하강 설정 사용)
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
            self.arming_state,   # 진단용(on_status() 수신값 그대로) -- 판정에는 안 씀
            int(self.no_px4),   # 리허설(PX4 미연결) 로그 구분용 -- passive 와 같은 방식(0/1)
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

    gl = p.add_mutually_exclusive_group()
    gl.add_argument("--land-before-disarm", dest="land_before_disarm", action="store_true",
                    help="ALERT_CLEARED 시 NAV_LAND 발행 후 착륙확인/타임아웃 뒤 DISARM. 기본.")
    gl.add_argument("--no-land-before-disarm", dest="land_before_disarm", action="store_false",
                    help="구 동작: 착륙 없이 즉시 DISARM(주의: 호버 고도에서 자유낙하).")
    p.set_defaults(land_before_disarm=LAND_BEFORE_DISARM)
    p.add_argument("--land-timeout-sec", type=float, default=LAND_TIMEOUT_SEC,
                   help=f"착륙확인(vehicle_land_detected) 대기 최대 시간(초). "
                        f"기본 {LAND_TIMEOUT_SEC:.0f}. 초과 시 착륙 미확인 상태로 강제 DISARM.")
    p.add_argument("--setpoint-mode", choices=["position", "attitude"], default=SETPOINT_MODE,
                   help="position(기본): TrajectorySetpoint, 로컬 위치 추정 필요(SITL 용). "
                        "attitude: VehicleAttitudeSetpoint(수평+thrust~0), 위치 추정 불요 "
                        "(실내/프로펠러-탈거 벤치 테스트용).")
    p.add_argument("--status-topic", default=STATUS_TOPIC,
                   help=f"VehicleStatus 구독 토픽. PX4 버전에 따라 vehicle_status_v1 대신 "
                        f"vehicle_status 일 수 있음. 기본 {STATUS_TOPIC}.")
    p.add_argument("--status-timeout-sec", type=float, default=STATUS_TIMEOUT_SEC,
                   help=f"기동 후 --status-topic 에서 메시지가 안 오면 경고를 남기기까지 "
                        f"대기 시간(초). 기본 {STATUS_TIMEOUT_SEC:.0f}.")
    p.add_argument("--bench-thrust", type=float, default=BENCH_THRUST,
                   help=f"--setpoint-mode attitude 전용. thrust_body z 크기(부호는 코드가 "
                        f"처리, 위쪽=음수). 기본 {BENCH_THRUST}(현행, 모터 안 돎). 0.1 근처 "
                        f"권장 -- 프로펠러 장착 상태에서 0 이 아닌 값 사용 금지.")
    p.add_argument("--no-px4", dest="no_px4", action="store_true",
                   help="px4_msgs 퍼블리셔/구독자를 만들지 않고 명령 발행을 전부 no-op 으로 "
                        "만든다(판정/event_log.csv 기록은 정상 동작). 드론/PX4 없는 개발 PC "
                        "에서 파이프라인 리허설용. 기본 off(현행처럼 px4_msgs 필수).")
    return p.parse_known_args(argv)[0]


def main(argv=None):
    args = _parse_args(argv)
    global PASSIVE, AI_GATE_MODE, AI_THRESHOLD, HOLD_SCALE
    global LAND_BEFORE_DISARM, LAND_TIMEOUT_SEC, SETPOINT_MODE, STATUS_TOPIC, STATUS_TIMEOUT_SEC
    global BENCH_THRUST, NO_PX4
    PASSIVE       = args.passive
    AI_GATE_MODE  = args.ai_gate
    AI_THRESHOLD  = args.ai_threshold
    HOLD_SCALE    = args.hold_scale
    LAND_BEFORE_DISARM = args.land_before_disarm
    LAND_TIMEOUT_SEC   = args.land_timeout_sec
    SETPOINT_MODE       = args.setpoint_mode
    STATUS_TOPIC        = args.status_topic
    STATUS_TIMEOUT_SEC  = args.status_timeout_sec
    BENCH_THRUST        = args.bench_thrust
    NO_PX4              = args.no_px4

    if not _HAVE_RCLPY:
        raise SystemExit("[SC] rclpy 없음 -- 이 노드 실행은 Jetson(ROS2 환경)에서만 가능. "
                         "로직 단위 검증은 ScFsmCore 를 직접 import 해서 할 것.")
    if not NO_PX4 and not _HAVE_PX4:
        raise SystemExit("[SC] px4_msgs 없음 -- PX4 SITL/실기 연결 환경에서만 실행 가능. "
                         "px4_msgs 없이 파이프라인만 리허설하려면 --no-px4.")

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
