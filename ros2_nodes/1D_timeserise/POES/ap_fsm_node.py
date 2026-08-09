"""
ap_fsm_node.py  --  cFS LC_action (AP 층), POES 온보드 파이프라인 S2
======================================================================
/wp_results(wp_poes_node) 를 받아 **RPN voting -> 단일 지속성 카운터 -> 2단계 상태기계**
로 PRE_ALERT/ALERT 를 확정하고 /sep_alert 로 발행한다.

cFS 정통 순서를 따른다: watchpoint 불리언 -> RPN 결합 -> 결합 결과에 persistence -> RTS.
  이유 (a) 비행소프트웨어 차용 주장의 일관성, (b) 지속성 카운터가 채널 수가 아니라
  1개로 줄어 온보드에서 더 싸다, (c) 채널 간 교대 초과를 놓치지 않는다(한 채널이 TRUE인
  동안 다른 채널이 TRUE로 바뀌어도 결합 결과는 계속 TRUE라 카운터가 끊기지 않음).
  -> WP 가 발행하는 채널별 watch 불리언(TRUE/FALSE/STALE)만 판정 입력으로 쓴다.
     WP 의 run_len/gate_open/alert_open 은 HK 텔레메트리 참고값일 뿐 판정에 쓰지 않는다
     (WP 는 채널별로 그 값을 냈을 뿐이고, AP 가 그걸 다시 조합해서 쓰면 지속성이 채널
     수만큼 늘어나는 구조가 돼 cFS LC_action 의 "결합 후 persistence" 원칙과 어긋난다).
  단일채널에서는 RPN 결합이 그 채널 값을 그대로 통과시키므로 이 순서와 "채널별 persistence
  후 결합" 순서가 수학적으로 동일하다 -- 그래서 검증 목표값(ALERT 전이 216회, 게이트 개방
  16,204틱)이 WP 의 run_len 재현치와 그대로 일치한다(verify_ap_node.py 검증 ⑥).
  다채널이 되면 두 순서가 갈린다(오프라인은 채널별로 검출한 뒤 클러스터링하는 형태라
  이 AP 결합-후-persistence 방식과 등가가 아니다) -- 논문 3.4절 소재, 코드에도 남겨둔다.

cFS 매핑
  LC ADT               : build_adt() -- RPN 수식 + gate_n/alert_n(2단계 MaxFailsBeforeRTS) +
                          발사할 RTS ID 를 테이블로.
  LC EvaluateRPN        : evaluate_rpn() -- KSEM/ap_fsm_node.py 이식(byte 단위 동일 로직).
                          3-값 논리: OR 결합에서 TRUE 우선, 그 다음 ERROR>STALE>FALSE 전파.
  ConsecutiveFail+MaxFailsBeforeRTS : ApFsmCore.counter (RPN 결합 결과 위 단일 카운터).
  발동 후 PASSIVE(재발사 방지) : ApFsmCore.latched -- ALERT 최초 확정 시 1회만 다운링크.
  LC ACTIVE/PASSIVE     : --dry-run -- latch 만 억제(상태 표시는 그대로, 액추에이션 등가만 억제).
  EVS 재각도+throttle   : EVSThrottle -- 같은 전이가 리셋창 안에서 N회 넘으면 로그만 억제.
  SUCHAI data repository: ApFsmCore.downlink_queue -- 확정 ALERT 만 (onset_ts, alert_ts,
                          trigger_channels, ap_counter_at_alert) 로 적재.

재사용 (재구현 없음): KSEM/ap_fsm_node.py 의 evaluate_rpn 로직(스택/3-값 결합 규칙)을
그대로 이식. 임계 계산은 WP 책임이라 여기선 채널 결과 조합만 한다(원본 docstring 원칙 승계).

구독: /wp_results (wp_poes_node)
발행: /sep_alert (String JSON):
    {"ts": <phys_ts>, "fsm_pub_ts": <wall-clock>, "t_sample_rx": <wp 수신 벽시계, 변경 없이 전달>,
     "ap_state": "NOMINAL"/"PRE_ALERT"/"ALERT",
     "counter": <int>, "trigger_channels": [...], "rpn_result": "TRIGGERED"/"NORMAL"/"STALE"/"ERROR",
     "has_data": <bool>}   # 결측 슬롯 여부(AI 가 게이팅 트리거에 그대로 씀 -- S3 참고)

상태기계 (RPN 결합 결과 위 단일 카운터, S0.5 확정 gate_n=2/alert_n=4 그대로 사용):
  매 틱:  rpn = evaluate_rpn(채널별 watch)
          rpn == TRUE           -> counter += 1
          rpn == FALSE          -> counter = 0 (latch 도 함께 해제)
          rpn in (STALE, ERROR) -> counter 동결(증가도 리셋도 안 함)
    NOMINAL    counter <  gate_n(2)
    PRE_ALERT  counter >= gate_n(2)   가역
    ALERT      counter >= alert_n(4)  latch(다운링크는 최초 확정 1회만). --dry-run 이면 latch 억제.

  STALE 동결은 오프라인 재현을 위한 것이다: 오프라인은 결측 행을 인덱스에서 제거하므로
  공백을 가로질러 run 이 이어진다(S0.5/S1에서 실측 확인). 비행 구현이라면 STALE 지속시간에
  상한을 둬야 하지만(무한정 얼어붙은 카운터는 안전하지 않다), 여기서는 오프라인과의
  비교 가능성이 우선이라 그대로 재현한다 -- 상한을 두는 순간 오프라인 배치와 더 이상
  같은 검출 집합을 내지 않게 되어 md 6절의 "재현이 목표" 원칙과 충돌한다.

누적 카운터(요구사항 6, 종료 시 JSON 저장):
  n_ticks_total 은 실제로 존재하는 리샘플 샘플 수(2019-2025 omni_p6 기준 236,848)로 센다.
  2019-2025 정규 15분 격자(245,472 근방)가 아니다 -- WP 는 그 격자 전체를 틱마다 발행하지만
  (STALE 명시, S1 참고), 결측 슬롯(모든 채널의 count 가 null)은 이 카운터에서 제외한다.
  S0.5 의 duty_cycle 6.84% 가 이 236,848 분모 기준이므로 어긋나면 논문 4.3절이 깨진다.
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path
from time import perf_counter

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    _HAVE_RCLPY = True
except ImportError:                       # Windows 개발 환경 -- 로직 단위 검증용
    _HAVE_RCLPY = False
    rclpy = None
    String = None

    class Node:                            # type: ignore -- 더미 베이스, 인스턴스화 안 함
        pass

RESULT_DIR = os.environ.get("RESULT_DIR", ".")

# ══════════════════════════════════════════════════════
# 파라미터 블록 -- 기본값(argparse 로 런타임 덮어쓰기, cFS TBL 철학)
# ══════════════════════════════════════════════════════
CHANNELS   = "omni_p6"        # WP 와 동일해야 함. 콤마구분.
GATE_N     = 2                # PRE_ALERT: 연속 2샘플(30분) -- S0.5 확정, WP 와 동일 값
ALERT_N    = 4                # ALERT    : 연속 4샘플(45분) -- S0.5 확정, WP 와 동일 값
RTS_ID     = "RTS_SEP_ALERT"

DRY_RUN         = False       # True 면 latch/다운링크만 억제(PASSIVE), 상태 표시는 그대로
EVS_WINDOW_H    = 1.0         # 같은 전이 유형 throttle 리셋창(시간)
EVS_MAX_COUNT   = 5           # 이 창 안에서 N회 넘으면 로그만 억제(누적 카운터는 항상 증가)
EVS_SUMMARY_EVERY_N_TICKS = 96 * 7   # 대략 1주(15분 격자) 마다 억제 요약 로그

SUMMARY_PATH = str(Path(RESULT_DIR) / "ap_fsm_summary.json")
NO_LOG       = False          # True 면 log.csv 기록 생략(S5 계측 오버헤드 A/B 측정용)

# WP 판정값 (wp_poes_node 와 일치)
WATCH_TRUE, WATCH_FALSE, WATCH_STALE, WATCH_ERROR = "TRUE", "FALSE", "STALE", "ERROR"

# RPN 평가 결과 (LC_ACTION_* 대응)
RPN_TRIGGERED, RPN_NORMAL, RPN_STALE, RPN_ERROR = "TRIGGERED", "NORMAL", "STALE", "ERROR"

# AP 운용 상태
STATE_NOMINAL, STATE_PRE_ALERT, STATE_ALERT = "NOMINAL", "PRE_ALERT", "ALERT"


# ══════════════════════════════════════════════════════
# RPN 평가 -- KSEM/ap_fsm_node.py:evaluate_rpn 이식(로직 재구현 아님, byte 단위 동일)
# ══════════════════════════════════════════════════════
def evaluate_rpn(equation: list[str], watch: dict[str, str]) -> tuple[str, list[str]]:
    """RPN voting 평가 (LC_EvaluateRPN 이식).
    equation: 후위표기 토큰 리스트 (채널명 / "AND" "OR" "XOR" "NOT" / "EQUAL")
    watch   : {채널명: "TRUE"/"FALSE"/"STALE"/"ERROR"}

    3-값 논리: OR 은 TRUE 우선, 그 다음 ERROR>STALE>FALSE 전파. AND 는 FALSE 우선,
    그 다음 ERROR>STALE>TRUE 전파(lc_action.c 우선순위 규칙, KSEM 승계).

    반환: (RPN_TRIGGERED/NORMAL/STALE/ERROR, 기여한 TRUE 채널 목록)
    """
    stack: list[str] = []
    trigger_channels: list[str] = []

    def combine_or(a: str, b: str) -> str:
        if a == WATCH_TRUE or b == WATCH_TRUE:
            return WATCH_TRUE
        if WATCH_ERROR in (a, b):
            return WATCH_ERROR
        if WATCH_STALE in (a, b):
            return WATCH_STALE
        return WATCH_FALSE

    def combine_and(a: str, b: str) -> str:
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
        return a

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
            val = watch.get(tok, WATCH_STALE)
            if val == WATCH_TRUE:
                trigger_channels.append(tok)
            stack.append(val)

    if len(stack) != 1:
        return RPN_ERROR, []

    final = stack[0]
    mapping = {WATCH_TRUE: RPN_TRIGGERED, WATCH_FALSE: RPN_NORMAL,
              WATCH_STALE: RPN_STALE, WATCH_ERROR: RPN_ERROR}
    return mapping[final], trigger_channels


def build_rpn_equation(channels: list[str]) -> list[str]:
    """channels 1개 -> [ch, 'EQUAL'] 로 RPN 이 퇴화(단일채널). 여러 개 -> OR 체인(KSEM 승계)."""
    if not channels:
        raise ValueError("channels 비어있음")
    eq: list[str] = [channels[0]]
    for ch in channels[1:]:
        eq.extend([ch, "OR"])
    eq.append("EQUAL")
    return eq


def build_adt(channels: list[str], rpn_equation: list[str], gate_n: int, alert_n: int,
             rts_id: str) -> list[dict]:
    """ADT(ActionPoint Definition Table). RPN 수식 + 2단계 MaxFailsBeforeRTS(gate_n/alert_n)
    + 발사할 RTS ID. 지금은 AP 1개(단일 결합 결과)지만 테이블 구조라 여러 AP 로 확장 가능."""
    return [{
        "ap_name": "AP_SEP_ONSET",
        "channels": list(channels),
        "rpn_equation": list(rpn_equation),
        "gate_n": gate_n,
        "alert_n": alert_n,
        "rts_id": rts_id,
    }]


# ══════════════════════════════════════════════════════
# EVS throttle -- 같은 전이 유형이 리셋창 안에서 N회 넘으면 로그만 억제
# ══════════════════════════════════════════════════════
class EVSThrottle:
    """누적 카운터(항목 6)는 이 클래스와 무관하게 호출부가 항상 증가시킨다.
    여기선 '로그를 찍어도 되는가'만 판단."""

    def __init__(self, reset_window_h: float = EVS_WINDOW_H, max_count: int = EVS_MAX_COUNT):
        self.reset_window_h = reset_window_h
        self.max_count = max_count
        self.history: dict[str, deque] = {}
        self.suppressed_total: dict[str, int] = {}

    def should_log(self, key: str, phys_ts: float) -> bool:
        h = self.history.setdefault(key, deque())
        cutoff = phys_ts - self.reset_window_h * 3600.0
        while h and h[0] < cutoff:
            h.popleft()
        h.append(phys_ts)
        if len(h) > self.max_count:
            self.suppressed_total[key] = self.suppressed_total.get(key, 0) + 1
            return False
        return True


# ══════════════════════════════════════════════════════
# AP 핵심 로직 -- rclpy 무관. Node 는 이 클래스를 감쌀 뿐(md 2절, bench_micro 요구사항).
# ══════════════════════════════════════════════════════
class ApFsmCore:
    """RPN 결합 -> 단일 지속성 카운터 -> 2단계 상태기계 -> latch -> 다운링크 -> 누적 카운터.
    watchpoint 불리언만 입력으로 받는다(WP 의 run_len 은 쓰지 않는다 -- 모듈 docstring 참고)."""

    def __init__(self, adt_entry: dict, dry_run: bool = False,
                evs_window_h: float = EVS_WINDOW_H, evs_max: int = EVS_MAX_COUNT):
        self.adt = adt_entry
        self.dry_run = dry_run
        self.counter = 0
        self.state = STATE_NOMINAL
        self.prev_state = STATE_NOMINAL
        self.latched = False
        self.onset_ts: float | None = None       # 현재 run(counter>0) 시작 시각
        self.evs = EVSThrottle(evs_window_h, evs_max)
        self.downlink_queue: list[dict] = []       # SUCHAI data repository

        self.n_ticks_total = 0
        self.n_ticks_gate_open = 0
        self.n_ticks_alert_open = 0
        self.n_transitions_by_type: dict[str, int] = {}

    def process_tick(self, watch: dict[str, str], counts: dict[str, float | None],
                     phys_ts: float) -> dict:
        rpn_result, trigger_channels = evaluate_rpn(self.adt["rpn_equation"], watch)

        if rpn_result == RPN_TRIGGERED:
            if self.counter == 0:
                self.onset_ts = phys_ts
            self.counter += 1
        elif rpn_result == RPN_NORMAL:
            self.counter = 0
            self.onset_ts = None
            self.latched = False                  # ALERT latch 해제는 rpn==FALSE 일 때
        # STALE/ERROR -> 아무 것도 안 함(동결) -- 오프라인 dropna() 인과 재현

        gate_n, alert_n = self.adt["gate_n"], self.adt["alert_n"]
        if self.counter >= alert_n:
            new_state = STATE_ALERT
        elif self.counter >= gate_n:
            new_state = STATE_PRE_ALERT
        else:
            new_state = STATE_NOMINAL

        from_state = self.state
        self.state = new_state

        # ── SUCHAI data repository: ALERT 최초 확정 시 1회만 다운링크 ──
        if new_state == STATE_ALERT and not self.latched:
            if not self.dry_run:
                self.latched = True
                self.downlink_queue.append({
                    "onset_ts": self.onset_ts,
                    "alert_ts": phys_ts,
                    "trigger_channels": list(trigger_channels),
                    "ap_counter_at_alert": self.counter,
                })
            # dry-run: latch 억제(다운링크 안 함) -- 상태 표시(new_state)는 그대로 ALERT

        # ── 누적 카운터: 실제 존재하는 샘플만(모든 채널 count 가 null 인 결측 슬롯은 제외) ──
        has_data = any(v is not None for v in counts.values())
        if has_data:
            self.n_ticks_total += 1
            if self.counter >= gate_n:
                self.n_ticks_gate_open += 1
            if self.counter >= alert_n:
                self.n_ticks_alert_open += 1

        transitioned = (new_state != from_state)
        should_log = True
        if transitioned:
            ttype = f"{from_state}->{new_state}"
            self.n_transitions_by_type[ttype] = self.n_transitions_by_type.get(ttype, 0) + 1
            should_log = self.evs.should_log(ttype, phys_ts)
        self.prev_state = from_state

        return {
            "rpn_result": rpn_result, "trigger_channels": trigger_channels,
            "from_state": from_state, "state": new_state, "counter": self.counter,
            "transitioned": transitioned, "should_log": should_log,
            "has_data": has_data,   # 결측 슬롯 여부 -- 다운스트림(AI)이 게이팅 판단에 그대로 재사용
        }

    def summary(self) -> dict:
        return {
            "n_ticks_total": self.n_ticks_total,
            "n_ticks_gate_open": self.n_ticks_gate_open,
            "n_ticks_alert_open": self.n_ticks_alert_open,
            "n_transitions_by_type": dict(self.n_transitions_by_type),
            "evs_suppressed_total": dict(self.evs.suppressed_total),
            "n_downlinked_alerts": len(self.downlink_queue),
            "downlink_queue": list(self.downlink_queue),
        }


# ──────────────────────────────────────────────────────
# ApFsmNode -- rclpy 없으면 인스턴스화만 못 함(클래스 정의 자체는 항상 가능)
# ──────────────────────────────────────────────────────
class ApFsmNode(Node):
    def __init__(self):
        super().__init__("ap_fsm_node")
        if String is None:
            raise RuntimeError("rclpy/std_msgs 없음 -- Jetson(ROS2 환경)에서 실행할 것")

        self.sub = self.create_subscription(String, "/wp_results", self.callback, 10)
        self.alert_pub = self.create_publisher(String, "/sep_alert", 10)

        channels = [c.strip() for c in CHANNELS.split(",") if c.strip()]
        rpn_eq = build_rpn_equation(channels)
        if len(channels) == 1:
            self.get_logger().info(
                f"[AP] RPN 퇴화(단일채널={channels[0]}): {rpn_eq} -- "
                f"결합 연산자 없이 그 채널 watch 값을 그대로 통과시킴")
        adt = build_adt(channels, rpn_eq, GATE_N, ALERT_N, RTS_ID)[0]
        self.get_logger().info(f"[AP] ADT: {adt}")

        self.core = ApFsmCore(adt, dry_run=DRY_RUN,
                              evs_window_h=EVS_WINDOW_H, evs_max=EVS_MAX_COUNT)
        self._tick_count = 0
        self._done = False

        # log.csv -- M1/M2/M4(WP 발행 payload 에서 그대로 pass-through) + M3/M5(AP 자체 측정)
        # 를 한 틱 1행으로 합친 파일. AP 가 WP 의 모든 틱을 1:1 로 받으므로 여기가
        # 자연스러운 결합 지점이다(KSEM 의 wp_eval_ms passthrough 관례 승계·확장).
        self.no_log = NO_LOG
        self.csv = None
        self.writer = None
        if not NO_LOG:
            log_path = os.path.join(RESULT_DIR, "log.csv")
            import csv
            self.csv = open(log_path, "w", newline="")
            self.writer = csv.writer(self.csv)
            self.writer.writerow([
                "phys_ts", "ap_pub_wall_ts", "n_channels",
                "resample_ms", "z_eval_ms", "bg_update_ms",           # M1/M2/M4 (WP pass-through)
                "dds_transport_ms", "fsm_eval_ms", "node_latency_ms", # M5/M3/(총)
                "from_state", "ap_state", "rpn_result", "counter", "trigger_channels",
            ])
        else:
            self.get_logger().info("[AP] --no-log: per-tick CSV 기록 생략(계측 오버헤드 측정용)")

        self.marker_path = os.path.join(RESULT_DIR, "REPLAY_DONE")
        self.marker_timer = self.create_timer(1.0, self._check_marker)

        self.get_logger().info(
            f"[AP] ready. gate_n={GATE_N} alert_n={ALERT_N} dry_run={DRY_RUN} "
            f"evs_window_h={EVS_WINDOW_H} evs_max={EVS_MAX_COUNT}. Waiting for /wp_results ...")

    def callback(self, msg):
        t0 = perf_counter()
        t0_wall = time.time()

        payload = json.loads(msg.data)
        phys_ts = payload.get("ts", 0.0)
        results = payload.get("results", [])
        wp_pub_ts = payload.get("wp_pub_ts")
        t_sample_rx = payload.get("t_sample_rx")   # WP 원 수신 벽시계 -- 변경 없이 그대로 전달(S4 M10용)
        n_channels = payload.get("n_channels")
        resample_ms = payload.get("resample_ms")   # M1 (WP pass-through, log.csv 결합용)
        z_eval_ms = payload.get("z_eval_ms")        # M2
        bg_update_ms = payload.get("bg_update_ms")  # M4 (없는 틱이면 None)
        dds_transport_ms = (t0_wall - wp_pub_ts) * 1000.0 if wp_pub_ts else float("nan")

        watch = {r["channel"]: r["watch"] for r in results}
        counts = {r["channel"]: r["count"] for r in results}

        t_fsm = perf_counter()
        out = self.core.process_tick(watch, counts, phys_ts)
        fsm_eval_ms = (perf_counter() - t_fsm) * 1000.0

        alert = {
            "ts": phys_ts, "fsm_pub_ts": time.time(), "t_sample_rx": t_sample_rx,
            "ap_state": out["state"], "counter": out["counter"],
            "trigger_channels": out["trigger_channels"], "rpn_result": out["rpn_result"],
            "has_data": out["has_data"],   # AI 게이팅용 -- 결측 슬롯에서 추론을 트리거하지 않도록
        }
        amsg = String()
        amsg.data = json.dumps(alert)
        self.alert_pub.publish(amsg)

        if out["transitioned"]:
            self._tick_count += 1
            if out["should_log"]:
                mode = "" if not DRY_RUN else " [DRY-RUN]"
                self.get_logger().info(
                    f"[AP]{mode} {out['from_state']}->{out['state']} rpn={out['rpn_result']} "
                    f"counter={out['counter']}/{GATE_N},{ALERT_N} trig={out['trigger_channels']}")
            if self._tick_count % EVS_SUMMARY_EVERY_N_TICKS == 0 and self.core.evs.suppressed_total:
                self.get_logger().info(f"[AP][EVS] 억제 누적: {self.core.evs.suppressed_total}")

        node_latency_ms = (perf_counter() - t0) * 1000.0
        if not self.no_log:
            self.writer.writerow([
                phys_ts, time.time(), n_channels,
                resample_ms, z_eval_ms, bg_update_ms,
                round(dds_transport_ms, 3) if dds_transport_ms == dds_transport_ms else "",
                round(fsm_eval_ms, 4), round(node_latency_ms, 3),
                out["from_state"], out["state"], out["rpn_result"], out["counter"],
                "|".join(out["trigger_channels"]),
            ])
            self.csv.flush()

    def _check_marker(self):
        if self._done or not os.path.exists(self.marker_path):
            return
        self._done = True
        self.get_logger().info("[AP] REPLAY_DONE marker 감지 -> 요약 저장 후 종료")
        self._write_summary()
        if self.csv is not None:
            self.csv.close()
        rclpy.shutdown()

    def _write_summary(self):
        with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
            json.dump(self.core.summary(), f, indent=2)
        self.get_logger().info(f"[AP] summary 저장 -> {SUMMARY_PATH}")

    def destroy_node(self):
        if self.csv is not None and not self.csv.closed:
            self.csv.close()
        super().destroy_node()


def _parse_args(argv=None):
    import argparse
    p = argparse.ArgumentParser(
        description="ap_fsm_node (AP) -- RPN voting + 단일 persistence 카운터, cFS LC_action")
    p.add_argument("--channels", type=str, default=CHANNELS, help="WP 와 동일해야 함")
    p.add_argument("--gate-n", type=int, default=GATE_N, help="PRE_ALERT 지속성. S0.5 확정=2.")
    p.add_argument("--alert-n", type=int, default=ALERT_N, help="ALERT 지속성. S0.5 확정=4.")
    p.add_argument("--rts-id", type=str, default=RTS_ID)
    p.add_argument("--evs-window-h", type=float, default=EVS_WINDOW_H)
    p.add_argument("--evs-max", type=int, default=EVS_MAX_COUNT)
    p.add_argument("--summary-path", type=str, default=SUMMARY_PATH)
    p.add_argument("--no-log", action="store_true",
                   help="log.csv per-tick 기록 생략(S5 계측 오버헤드 A/B 측정용)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--active", dest="active", action="store_true", help="실제 발동(latch). 기본.")
    g.add_argument("--dry-run", dest="active", action="store_false",
                   help="dry-run(PASSIVE): 상태만 표시, latch/다운링크 억제.")
    p.set_defaults(active=True)
    return p.parse_known_args(argv)[0]


def main(argv=None):
    args = _parse_args(argv)

    global CHANNELS, GATE_N, ALERT_N, RTS_ID, DRY_RUN, EVS_WINDOW_H, EVS_MAX_COUNT, SUMMARY_PATH, NO_LOG
    CHANNELS       = args.channels
    GATE_N         = args.gate_n
    ALERT_N        = args.alert_n
    RTS_ID         = args.rts_id
    DRY_RUN        = not args.active
    EVS_WINDOW_H   = args.evs_window_h
    EVS_MAX_COUNT  = args.evs_max
    SUMMARY_PATH   = args.summary_path
    NO_LOG         = args.no_log

    if not _HAVE_RCLPY:
        raise SystemExit("[AP] rclpy 없음 -- 이 노드 실행은 Jetson(ROS2 환경)에서만 가능. "
                         "로직 단위 검증은 ApFsmCore/evaluate_rpn 을 직접 import 해서 할 것.")

    rclpy.init()
    node = ApFsmNode()
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
