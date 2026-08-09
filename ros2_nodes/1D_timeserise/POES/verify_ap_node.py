"""
verify_ap_node.py
==================
S2 로직 단위 검증 (Windows, rclpy 없이). 읽기 전용, C_PD 아래 아무것도 쓰지 않는다.

검증 6종 (md 요청 그대로):
  1. rclpy 없이 ap_fsm_node import 성공.
  2. WP 출력을 메모리로 먹여 전 구간 리플레이: ALERT 전이 216회, 게이트 개방 틱 16,204개.
     (15,235 는 세그먼트/onset 수, 16,204 는 틱 수 -- 다른 지표, 혼동하지 않는다.)
  3. 3-값 논리 단위 테스트: STALE/ERROR 가 섞인 OR 결합이 규칙대로 나오는지.
  4. latch: ALERT 후 같은 run 안에서 재발화(다운링크 중복) 억제, --dry-run 에서 latch 안 됨.
  5. EVS throttling: 억제 카운트가 맞고, 억제 중에도 누적 카운터(n_transitions_by_type)가
     정상 증가하는지.
  6. 단일채널 교차확인: AP 자체 카운터(ApFsmCore.counter)가 매 틱 WP 의 run_len 과 동일한지.
     불일치하면 RPN/persistence 순서 구현이 잘못된 것.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wp_poes_node as wp   # noqa: E402  -- 리플레이 생성용(ChannelState 등)
import ap_fsm_node as ap    # noqa: E402  -- rclpy 없이 import 되는지 자체가 검증 1

assert not ap._HAVE_RCLPY, "이 Windows 환경엔 rclpy 가 없어야 하는데 있음(예상과 다름)"
print(f"[verify_ap] 검증 1: rclpy 없이 ap_fsm_node import 성공 (_HAVE_RCLPY={ap._HAVE_RCLPY})")


def run_full_replay():
    """WP 리플레이를 메모리로 돌려 매 틱 (watch, counts, phys_ts, wp_run_len) 을 생성.
    S1 의 verify_wp_node.py 와 동일 파이프라인(재사용, 재구현 아님)."""
    cache_dir = wp.C_PD / "POES" / "MetOp03_count" / "poes_metop03_cache_parquet"
    raw = wp.load_raw_channel(cache_dir, "omni_p6")
    schedule = wp.build_tick_schedule(raw)
    cs = wp.ChannelState("omni_p6", bg_window_days=7)
    cadence = pd.Timedelta(seconds=900)
    k, onset_floor, gate_n, alert_n = 7.0, 0.1, 2, 4

    for ts in schedule:
        count = wp.resample_window(raw, ts, cadence)
        cs.push_sample(ts, count)
        cs.maybe_update_bg(ts)
        res = cs.eval_watchpoint(count, ts, k, onset_floor, gate_n, alert_n)
        yield ts, res


def check_full_replay_and_cross_check():
    print("=" * 70)
    print("[verify_ap] 검증 2+6: 전체 리플레이 -- ALERT 전이/게이트 틱 + WP run_len 교차확인")
    print("=" * 70)
    adt = ap.build_adt(["omni_p6"], ap.build_rpn_equation(["omni_p6"]), 2, 4, "RTS_SEP_ALERT")[0]
    core = ap.ApFsmCore(adt, dry_run=False)

    n_alert_transitions = 0
    mismatch_ticks = []
    n = 0
    for ts, res in run_full_replay():
        watch = {"omni_p6": res["watch"]}
        counts = {"omni_p6": res["count"]}
        phys_ts = ts.timestamp()
        out = core.process_tick(watch, counts, phys_ts)
        n += 1

        if core.counter != res["run_len"]:
            mismatch_ticks.append((ts, core.counter, res["run_len"]))
        if out["transitioned"] and out["from_state"] != ap.STATE_ALERT and out["state"] == ap.STATE_ALERT:
            n_alert_transitions += 1

    print(f"처리 틱 수: {n}")
    print(f"ALERT 전이 횟수: {n_alert_transitions}  (기대: 216)")
    print(f"게이트 개방 틱(n_ticks_gate_open): {core.n_ticks_gate_open}  (기대: 16204)")
    print(f"ALERT 개방 틱(n_ticks_alert_open): {core.n_ticks_alert_open}")
    print(f"n_ticks_total(실제 샘플만): {core.n_ticks_total}  (기대: 236848)")
    print(f"AP 카운터 vs WP run_len 불일치 틱 수: {len(mismatch_ticks)}  (기대: 0)")
    if mismatch_ticks:
        print(f"  첫 5개: {mismatch_ticks[:5]}")

    ok = (n_alert_transitions == 216 and core.n_ticks_gate_open == 16204
         and core.n_ticks_total == 236848 and len(mismatch_ticks) == 0)
    print(f"검증 2+6 종합: {ok}")
    return ok, core


def check_three_valued_logic():
    print("=" * 70)
    print("[verify_ap] 검증 3: 3-값 논리 단위 테스트")
    print("=" * 70)
    T, F, S, E = ap.WATCH_TRUE, ap.WATCH_FALSE, ap.WATCH_STALE, ap.WATCH_ERROR
    eq2 = ap.build_rpn_equation(["a", "b"])  # ["a","b","OR","EQUAL"]
    cases = [
        ({"a": T, "b": F}, ap.RPN_TRIGGERED),
        ({"a": T, "b": S}, ap.RPN_TRIGGERED),      # TRUE 우선
        ({"a": T, "b": E}, ap.RPN_TRIGGERED),      # TRUE 우선
        ({"a": F, "b": S}, ap.RPN_STALE),          # ERROR 없음 -> STALE
        ({"a": F, "b": E}, ap.RPN_ERROR),          # ERROR > STALE
        ({"a": S, "b": E}, ap.RPN_ERROR),          # ERROR > STALE
        ({"a": F, "b": F}, ap.RPN_NORMAL),
        ({"a": S, "b": S}, ap.RPN_STALE),
    ]
    ok = True
    for watch, expected in cases:
        result, trig = ap.evaluate_rpn(eq2, watch)
        good = (result == expected)
        ok = ok and good
        print(f"  {watch} -> {result} (기대 {expected}) {'OK' if good else 'FAIL'}")

    # 단일채널 퇴화 확인
    eq1 = ap.build_rpn_equation(["omni_p6"])
    print(f"단일채널 RPN: {eq1} (기대: ['omni_p6', 'EQUAL'])")
    ok = ok and (eq1 == ["omni_p6", "EQUAL"])
    r, trig = ap.evaluate_rpn(eq1, {"omni_p6": T})
    ok = ok and (r == ap.RPN_TRIGGERED and trig == ["omni_p6"])
    print(f"단일채널 평가: watch=TRUE -> {r}, trigger={trig}")
    print(f"검증 3 종합: {ok}")
    return ok


def check_latch():
    print("=" * 70)
    print("[verify_ap] 검증 4: latch (ALERT 재발화 억제 + --dry-run 에서 latch 안 됨)")
    print("=" * 70)
    T, F = ap.WATCH_TRUE, ap.WATCH_FALSE
    adt = ap.build_adt(["ch"], ap.build_rpn_equation(["ch"]), 2, 4, "RTS_SEP_ALERT")[0]

    # ACTIVE: 6틱 연속 TRUE -> counter 1..6, ALERT 는 counter>=4 부터 유지.
    # 다운링크는 ALERT 최초 확정 시(4번째 틱) 1회만 쌓여야 한다.
    core = ap.ApFsmCore(adt, dry_run=False)
    t = 1000.0
    for i in range(6):
        core.process_tick({"ch": T}, {"ch": 1.0}, t + i * 900)
    print(f"ACTIVE 6틱 연속 TRUE 후 downlink_queue: {len(core.downlink_queue)}개 (기대: 1)")
    ok_active = len(core.downlink_queue) == 1

    # FALSE 로 리셋 후 다시 4틱 TRUE -> 새 run 이므로 다운링크 2번째 항목이 쌓여야 한다.
    core.process_tick({"ch": F}, {"ch": 0.0}, t + 6 * 900)
    for i in range(4):
        core.process_tick({"ch": T}, {"ch": 1.0}, t + (7 + i) * 900)
    print(f"리셋 후 재확정 downlink_queue: {len(core.downlink_queue)}개 (기대: 2)")
    ok_reset = len(core.downlink_queue) == 2

    # DRY-RUN: 같은 시퀀스라도 다운링크가 전혀 쌓이지 않아야 한다. 상태는 ALERT 로 보고돼야 한다.
    core_dry = ap.ApFsmCore(adt, dry_run=True)
    out = None
    for i in range(6):
        out = core_dry.process_tick({"ch": T}, {"ch": 1.0}, t + i * 900)
    print(f"DRY-RUN 6틱 연속 TRUE 후 downlink_queue: {len(core_dry.downlink_queue)}개 (기대: 0), "
          f"state={out['state']} (기대: ALERT)")
    ok_dry = (len(core_dry.downlink_queue) == 0 and out["state"] == ap.STATE_ALERT)

    ok = ok_active and ok_reset and ok_dry
    print(f"검증 4 종합: {ok}")
    return ok


def check_evs_throttle():
    print("=" * 70)
    print("[verify_ap] 검증 5: EVS throttling (억제 카운트 + 억제 중에도 항목6 카운터 증가)")
    print("=" * 70)
    T, F = ap.WATCH_TRUE, ap.WATCH_FALSE
    adt = ap.build_adt(["ch"], ap.build_rpn_equation(["ch"]), 2, 4, "RTS_SEP_ALERT")[0]
    core = ap.ApFsmCore(adt, dry_run=False, evs_window_h=1.0, evs_max=5)

    # NOMINAL<->PRE_ALERT 를 반복 왕복(짧게 TRUE 2번으로 PRE_ALERT, 곧장 FALSE 로 복귀)해서
    # 같은 전이 유형("NOMINAL->PRE_ALERT")을 1시간 안에 5회 넘게 만든다.
    t = 2000.0
    should_log_hist = []
    for cycle in range(8):
        base = t + cycle * 600   # 10분 간격 왕복 -> 1시간 안에 여러 사이클
        core.process_tick({"ch": T}, {"ch": 1.0}, base)
        out = core.process_tick({"ch": T}, {"ch": 1.0}, base + 60)   # counter=2 -> PRE_ALERT 전이
        should_log_hist.append(out["should_log"])
        core.process_tick({"ch": F}, {"ch": 0.0}, base + 120)        # NOMINAL 복귀(리셋)

    key = "NOMINAL->PRE_ALERT"
    n_transitions = core.n_transitions_by_type.get(key, 0)
    n_suppressed = core.evs.suppressed_total.get(key, 0)
    print(f"전이 유형 '{key}' 발생 횟수(n_transitions_by_type): {n_transitions} (기대: 8, throttle 무관하게 항상 증가)")
    print(f"should_log 이력: {should_log_hist}")
    print(f"억제 횟수(evs.suppressed_total): {n_suppressed} (기대: {max(0, 8 - 5)}=3, max_count=5 넘는 6~8번째)")

    ok = (n_transitions == 8 and n_suppressed == 3
         and should_log_hist == [True, True, True, True, True, False, False, False])
    print(f"검증 5 종합: {ok}")
    return ok


def main():
    r2, core = check_full_replay_and_cross_check()
    r3 = check_three_valued_logic()
    r4 = check_latch()
    r5 = check_evs_throttle()
    print("=" * 70)
    print(f"[verify_ap] 종합: 리플레이+교차확인={r2} 3값논리={r3} latch={r4} EVS={r5}")
    print(f"[verify_ap] 전체 통과: {all([r2, r3, r4, r5])}")


if __name__ == "__main__":
    main()
