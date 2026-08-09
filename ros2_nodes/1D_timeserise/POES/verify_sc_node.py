"""
verify_sc_node.py
==================
S4 로직 단위 검증 (Windows, rclpy/px4_msgs 없이). 읽기 전용, C_PD 아래 아무것도 쓰지 않는다.

검증 7종 (md 요청 그대로):
  1. rclpy/px4_msgs 없이 sc_offboard_node import 성공.
  2. 상태 매핑: PRE_ALERT -> pre_streaming True, 전이 없음.
     ALERT -> TRACKING, NOMINAL -> HOLD -> (타임아웃) -> IDLE.
  3. off vs shadow 전이 시퀀스가 바이트 단위로 같은지(같은 입력을 두 번 돌려 비교).
  4. gate 모드: p_event 가 임계 미만이면 ALERT 여도 전이 안 되는지.
  5. passive: 실제 명령이 "발행"(commands_sent)되지 않고 PASSIVE_SUPPRESS(commands_suppressed)
     만 기록되는지.
  6. 전 구간 리플레이(off 모드): IDLE->TRACKING 전이 횟수가 AP 의 ALERT 전이 216회와 일치하는지.
  7. t_sample_rx 가 WP->AP->AI->SC 를 거치며(JSON 직렬화 3회 포함) 변조되지 않는지.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wp_poes_node as wp    # noqa: E402
import ap_fsm_node as ap     # noqa: E402
import sc_offboard_node as sc  # noqa: E402  -- rclpy/px4_msgs 없이 import 되는지 자체가 검증 1

assert not sc._HAVE_RCLPY, "이 Windows 환경엔 rclpy 가 없어야 하는데 있음(예상과 다름)"
assert not sc._HAVE_PX4, "이 Windows 환경엔 px4_msgs 가 없어야 하는데 있음(예상과 다름)"
print(f"[verify_sc] 검증 1: rclpy/px4_msgs 없이 sc_offboard_node import 성공 "
      f"(_HAVE_RCLPY={sc._HAVE_RCLPY}, _HAVE_PX4={sc._HAVE_PX4})")


def check_state_mapping():
    print("=" * 70)
    print("[verify_sc] 검증 2: 상태 매핑 (PRE_ALERT/ALERT/NOMINAL/HOLD 타임아웃)")
    print("=" * 70)
    core = sc.ScFsmCore(ai_gate_mode="off", hold_duration_sec=3.0, passive=True)
    t = 1000.0

    ev1 = core.on_alert(sc.AP_PRE_ALERT, t, fsm_pub_ts=t - 0.01, t_sample_rx=t - 0.02)
    ok_pre = (core.pre_streaming is True and core.state == sc.STATE_IDLE and ev1 == ["PRE_ALERT_PRESTREAM"])
    print(f"PRE_ALERT: pre_streaming={core.pre_streaming}, state={core.state}, events={ev1} "
          f"(기대: True/IDLE/['PRE_ALERT_PRESTREAM'])")

    ev2 = core.on_alert(sc.AP_ALERT, t + 1, fsm_pub_ts=t + 0.99, t_sample_rx=t + 0.98)
    ok_alert = (core.state == sc.STATE_TRACKING and ev2 == ["ALERT_TRIGGER"])
    print(f"ALERT: state={core.state}, events={ev2} (기대: TRACKING/['ALERT_TRIGGER'])")

    ev3 = core.on_alert(sc.AP_NOMINAL, t + 2, fsm_pub_ts=t + 1.99, t_sample_rx=t + 1.98)
    ok_nominal = (core.state == sc.STATE_HOLD and core.pre_streaming is False and ev3 == ["ALERT_CLEARED"])
    print(f"NOMINAL: state={core.state}, pre_streaming={core.pre_streaming}, events={ev3} "
          f"(기대: HOLD/False/['ALERT_CLEARED'])")

    # HOLD -> IDLE: hold_duration_sec*2 = 6초 경과 후 tick_control 에서 타임아웃
    ev4 = core.tick_control(t + 2 + 5.0)      # 아직 6초 안 됨
    ok_hold_wait = (core.state == sc.STATE_HOLD and ev4 == [])
    ev5 = core.tick_control(t + 2 + 6.1)      # 6초 경과
    ok_hold_timeout = (core.state == sc.STATE_IDLE and ev5 == ["HOLD_TIMEOUT"])
    print(f"HOLD 5초 경과: state={core.state}, events={ev4} (기대: HOLD/[])")
    print(f"HOLD 6.1초 경과: state={core.state}, events={ev5} (기대: IDLE/['HOLD_TIMEOUT'])")

    ok = ok_pre and ok_alert and ok_nominal and ok_hold_wait and ok_hold_timeout
    print(f"검증 2 종합: {ok}")
    return ok


def _drive_sequence(core: "sc.ScFsmCore", feed_verdicts: bool) -> list[tuple]:
    """동일 alert 시퀀스를 core 에 흘리고 (event, state) 이력을 반환.
    feed_verdicts=True 면 중간중간 on_ai_verdict() 도 호출(shadow 모드 시뮬레이션)."""
    t = 5000.0
    seq = [
        (sc.AP_PRE_ALERT, 0.0), (sc.AP_ALERT, 0.9), (sc.AP_ALERT, 1.9),
        (sc.AP_NOMINAL, 2.9), (sc.AP_PRE_ALERT, 20.0), (sc.AP_ALERT, 20.9),
        (sc.AP_NOMINAL, 30.0),
    ]
    hist = []
    for i, (ap_state, dt) in enumerate(seq):
        now = t + dt
        if feed_verdicts:
            core.on_ai_verdict(0.1 + 0.05 * i)   # state 에 영향 없어야 함
        events = core.on_alert(ap_state, now, fsm_pub_ts=now - 0.01, t_sample_rx=now - 0.02)
        for ev in events:
            hist.append((ev, core.state))
        ctrl_events = core.tick_control(now)
        for ev in ctrl_events:
            hist.append((ev, core.state))
    return hist


def check_off_vs_shadow():
    print("=" * 70)
    print("[verify_sc] 검증 3: off vs shadow 전이 시퀀스 바이트 비교")
    print("=" * 70)
    core_off = sc.ScFsmCore(ai_gate_mode="off", passive=True)
    core_shadow = sc.ScFsmCore(ai_gate_mode="shadow", passive=True)

    hist_off = _drive_sequence(core_off, feed_verdicts=False)
    hist_shadow = _drive_sequence(core_shadow, feed_verdicts=True)   # shadow 는 verdict 도 받음

    print(f"off    이력: {hist_off}")
    print(f"shadow 이력: {hist_shadow}")
    ok = (hist_off == hist_shadow)
    print(f"검증 3 종합(바이트 단위 동일): {ok}")
    return ok


def check_gate_mode():
    print("=" * 70)
    print("[verify_sc] 검증 4: gate 모드 (p_event 미달 시 전이 억제)")
    print("=" * 70)
    core = sc.ScFsmCore(ai_gate_mode="gate", ai_threshold=0.5, passive=True)
    t = 9000.0
    core.on_ai_verdict(0.2)   # 임계 미달
    ev1 = core.on_alert(sc.AP_ALERT, t, fsm_pub_ts=t - 0.01, t_sample_rx=t - 0.02)
    ok_suppressed = (core.state == sc.STATE_IDLE and ev1 == ["ALERT_GATED_SUPPRESS"])
    print(f"p_event=0.2 < 0.5: state={core.state}, events={ev1} (기대: IDLE/['ALERT_GATED_SUPPRESS'])")

    core.on_ai_verdict(0.7)   # 임계 충족
    ev2 = core.on_alert(sc.AP_ALERT, t + 1, fsm_pub_ts=t + 0.99, t_sample_rx=t + 0.98)
    ok_allowed = (core.state == sc.STATE_TRACKING and ev2 == ["ALERT_TRIGGER"])
    print(f"p_event=0.7 >= 0.5: state={core.state}, events={ev2} (기대: TRACKING/['ALERT_TRIGGER'])")

    ok = ok_suppressed and ok_allowed
    print(f"검증 4 종합: {ok}")
    return ok


def check_passive():
    print("=" * 70)
    print("[verify_sc] 검증 5: passive (명령 미발행, PASSIVE_SUPPRESS 만 기록)")
    print("=" * 70)
    core = sc.ScFsmCore(ai_gate_mode="off", passive=True)
    t = 12000.0
    core.on_alert(sc.AP_ALERT, t, fsm_pub_ts=t - 0.01, t_sample_rx=t - 0.02)   # IDLE->TRACKING
    for i in range(10):
        core.tick_control(t + 0.1 * i)   # counter 0..9 -> 10번째에서 트리거
    ev = core.tick_control(t + 1.0)
    print(f"tick_control 이벤트: {ev}")
    print(f"commands_sent(실제 발행): {core.commands_sent} (기대: 빈 리스트)")
    print(f"commands_suppressed(PASSIVE): {core.commands_suppressed} (기대: ['OFFBOARD_MODE_CMD','ARM_CMD'])")
    ok_passive = (core.commands_sent == [] and core.commands_suppressed == ["OFFBOARD_MODE_CMD", "ARM_CMD"])

    core_active = sc.ScFsmCore(ai_gate_mode="off", passive=False)
    core_active.on_alert(sc.AP_ALERT, t, fsm_pub_ts=t - 0.01, t_sample_rx=t - 0.02)
    for i in range(10):
        core_active.tick_control(t + 0.1 * i)
    core_active.tick_control(t + 1.0)
    ok_active = (core_active.commands_sent == ["OFFBOARD_MODE_CMD", "ARM_CMD"]
                and core_active.commands_suppressed == [])
    print(f"ACTIVE 비교: commands_sent={core_active.commands_sent} "
          f"commands_suppressed={core_active.commands_suppressed} (기대: 발행 있음/억제 없음)")

    ok = ok_passive and ok_active
    print(f"검증 5 종합: {ok}")
    return ok


def check_full_replay_idle_to_tracking():
    print("=" * 70)
    print("[verify_sc] 검증 6: 전 구간 리플레이(off) -- IDLE->TRACKING 전이 216회")
    print("=" * 70)
    cache_dir = wp.C_PD / "POES" / "MetOp03_count" / "poes_metop03_cache_parquet"
    raw = wp.load_raw_channel(cache_dir, "omni_p6")
    schedule = wp.build_tick_schedule(raw)
    cs = wp.ChannelState("omni_p6", bg_window_days=7)
    cadence = pd.Timedelta(seconds=900)
    k, onset_floor, gate_n, alert_n = 7.0, 0.1, 2, 4

    adt = ap.build_adt(["omni_p6"], ap.build_rpn_equation(["omni_p6"]), gate_n, alert_n,
                       "RTS_SEP_ALERT")[0]
    ap_core = ap.ApFsmCore(adt, dry_run=False)
    sc_core = sc.ScFsmCore(ai_gate_mode="off", hold_duration_sec=3.0, passive=True)

    n_alert_trigger = 0
    for ts in schedule:
        count = wp.resample_window(raw, ts, cadence)
        cs.push_sample(ts, count)
        cs.maybe_update_bg(ts)
        wp_res = cs.eval_watchpoint(count, ts, k, onset_floor, gate_n, alert_n)

        phys_ts = ts.timestamp()
        ap_out = ap_core.process_tick({"omni_p6": wp_res["watch"]}, {"omni_p6": wp_res["count"]}, phys_ts)

        # phys_ts 를 SC 의 "now" 로 그대로 써서 HOLD 타임아웃이 물리 시간축으로 정상 순환하게 한다
        # (SC 는 원래 벽시계 기준이지만, 리플레이 로직검증에선 phys_ts 를 시계로 취급 -- md 4절 취지 연장).
        events = sc_core.on_alert(ap_out["state"], phys_ts, fsm_pub_ts=phys_ts, t_sample_rx=phys_ts)
        if "ALERT_TRIGGER" in events:
            n_alert_trigger += 1
        sc_core.tick_control(phys_ts)

    print(f"IDLE->TRACKING(ALERT_TRIGGER) 전이 횟수: {n_alert_trigger}  (기대: 216)")
    ok = (n_alert_trigger == 216)
    print(f"검증 6 종합: {ok}")
    return ok


def check_t_sample_rx_integrity():
    print("=" * 70)
    print("[verify_sc] 검증 7: t_sample_rx 무변조 (WP->AP->AI->SC, JSON 직렬화 3회 포함)")
    print("=" * 70)
    t_sample_rx_orig = time.time()

    # WP -> /wp_results (t_sample_rx 부착)
    wp_payload = {"ts": 1000.0, "t_sample_rx": t_sample_rx_orig, "wp_pub_ts": time.time(),
                 "n_channels": 1, "results": [{"channel": "omni_p6", "watch": "TRUE", "count": 5.0,
                                                "z": 3.0, "run_len": 4, "gate_open": True, "alert_open": True}]}
    wire1 = json.loads(json.dumps(wp_payload))   # WP -> AP 전송(직렬화 1회)

    # AP: ap_fsm_node.py callback 과 동일 추출/재발행 패턴(payload.get 그대로 전달)
    t_from_wp = wire1.get("t_sample_rx")
    ap_alert = {"ts": wire1["ts"], "fsm_pub_ts": time.time(), "t_sample_rx": t_from_wp,
               "ap_state": "ALERT", "counter": 4, "trigger_channels": ["omni_p6"],
               "rpn_result": "TRIGGERED", "has_data": True}
    wire2 = json.loads(json.dumps(ap_alert))   # AP -> AI/SC 전송(직렬화 2회)

    # AI: ai_tcn_node.py on_sep_alert 와 동일 추출/재발행 패턴
    t_from_ap = wire2.get("t_sample_rx")
    ai_verdict = {"ts": wire2["ts"], "ai_pub_ts": time.time(), "t_sample_rx": t_from_ap,
                 "ap_state": wire2["ap_state"], "status": "OK", "p_event": 0.8, "y_pred": 1,
                 "n_models": 1, "window_ok": True}
    wire3 = json.loads(json.dumps(ai_verdict))   # AI -> (참고용) 직렬화 3회

    # SC: /sep_alert(wire2) 를 주 경로로 받는다 -- ScFsmCore.on_alert 로 그대로 전달
    core = sc.ScFsmCore(ai_gate_mode="off", passive=True)
    core.on_alert(wire2["ap_state"], time.time(), fsm_pub_ts=wire2["fsm_pub_ts"],
                 t_sample_rx=wire2.get("t_sample_rx"))

    print(f"원본 t_sample_rx:        {t_sample_rx_orig!r}")
    print(f"WP->AP 전달(wire1):      {wire1['t_sample_rx']!r}")
    print(f"AP->AI/SC 전달(wire2):   {wire2['t_sample_rx']!r}")
    print(f"AI->(참고) 전달(wire3):  {wire3['t_sample_rx']!r}")
    print(f"SC core.last_t_sample_rx: {core.last_t_sample_rx!r}")

    ok = (t_sample_rx_orig == wire1["t_sample_rx"] == wire2["t_sample_rx"]
         == wire3["t_sample_rx"] == core.last_t_sample_rx)
    print(f"검증 7 종합(전 구간 정확히 동일): {ok}")
    return ok


def main():
    r2 = check_state_mapping()
    r3 = check_off_vs_shadow()
    r4 = check_gate_mode()
    r5 = check_passive()
    r6 = check_full_replay_idle_to_tracking()
    r7 = check_t_sample_rx_integrity()
    print("=" * 70)
    print(f"[verify_sc] 종합: 상태매핑={r2} off/shadow동일={r3} gate={r4} passive={r5} "
          f"전구간216회={r6} t_sample_rx무변조={r7}")
    print(f"[verify_sc] 전체 통과: {all([r2, r3, r4, r5, r6, r7])}")


if __name__ == "__main__":
    main()
