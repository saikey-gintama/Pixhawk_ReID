"""
verify_wp_node.py
==================
S1 로직 단위 검증 (Windows, rclpy 없이). 읽기 전용, C_PD 아래 아무것도 쓰지 않는다.

md 2절 요구사항: "판정 로직은 rclpy 없이 import 가능해야 하므로, 로직 단위 검증은
Windows에서 반드시 수행할 것." 이 스크립트가 그 검증이다.

검증 4종:
  1. rclpy 없는 환경에서 wp_poes_node import 자체가 성공하는가(이 스크립트가 돌아간다는
     사실 자체가 증거 -- 별도 assert 불필요, 명시적으로 한 번 더 확인).
  2. 전체 omni_p6 리플레이(1분 원시 -> ChannelState.eval_watchpoint() tick-count)로
     alert_n=4 onset 수가 S0.5 확정 수치(216, md 인과 실현 갭)와 정확히 같은가.
     gate_n=2 onset 수가 S0.5 sweep 표의 N=2 행(n_det=15235)과 같은가.
  3. 비교연산자 '>=' 경계값 케이스(count == threshold 정확히 같을 때 TRUE인가).
  4. STALE 이 run_len 을 리셋하지 않는지 합성 시퀀스로 확인.
  5. snapshot()/restore() 라운드트립 + 체크섬 변조 탐지(콜드스타트 폴백 로그) 확인.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wp_poes_node as wp  # noqa: E402  -- rclpy 없이 import 되는지 자체가 검증 1

assert not wp._HAVE_RCLPY, "이 Windows 환경엔 rclpy 가 없어야 하는데 있음(예상과 다름)"
print(f"[verify_wp] 검증 1: rclpy 없이 wp_poes_node import 성공 (_HAVE_RCLPY={wp._HAVE_RCLPY})")


def check_full_replay():
    print("=" * 70)
    print("[verify_wp] 검증 2: 전체 omni_p6 리플레이 -- tick-count 재현")
    print("=" * 70)
    cache_dir = wp.C_PD / "POES" / "MetOp03_count" / "poes_metop03_cache_parquet"
    raw = wp.load_raw_channel(cache_dir, "omni_p6")
    schedule = wp.build_tick_schedule(raw)
    print(f"omni_p6 원시 1분 표본 {len(raw)}개, 틱 스케줄(15분 grid, STALE 포함) {len(schedule)}개")

    cs = wp.ChannelState("omni_p6", bg_window_days=7)
    cadence = pd.Timedelta(seconds=900)
    k, onset_floor, gate_n, alert_n = 7.0, 0.1, 2, 4

    alert_onsets = []
    gate_onsets = []
    prev_run_len = 0
    for ts in schedule:
        count = wp.resample_window(raw, ts, cadence)
        cs.push_sample(ts, count)
        cs.maybe_update_bg(ts)
        res = cs.eval_watchpoint(count, ts, k, onset_floor, gate_n, alert_n)
        if res["run_len"] == alert_n and prev_run_len < alert_n:
            alert_onsets.append(ts)
        if res["run_len"] == gate_n and prev_run_len < gate_n:
            gate_onsets.append(ts)
        prev_run_len = res["run_len"]

    print(f"alert_n={alert_n} onset 수: {len(alert_onsets)}  "
          f"(기대: 216, S0.5/모듈 docstring 인과 실현 수)")
    print(f"gate_n={gate_n} onset 수: {len(gate_onsets)}  "
          f"(기대: 15235, S0.5 gate_persistence_sweep.csv N=2 행 n_det)")
    ok_alert = len(alert_onsets) == 216
    ok_gate = len(gate_onsets) == 15235
    print(f"alert_n=4 일치: {ok_alert}   gate_n=2 일치: {ok_gate}")
    return ok_alert, ok_gate


def check_ge_operator():
    print("=" * 70)
    print("[verify_wp] 검증 3: 비교연산자 '>=' 경계값")
    print("=" * 70)
    cs = wp.ChannelState("test", bg_window_days=7)
    t0 = pd.Timestamp("2020-01-01", tz="UTC")
    # 배경을 강제로 채워 threshold 를 결정론적으로 만든다.
    for i in range(10):
        cs.push_sample(t0 - pd.Timedelta(days=1) + pd.Timedelta(minutes=15 * i), 5.0)
    cs.maybe_update_bg(t0)
    # 배경 median=5.0, std=0 (전부 동일값) -> threshold = max(5.0 + 7*0, 0.1) = 5.0
    print(f"bg_median={cs.bg_median} bg_std={cs.bg_std} (threshold 기대값 5.0)")
    res_eq = cs.eval_watchpoint(5.0, t0 + pd.Timedelta(minutes=15), 7.0, 0.1, 2, 4)
    print(f"count == threshold(5.0) -> watch={res_eq['watch']} (기대: TRUE, '>' 였다면 FALSE)")
    ok = res_eq["watch"] == wp.WATCH_TRUE
    print(f"'>=' 연산자 확인: {ok}")
    return ok


def check_stale_skip():
    print("=" * 70)
    print("[verify_wp] 검증 4: STALE 이 run_len 을 리셋하지 않는지")
    print("=" * 70)
    cs = wp.ChannelState("test2", bg_window_days=7)
    t0 = pd.Timestamp("2020-01-01", tz="UTC")
    for i in range(10):
        cs.push_sample(t0 - pd.Timedelta(days=1) + pd.Timedelta(minutes=15 * i), 1.0)
    cs.maybe_update_bg(t0)  # bg_median=1.0, bg_std=0 -> threshold=max(1.0,0.1)=1.0

    seq = [10.0, 10.0, float("nan"), float("nan"), 10.0, 0.0]
    run_lens = []
    t = t0
    for v in seq:
        t = t + pd.Timedelta(minutes=15)
        res = cs.eval_watchpoint(v, t, 7.0, 0.1, 2, 4)
        run_lens.append((res["watch"], res["run_len"]))
    print(f"입력: {seq}")
    print(f"결과(watch, run_len): {run_lens}")
    # 기대: TRUE,1 / TRUE,2 / STALE,2(유지) / STALE,2(유지) / TRUE,3(이어짐, 리셋 아님) / FALSE,0
    expected = [("TRUE", 1), ("TRUE", 2), ("STALE", 2), ("STALE", 2), ("TRUE", 3), ("FALSE", 0)]
    ok = run_lens == expected
    print(f"기대값과 일치: {ok}  (기대: {expected})")
    return ok


def check_snapshot_roundtrip():
    print("=" * 70)
    print("[verify_wp] 검증 5: snapshot()/restore() 라운드트립 + 체크섬 변조 탐지")
    print("=" * 70)
    cs = wp.ChannelState("omni_p6", bg_window_days=7)
    t0 = pd.Timestamp("2020-01-01", tz="UTC")
    for i in range(5):
        cs.push_sample(t0 + pd.Timedelta(minutes=15 * i), 3.0 + i)
    cs.maybe_update_bg(t0)
    cs.eval_watchpoint(50.0, t0 + pd.Timedelta(minutes=90), 7.0, 0.1, 2, 4)
    cs.eval_watchpoint(50.0, t0 + pd.Timedelta(minutes=105), 7.0, 0.1, 2, 4)

    state = wp.SystemState({"omni_p6": cs})
    tmp_dir = Path(tempfile.mkdtemp(prefix="verify_wp_snap_"))
    snap_path = tmp_dir / "snap.json"
    state.snapshot(snap_path)

    cs2 = wp.ChannelState("omni_p6", bg_window_days=7)
    state2 = wp.SystemState({"omni_p6": cs2})
    ok_restore, reason = state2.restore(snap_path)
    match = (cs2.bg_median == cs.bg_median and cs2.run_len == cs.run_len
            and cs2.onset_ts == cs.onset_ts and list(cs2.recent_z) == list(cs.recent_z))
    print(f"정상 복구: {ok_restore} ({reason})  상태 일치: {match}")

    # 체크섬 변조 -- 파싱 후 데이터 필드만 바꾸고 checksum 은 그대로 둬(포맷 이슈 없이 확실히 불일치)
    import json as _json
    corrupt_path = tmp_dir / "corrupt.json"
    raw_dict = _json.loads(snap_path.read_text(encoding="utf-8"))
    raw_dict["channels"]["omni_p6"]["run_len"] = 999
    corrupt_path.write_text(_json.dumps(raw_dict), encoding="utf-8")
    cs3 = wp.ChannelState("omni_p6", bg_window_days=7)
    state3 = wp.SystemState({"omni_p6": cs3})
    ok_restore3, reason3 = state3.restore(corrupt_path)
    print(f"변조 파일 복구 결과: {ok_restore3} ({reason3})  "
          f"-- False 여야 콜드스타트 폴백이 맞게 동작한 것")

    # 파일 없음 -> 콜드스타트
    missing_path = tmp_dir / "missing.json"
    cs4 = wp.ChannelState("omni_p6", bg_window_days=7)
    state4 = wp.SystemState({"omni_p6": cs4})
    ok_restore4, reason4 = state4.restore(missing_path)
    print(f"파일 없음 복구 결과: {ok_restore4} ({reason4})")

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    ok = ok_restore and match and (not ok_restore3) and (not ok_restore4)
    print(f"검증 5 종합: {ok}")
    return ok


def main():
    r2a, r2b = check_full_replay()
    r3 = check_ge_operator()
    r4 = check_stale_skip()
    r5 = check_snapshot_roundtrip()
    print("=" * 70)
    print(f"[verify_wp] 종합: alert_n 재현={r2a} gate_n 재현={r2b} "
          f">= 연산자={r3} STALE skip={r4} snapshot={r5}")
    print(f"[verify_wp] 전체 통과: {all([r2a, r2b, r3, r4, r5])}")


if __name__ == "__main__":
    main()
