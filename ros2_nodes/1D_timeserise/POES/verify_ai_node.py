"""
verify_ai_node.py
==================
S3 로직 단위 검증 (Windows, rclpy 없이). 읽기 전용, C_PD 아래 아무것도 쓰지 않는다.

검증 6종 (md 요청 그대로):
  1. rclpy 없이 ai_tcn_node import 성공 + torch import 확인.
  2. 윈도우 조립 동등성 -- 가장 중요. windows_metop03_omni_p6.parquet 에서 임의 100행을
     뽑아, 같은 window_end_time 에서 온보드 ZBuffer 가 조립한 (1,14) 이 그 행의
     z_lag13..z_lag0 과 일치하는지. 목표: 최대절대오차 < 1e-9.
  3. 체크포인트 정합 -- fold0.pt 추론 확률이 oof_predictions.parquet(fold==0) 과 일치하는지.
  4. 게이팅: 전 구간 리플레이(WP+AP+AI)에서 추론 호출 16,204회, 게이트 닫힌 틱에서는
     추론 시도 자체가 없는지(None 반환).
  5. NaN/미충족 윈도우 -> INSUFFICIENT_DATA, 추론(forward) 안 하는지.
  6. 워밍업 값이 집계에서 제외되는지(warmup() 결과와 per-tick 결과가 완전히 분리된
     구조인지 -- tcn_startup.json 용 리스트 vs AiTcnCore.maybe_infer() 반환값).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wp_poes_node as wp    # noqa: E402
import ap_fsm_node as ap     # noqa: E402
import ai_tcn_node as ai     # noqa: E402  -- rclpy 없이 import 되는지 자체가 검증 1

assert not ai._HAVE_RCLPY, "이 Windows 환경엔 rclpy 가 없어야 하는데 있음(예상과 다름)"
print(f"[verify_ai] 검증 1: rclpy 없이 ai_tcn_node import 성공 (_HAVE_RCLPY={ai._HAVE_RCLPY}), "
      f"torch {torch.__version__}")


def check_window_assembly():
    print("=" * 70)
    print("[verify_ai] 검증 2: 윈도우 조립 동등성 (링버퍼 vs windows_metop03_omni_p6.parquet)")
    print("=" * 70)
    ts = pd.read_parquet(ai.C_PD / "predict_v0" / "dataset_v0" / "timeseries_metop03_omni_p6.parquet")
    ts = ts.set_index("time")
    z = ts["zscore"]

    windows = pd.read_parquet(ai.C_PD / "predict_v0" / "dataset_v0" / "windows_metop03_omni_p6.parquet")
    sample = windows.sample(100, random_state=42)
    lag_cols = [f"z_lag{13 - j}" for j in range(14)]   # z_lag13..z_lag0, 오래된->최신

    max_diff = 0.0
    n_bad = 0
    for _, row in sample.iterrows():
        wet = row["window_end_time"]
        pos = ts.index.get_loc(wet)
        z_slice = z.iloc[pos - 13: pos + 1].values

        zbuf = ai.ZBuffer(["omni_p6"], maxlen=14)
        for v in z_slice:
            zbuf.update("omni_p6", v)
        X = zbuf.assemble().flatten()

        expected = row[lag_cols].values.astype(float)
        diff = np.abs(X - expected).max()
        max_diff = max(max_diff, diff)
        if diff > 1e-9:
            n_bad += 1

    print(f"표본 100개, 최대 절대오차: {max_diff:.3e}, 불일치(>1e-9) 표본 수: {n_bad}")
    ok = (max_diff < 1e-9)
    print(f"검증 2 종합: {ok}")
    return ok


def check_checkpoint_consistency():
    print("=" * 70)
    print("[verify_ai] 검증 3: 체크포인트 정합 (fold0.pt vs oof_predictions.parquet)")
    print("=" * 70)
    ckpt_dir = ai.CKPT_ROOT / "omni_p6_binary" / "checkpoints"
    model, manifest = ai.load_single_model(ckpt_dir, fold=0)

    oof = pd.read_parquet(ai.CKPT_ROOT / "omni_p6_binary" / "oof_predictions.parquet")
    oof0 = oof[oof["fold"] == 0].copy()
    oof0["window_end_time"] = pd.to_datetime(oof0["window_end_time"], utc=True)
    print(f"fold==0 OOF 행 수: {len(oof0)}")

    windows = pd.read_parquet(ai.C_PD / "predict_v0" / "dataset_v0" / "windows_metop03_omni_p6.parquet")
    windows = windows.set_index("window_end_time")
    lag_cols = [f"z_lag{13 - j}" for j in range(14)]

    sample = oof0.sample(min(200, len(oof0)), random_state=42)
    max_diff = 0.0
    n_missing = 0
    n_checked = 0
    for _, r in sample.iterrows():
        wet = r["window_end_time"]
        if wet not in windows.index:
            n_missing += 1
            continue
        X = windows.loc[wet, lag_cols].values.astype(np.float32).reshape(1, -1)
        xb = torch.tensor(X, dtype=torch.float32)
        p_event, y_pred, _ = ai.infer_from_tensor([model], manifest, xb)
        diff = abs(p_event - float(r["proba_event"]))
        max_diff = max(max_diff, diff)
        n_checked += 1

    print(f"대조 표본 {n_checked}개(윈도우 매칭 실패 {n_missing}개), 최대 절대오차: {max_diff:.3e}")
    ok = (n_checked > 0 and max_diff < 1e-4)   # 부동소수/round 오차 여유
    print(f"검증 3 종합: {ok}")
    return ok


def run_full_pipeline():
    """S1(WP)+S2(AP)+S3(AI) 를 메모리로 연결해 전 구간 리플레이. 재사용, 재구현 아님."""
    cache_dir = wp.C_PD / "POES" / "MetOp03_count" / "poes_metop03_cache_parquet"
    raw = wp.load_raw_channel(cache_dir, "omni_p6")
    schedule = wp.build_tick_schedule(raw)
    cs = wp.ChannelState("omni_p6", bg_window_days=7)
    cadence = pd.Timedelta(seconds=900)
    k, onset_floor, gate_n, alert_n = 7.0, 0.1, 2, 4

    adt = ap.build_adt(["omni_p6"], ap.build_rpn_equation(["omni_p6"]), gate_n, alert_n,
                       "RTS_SEP_ALERT")[0]
    ap_core = ap.ApFsmCore(adt, dry_run=False)

    ckpt_dir = ai.CKPT_ROOT / "omni_p6_binary" / "checkpoints"
    model, manifest = ai.load_single_model(ckpt_dir, fold=0)
    ai_core = ai.AiTcnCore([model], manifest, ["omni_p6"], window=14)

    for ts in schedule:
        count = wp.resample_window(raw, ts, cadence)
        cs.push_sample(ts, count)
        cs.maybe_update_bg(ts)
        wp_res = cs.eval_watchpoint(count, ts, k, onset_floor, gate_n, alert_n)

        phys_ts = ts.timestamp()
        ap_out = ap_core.process_tick({"omni_p6": wp_res["watch"]}, {"omni_p6": wp_res["count"]}, phys_ts)

        ai_core.update_buffer("omni_p6", wp_res["z"])
        ai_out = ai_core.maybe_infer(ap_out["state"], ap_out["has_data"])

        yield ts, wp_res, ap_out, ai_out, ai_core


def check_gating_and_insufficient():
    print("=" * 70)
    print("[verify_ai] 검증 4+5: 전 구간 게이팅(추론 16,204회) + INSUFFICIENT_DATA 처리")
    print("=" * 70)
    n_closed_but_called = 0
    last_ai_core = None
    n = 0
    for ts, wp_res, ap_out, ai_out, ai_core in run_full_pipeline():
        n += 1
        gate_open = ap_out["state"] in ("PRE_ALERT", "ALERT")
        if not gate_open and ai_out is not None:
            n_closed_but_called += 1
        last_ai_core = ai_core

    print(f"처리 틱 수: {n}")
    print(f"AI 추론 호출 횟수(n_inference_calls): {last_ai_core.n_inference_calls}  (기대: 16204)")
    print(f"  성공(OK): {last_ai_core.n_ok}, INSUFFICIENT_DATA: {last_ai_core.n_insufficient}")
    print(f"게이트 닫힌 틱인데 추론이 호출된 경우: {n_closed_but_called}개  (기대: 0)")

    ok4 = (last_ai_core.n_inference_calls == 16204 and n_closed_but_called == 0)
    ok5 = (last_ai_core.n_insufficient > 0
          and last_ai_core.n_ok + last_ai_core.n_insufficient == last_ai_core.n_inference_calls)
    print(f"검증 4 종합: {ok4}")
    print(f"검증 5 종합(INSUFFICIENT_DATA {last_ai_core.n_insufficient}건 존재 + 카운트 정합): {ok5}")

    # 5-보조: 합성 케이스로 직접 확인(버퍼 3개만 채운 상태 -> INSUFFICIENT_DATA, 추론 안 함)
    model, manifest = ai.load_single_model(ai.CKPT_ROOT / "omni_p6_binary" / "checkpoints", fold=0)
    synth = ai.AiTcnCore([model], manifest, ["omni_p6"], window=14)
    for v in [1.0, 2.0, 3.0]:
        synth.update_buffer("omni_p6", v)
    out = synth.maybe_infer("PRE_ALERT")
    print(f"합성: 버퍼 3/14개 -> status={out['status']} (기대: INSUFFICIENT_DATA), "
          f"n_ok={synth.n_ok}(기대 0), n_insufficient={synth.n_insufficient}(기대 1)")
    ok5_synth = (out["status"] == "INSUFFICIENT_DATA" and synth.n_ok == 0 and synth.n_insufficient == 1)

    # NaN 이 섞인 14개 -> 역시 INSUFFICIENT_DATA
    synth2 = ai.AiTcnCore([model], manifest, ["omni_p6"], window=14)
    for v in [1.0] * 13 + [float("nan")]:
        synth2.update_buffer("omni_p6", v)
    out2 = synth2.maybe_infer("ALERT")
    print(f"합성: 14개 채웠으나 마지막이 NaN -> status={out2['status']} (기대: INSUFFICIENT_DATA)")
    ok5_synth2 = (out2["status"] == "INSUFFICIENT_DATA")

    ok5_all = ok5 and ok5_synth and ok5_synth2
    print(f"검증 5(합성 포함) 종합: {ok5_all}")
    return ok4, ok5_all


def check_warmup_isolation():
    print("=" * 70)
    print("[verify_ai] 검증 6: 워밍업 값이 집계에서 분리되는지")
    print("=" * 70)
    model, manifest = ai.load_single_model(ai.CKPT_ROOT / "omni_p6_binary" / "checkpoints", fold=0)

    warmup_times = ai.warmup([model], (1, 14), n_iters=20)
    print(f"warmup() 반환: {len(warmup_times)}개 시간값(tcn_startup.json 전용)")

    core = ai.AiTcnCore([model], manifest, ["omni_p6"], window=14)
    for v in np.random.default_rng(0).normal(size=14):
        core.update_buffer("omni_p6", float(v))
    per_tick_infer_ms = []
    for _ in range(5):
        out = core.maybe_infer("PRE_ALERT")
        per_tick_infer_ms.append(out["infer_ms"])
    print(f"per-tick maybe_infer() infer_ms: {[round(x, 4) for x in per_tick_infer_ms]}")

    # 구조적 분리 확인: warmup_times 와 per_tick_infer_ms 는 서로 다른 리스트/출처이며,
    # AiTcnCore 에는 warmup 이력을 저장하는 필드가 아예 없다(코드 검사로 확인).
    no_warmup_field = not hasattr(core, "warmup_ms") and not hasattr(core, "warmup_times")
    disjoint_lists = (warmup_times is not per_tick_infer_ms)
    both_have_values = (len(warmup_times) == 20 and len(per_tick_infer_ms) == 5)
    print(f"AiTcnCore 에 warmup 관련 필드 없음(분리 구조): {no_warmup_field}")
    print(f"warmup 리스트와 per-tick 리스트가 별개 객체: {disjoint_lists}")

    ok = no_warmup_field and disjoint_lists and both_have_values
    print(f"검증 6 종합: {ok}")
    return ok


def main():
    r2 = check_window_assembly()
    r3 = check_checkpoint_consistency()
    r4, r5 = check_gating_and_insufficient()
    r6 = check_warmup_isolation()
    print("=" * 70)
    print(f"[verify_ai] 종합: 윈도우조립={r2} 체크포인트정합={r3} 게이팅={r4} "
          f"INSUFFICIENT_DATA={r5} 워밍업분리={r6}")
    print(f"[verify_ai] 전체 통과: {all([r2, r3, r4, r5, r6])}")


if __name__ == "__main__":
    main()
