"""
verify_multichannel.py
=======================
S5.2 다채널 경로 검증 (Windows, rclpy 없이, 젯슨 전에). 읽기 전용, C_PD 아래
아무것도 쓰지 않는다. S1~S3 검증은 전부 단일채널이었다 -- 시나리오 (d) 가 쓰는
3채널 경로(omni_p6, pro_tel0_p5, pro_tel90_p5)는 여기서 처음 검증한다.

검증 4종 (md 요청 그대로):
  ① WP: 채널별 watch 불리언이 3개 나오는지. 각 채널이 자기 배경·자기 임계를
        갖는지(공유하면 버그) -- 수치 목표 없음, 정성 확인.
  ② AP: RPN 이 퇴화하지 않고 OR 결합이 실제로 동작하는지(한 채널만 TRUE 여도
        전체 TRIGGERED). ALERT 전이 수는 기록만, 216 과 비교하지 않는다
        (cFS 정통 순서라 오프라인 다채널 형태와 갈리는 것이 이미 문서화됨 -- S2).
  ③ AI: manifest["channels"] 순서로 (1,3,14) 텐서가 조립되는지.
        windows_metop03_omni_p6-pro_tel0_p5-pro_tel90_p5.parquet 에서 임의 100행을
        뽑아 최대절대오차 < 1e-9. 채널을 이름으로 매핑했는지(위치 아님) 별도
        합성 케이스로 확인 -- update() 호출 순서를 일부러 섞어도 assemble() 이
        manifest 순서를 지키는지 본다.
  ④ AI: 3채널 체크포인트(input_size=3) 로드 후 실제 추론이 도는지.
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
import ai_tcn_node as ai     # noqa: E402

CHANNELS = ["omni_p6", "pro_tel0_p5", "pro_tel90_p5"]
CACHE_DIR = wp.C_PD / "POES" / "MetOp03_count" / "poes_metop03_cache_parquet"
CKPT_DIR = ai.CKPT_ROOT / "omni_p6_pro_tel0_p5_pro_tel90_p5_binary" / "checkpoints"


def run_3ch_replay():
    """3채널 전 구간 리플레이. 반환: z_history(dict[ch][ts]->z), wp 결과 이력,
    AP 코어(리플레이 끝난 상태)."""
    raws = {ch: wp.load_raw_channel(CACHE_DIR, ch) for ch in CHANNELS}
    schedule = None
    for s in raws.values():
        sch = wp.build_tick_schedule(s)
        schedule = sch if schedule is None else schedule.union(sch)

    css = {ch: wp.ChannelState(ch, bg_window_days=7) for ch in CHANNELS}
    cadence = pd.Timedelta(seconds=900)
    k, onset_floor, gate_n, alert_n = 7.0, 0.1, 2, 4

    adt = ap.build_adt(CHANNELS, ap.build_rpn_equation(CHANNELS), gate_n, alert_n, "RTS_SEP_ALERT")[0]
    ap_core = ap.ApFsmCore(adt, dry_run=False)

    z_history = {ch: {} for ch in CHANNELS}
    watch_history = {ch: {} for ch in CHANNELS}
    bg_history = {ch: {} for ch in CHANNELS}
    only_one_true_ticks = 0   # OR 이 실제로 동작을 증명하는 사례 카운트(한 채널만 TRUE)

    for ts in schedule:
        watch = {}
        counts = {}
        for ch in CHANNELS:
            count = wp.resample_window(raws[ch], ts, cadence)
            css[ch].push_sample(ts, count)
            css[ch].maybe_update_bg(ts)
            res = css[ch].eval_watchpoint(count, ts, k, onset_floor, gate_n, alert_n)
            z_history[ch][ts] = res["z"]
            watch_history[ch][ts] = res["watch"]
            bg_history[ch][ts] = (res["bg_median"], res["bg_std"])
            watch[ch] = res["watch"]
            counts[ch] = res["count"]

        n_true = sum(1 for v in watch.values() if v == wp.WATCH_TRUE)
        if n_true == 1:
            only_one_true_ticks += 1

        ap_core.process_tick(watch, counts, ts.timestamp())

    return {
        "schedule": schedule, "z_history": z_history, "watch_history": watch_history,
        "bg_history": bg_history, "ap_core": ap_core, "only_one_true_ticks": only_one_true_ticks,
    }


def check_wp_independence(replay: dict) -> bool:
    print("=" * 70)
    print("[verify_mc] 검증 ① WP: 채널별 독립 watch/배경 (정성 확인, 수치 목표 없음)")
    print("=" * 70)
    bg_hist = replay["bg_history"]
    watch_hist = replay["watch_history"]
    schedule = replay["schedule"]

    # 배경이 서로 다른지(공유 인스턴스 버그였다면 전부 동일했을 것)
    sample_ts = schedule[len(schedule) // 2]
    bgs = {ch: bg_hist[ch][sample_ts] for ch in CHANNELS}
    print(f"중간 시점({sample_ts}) 채널별 (bg_median,bg_std): {bgs}")
    distinct_bg = len({v for v in bgs.values()}) > 1
    print(f"채널별 배경이 서로 다름(독립): {distinct_bg}")

    # watch 시퀀스가 채널마다 다른지(동일 인스턴스 공유였다면 완전히 같았을 것)
    seq = {ch: tuple(watch_hist[ch][t] for t in schedule[:2000]) for ch in CHANNELS}
    all_identical = (seq[CHANNELS[0]] == seq[CHANNELS[1]] == seq[CHANNELS[2]])
    print(f"앞 2000틱 watch 시퀀스가 3채널 모두 동일함(True 면 공유 버그 의심): {all_identical}")

    ok = distinct_bg and not all_identical
    print(f"검증 ① 종합: {ok}")
    return ok


def check_ap_or_combination(replay: dict) -> bool:
    print("=" * 70)
    print("[verify_mc] 검증 ② AP: RPN OR 결합 동작 + ALERT 전이 수(216 과 비교 안 함)")
    print("=" * 70)
    eq = ap.build_rpn_equation(CHANNELS)
    print(f"3채널 RPN: {eq}")
    degenerate = (eq == [CHANNELS[0], "EQUAL"])
    print(f"퇴화 여부(3채널인데 단일항이면 버그): {degenerate} (기대: False)")

    n_only_one = replay["only_one_true_ticks"]
    print(f"'한 채널만 TRUE'인 틱 수: {n_only_one} (0이면 OR 이 실제로 시험된 적이 없다는 뜻)")

    n_alert = replay["ap_core"].n_transitions_by_type.get("PRE_ALERT->ALERT", 0) \
        + replay["ap_core"].n_transitions_by_type.get("NOMINAL->ALERT", 0)
    print(f"ALERT 전이 횟수(3채널, cFS 정통 순서): {n_alert}  "
          f"(기록만 -- 216 과 비교하지 않음, 오프라인 다채널 형태와 다른 게 정상)")
    print(f"n_transitions_by_type 전체: {replay['ap_core'].n_transitions_by_type}")

    ok = (not degenerate) and (n_only_one > 0)
    print(f"검증 ② 종합: {ok}")
    return ok


def check_ai_tensor_assembly(replay: dict) -> bool:
    print("=" * 70)
    print("[verify_mc] 검증 ③ AI: manifest 순서 (1,3,14) 텐서 조립 -- 가장 중요")
    print("=" * 70)
    manifest_path = CKPT_DIR / "manifest.json"
    import json
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"manifest['channels']: {manifest['channels']}  (기대: {CHANNELS})")

    windows = pd.read_parquet(
        wp.C_PD / "predict_v0" / "dataset_v0" /
        "windows_metop03_omni_p6-pro_tel0_p5-pro_tel90_p5.parquet")
    ts_primary = pd.read_parquet(
        wp.C_PD / "predict_v0" / "dataset_v0" / "timeseries_metop03_omni_p6.parquet"
    ).set_index("time").index   # 오프라인 lag 구조가 기준으로 삼는 gap-collapsed 축

    sample = windows.sample(100, random_state=7)
    z_history = replay["z_history"]
    max_diff = 0.0
    n_bad = 0
    n_skipped = 0
    for _, row in sample.iterrows():
        wet = row["window_end_time"]
        pos = ts_primary.get_loc(wet)
        if pos < 13:
            n_skipped += 1
            continue
        lag_ts = ts_primary[pos - 13: pos + 1]   # 오래된 -> 최신, 14개(오프라인과 동일 축)

        try:
            online = np.array([[z_history[ch][t] for t in lag_ts] for ch in manifest["channels"]],
                              dtype=float)
        except KeyError:
            n_skipped += 1
            continue
        if np.isnan(online).any():
            n_skipped += 1
            continue

        expected = np.array([
            [row[f"z_ch{i}_lag{13 - j}"] for j in range(14)]
            for i in range(len(manifest["channels"]))
        ], dtype=float)

        diff = np.abs(online - expected).max()
        max_diff = max(max_diff, diff)
        if diff > 1e-9:
            n_bad += 1

    print(f"표본 100개 중 스킵(온라인 NaN/워밍업 등) {n_skipped}개, 실비교 {100 - n_skipped}개")
    print(f"최대 절대오차: {max_diff:.3e}, 불일치(>1e-9) {n_bad}개")
    ok_numeric = (100 - n_skipped) > 0 and max_diff < 1e-9 and n_bad == 0

    # ── 이름 기반 매핑 합성 확인: update() 호출 순서를 일부러 뒤섞는다 ──
    zbuf = ai.ZBuffer(list(manifest["channels"]), maxlen=3)
    shuffled_updates = [
        ("pro_tel90_p5", 30.0), ("omni_p6", 10.0), ("pro_tel0_p5", 20.0),
        ("pro_tel90_p5", 31.0), ("omni_p6", 11.0), ("pro_tel0_p5", 21.0),
        ("pro_tel90_p5", 32.0), ("omni_p6", 12.0), ("pro_tel0_p5", 22.0),
    ]
    for ch, v in shuffled_updates:
        zbuf.update(ch, v)
    X = zbuf.assemble()
    expected_shuffle = np.array([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0], [30.0, 31.0, 32.0]])
    name_mapping_ok = np.allclose(X[0], expected_shuffle)
    print(f"이름 기반 매핑 합성 확인: update 호출 순서를 섞어도(pro_tel90_p5 먼저) "
          f"assemble() 결과가 manifest 순서(omni_p6,pro_tel0_p5,pro_tel90_p5) 를 지키는가: "
          f"{name_mapping_ok}")
    print(f"  결과: {X[0].tolist()}  (기대: {expected_shuffle.tolist()})")

    ok = ok_numeric and name_mapping_ok
    print(f"검증 ③ 종합: {ok}")
    return ok


def check_ai_3ch_inference() -> bool:
    print("=" * 70)
    print("[verify_mc] 검증 ④ AI: 3채널 체크포인트(input_size=3) 로드 + 추론")
    print("=" * 70)
    model, manifest = ai.load_single_model(CKPT_DIR, fold=0)
    print(f"모델 로드 성공. input_size={manifest['input_size']} channels={manifest['channels']}")

    rng = np.random.default_rng(0)
    X = rng.normal(size=(1, 3, 14)).astype(np.float32)
    xb = torch.tensor(X, dtype=torch.float32)
    p_event, y_pred, per_model_ms = ai.infer_from_tensor([model], manifest, xb)
    print(f"추론 성공: p_event={p_event:.4f} y_pred={y_pred} time={per_model_ms[0]:.3f}ms")

    ok = (0.0 <= p_event <= 1.0) and y_pred in (0, 1)
    print(f"검증 ④ 종합: {ok}")
    return ok


def main():
    print(f"[verify_mc] 3채널 전 구간 리플레이 시작 -- {CHANNELS}")
    replay = run_3ch_replay()
    print(f"[verify_mc] 리플레이 완료: {len(replay['schedule'])}틱")

    r1 = check_wp_independence(replay)
    r2 = check_ap_or_combination(replay)
    r3 = check_ai_tensor_assembly(replay)
    r4 = check_ai_3ch_inference()

    print("=" * 70)
    print(f"[verify_mc] 종합: ①WP독립={r1} ②AP-OR={r2} ③AI텐서조립={r3} ④AI3ch추론={r4}")
    print(f"[verify_mc] ③이 가장 중요(수치 목표 있음) -- 통과 못 하면 시나리오 (d) 무효.")
    print(f"[verify_mc] 전체 통과: {all([r1, r2, r3, r4])}")


if __name__ == "__main__":
    main()
