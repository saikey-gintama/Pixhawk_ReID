"""
1b_list_fsm_diffs.py
==================
1_check_labels.py 산출물(fsm_only_no_manual.csv/fsm_missed_manual.csv)을 사람이
직접 열어보고 GUI로 재검토하기 쉽게 날짜/보조정보를 채워 재저장 + FSM이 강한
이벤트를 놓친 이유 예비 진단(배경창 부풀림 가설 확인).

재사용 (재구현 없음 -- import만):
  1_check_labels.py : load_manual_labels, build_reconciled_events,
      report_completeness(split_pairs), _fsm_onset_csv_path, MATCH_TOL_H
  0_label_events_gui.py    : load_channel_series
  fsm_count_spe_quietoff_mad_poes.py : compute_rolling_bg/build_threshold/detect_segments
      (예비 진단에서 threshold vs count 재현 -- 새 계산식 아님, 기존 엔진 그대로)

출력 (predict_v0/quality_check/):
  A_fsm_only_no_manual.csv   : FSM만 잡음(7개) -- onset/peak 시각, peak_count, maglat,
      in_saa, 가장 가까운 손라벨까지 시간차(부호: +=손라벨이 나중, -=손라벨이 먼저)
  B_fsm_missed_manual.csv    : 손라벨만 잡음(32개, FSM 놓침) -- peak_count 내림차순

사용:
  python 1b_list_fsm_diffs.py --detector metop03 --channel omni_p6
"""
from __future__ import annotations
import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent   # C_PD/predict_v0/
C_PD = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(C_PD / "POES"))
sys.path.insert(0, str(C_PD / "POES" / "count_FSM"))

# "0_label_events_gui"/"1_check_labels"는 숫자로 시작해 `import` 문으로 직접 못 씀 --
# importlib.import_module로 파일명 그대로 로드(1_check_labels.py 참고).
_check = importlib.import_module("1_check_labels")
load_manual_labels = _check.load_manual_labels
report_completeness = _check.report_completeness
build_reconciled_events = _check.build_reconciled_events
MATCH_TOL_H = _check.MATCH_TOL_H
_FSM_ONSET_CSV = _check._fsm_onset_csv_path(7, 7)  # 기존 하드코딩 w7k7과 동일 기본값 유지
load_channel_series = importlib.import_module("0_label_events_gui").load_channel_series
import fsm_count_spe_quietoff_mad_poes as fsm_engine  # noqa: E402


def _nearest_manual(t, manual_onsets: pd.Series):
    """t 에 가장 가까운 손라벨 onset까지의 부호 있는 시간차(시간) + 그 event_id.
    +값=손라벨이 t보다 나중, -값=손라벨이 t보다 먼저."""
    dh = (manual_onsets.values - pd.Timestamp(t).to_datetime64()) / np.timedelta64(1, "h")
    i = int(np.argmin(np.abs(dh)))
    return float(dh[i]), int(manual_onsets.index[i])


def build_listing_A(df: pd.DataFrame, manual_events: pd.DataFrame) -> pd.DataFrame:
    fsm_raw = pd.read_csv(_FSM_ONSET_CSV, parse_dates=["onset_time", "peak_time", "end_time"])
    fsm_raw = fsm_raw[fsm_raw["channel"] == "omni_p6"].reset_index(drop=True)
    if fsm_raw["onset_time"].dt.tz is None:
        fsm_raw["onset_time"] = fsm_raw["onset_time"].dt.tz_localize("UTC")
    import _match_core_poes as core
    clusters = core._cluster_indices(fsm_raw["onset_time"].values, MATCH_TOL_H)
    fsm_events = fsm_raw.iloc[[int(g[0]) for g in clusters]].copy()

    manual = manual_events.dropna(subset=["onset_time"])
    manual_onsets = pd.Series(manual["onset_time"].values, index=manual["event_id"].values)

    fsm_hit = core.det_matched_mask(fsm_events, manual.set_index("onset_time"), MATCH_TOL_H)
    fsm_only = fsm_events[~fsm_hit].copy()

    gaps, nearest_ids = [], []
    for t in fsm_only["onset_time"]:
        gap_h, eid = _nearest_manual(t, manual_onsets)
        gaps.append(round(gap_h, 1))
        nearest_ids.append(eid)
    fsm_only["nearest_manual_event_id"] = nearest_ids
    fsm_only["nearest_manual_gap_h"] = gaps
    fsm_only["onset_date"] = fsm_only["onset_time"].dt.strftime("%Y-%m-%d")
    fsm_only["peak_date"] = fsm_only["peak_time"].dt.strftime("%Y-%m-%d")

    cols = ["onset_date", "onset_time", "peak_date", "peak_time", "peak_count",
            "onset_maglat", "in_saa", "nearest_manual_event_id", "nearest_manual_gap_h"]
    out = fsm_only[cols].rename(columns={"onset_maglat": "maglat"}).sort_values("onset_time")
    return out.reset_index(drop=True)


def build_listing_B(df: pd.DataFrame, manual_events: pd.DataFrame) -> pd.DataFrame:
    fsm_raw = pd.read_csv(_FSM_ONSET_CSV, parse_dates=["onset_time", "peak_time", "end_time"])
    fsm_raw = fsm_raw[fsm_raw["channel"] == "omni_p6"].reset_index(drop=True)
    if fsm_raw["onset_time"].dt.tz is None:
        fsm_raw["onset_time"] = fsm_raw["onset_time"].dt.tz_localize("UTC")
    import _match_core_poes as core
    clusters = core._cluster_indices(fsm_raw["onset_time"].values, MATCH_TOL_H)
    fsm_events = fsm_raw.iloc[[int(g[0]) for g in clusters]].copy()

    manual = manual_events.dropna(subset=["onset_time"]).copy()
    manual_hit = core.det_matched_mask(manual, fsm_events.set_index("onset_time"), MATCH_TOL_H)
    missed = manual[~manual_hit].copy()

    # in_saa는 원본 raw 라벨(각 event_id의 'o' 행)에서 그대로 조회 -- tag_onset_geo가
    # onset 시점에 이미 태깅해 둔 값, 재계산 없음.
    onset_saa = df[df["label_type"] == "o"].set_index("event_id")["in_saa"]
    missed["in_saa"] = missed["event_id"].map(onset_saa)
    missed["onset_date"] = missed["onset_time"].dt.strftime("%Y-%m-%d")
    missed["peak_date"] = missed["peak_time"].dt.strftime("%Y-%m-%d")

    cols = ["event_id", "onset_date", "onset_time", "peak_date", "peak_time",
            "peak_count", "in_saa"]
    out = missed[cols].sort_values("peak_count", ascending=False)
    return out.reset_index(drop=True)


def diagnose_threshold(cnt: pd.Series, w: int, k: float, onset_floor: float,
                        event_row: pd.Series, label: str):
    print(f"\n--- {label}: event {int(event_row['event_id'])} "
          f"(onset={event_row['onset_time']}, peak={event_row['peak_time']}, "
          f"peak_count={event_row['peak_count']:.1f}) ---")
    bg = fsm_engine.compute_rolling_bg(cnt, w, None, fsm_engine.BG_UPDATE_FREQ)
    thr = fsm_engine.build_threshold(bg, k, onset_floor)
    win = cnt.loc[event_row["onset_time"] - pd.Timedelta(hours=6):
                  event_row["peak_time"] + pd.Timedelta(hours=6)]
    thr_win = thr.reindex(win.index).ffill()
    bg_med_win = bg["bg_median"].reindex(win.index).ffill()
    bg_sig_win = bg["bg_std"].reindex(win.index).ffill()
    trace = pd.DataFrame({"count": win, "threshold": thr_win,
                          "bg_median": bg_med_win, "bg_sigma": bg_sig_win,
                          "above_threshold": win >= thr_win})
    n_above = int(trace["above_threshold"].sum())
    max_run = 0
    run = 0
    for v in trace["above_threshold"]:
        run = run + 1 if v else 0
        max_run = max(max_run, run)
    print(f"  bg_median~{bg_med_win.mean():.2f}  bg_sigma~{bg_sig_win.mean():.2f}  "
          f"threshold~{thr_win.mean():.2f} (=bg_median+{k}*sigma, floor={onset_floor})")
    print(f"  이 구간에서 count>=threshold 인 15분 샘플: {n_above}/{len(trace)}, "
          f"최장 연속 구간: {max_run}샘플({max_run*15}분) "
          f"(MIN_SPE_DURATION_H={fsm_engine.MIN_SPE_DURATION_H}h = {int(fsm_engine.MIN_SPE_DURATION_H*4)}샘플 필요)")
    if n_above == 0:
        print("  => 진단: count가 threshold를 아예 넘지 못함 -- peak가 강해도 배경(bg_median+7*sigma)이 "
              "그보다 더 높게 부풀어 있었음 (7일 배경창 부풀림 가설 지지).")
    elif max_run < fsm_engine.MIN_SPE_DURATION_H * 4:
        print(f"  => 진단: threshold는 잠깐 넘었지만({max_run*15}분) 최소 지속시간"
              f"({fsm_engine.MIN_SPE_DURATION_H}h) 미달로 detect_segments가 세그먼트로 인정 안 함 "
              "-- 배경 부풀림보다는 '짧은 스파이크' 문제.")
    else:
        print("  => 진단: threshold를 min_duration 이상 넘었는데도 fsm_onset 산출물엔 없음 "
              "-- 클러스터링/필터 단계 재확인 필요(예상 밖 케이스).")
    print(trace[trace["above_threshold"] | (trace.index == event_row["peak_time"])].to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="metop03")
    ap.add_argument("--channel", default="omni_p6")
    ap.add_argument("--out-dir", default=str(HERE / "quality_check"))
    args = ap.parse_args()

    labels_path = HERE / "manual_labels" / f"manual_labels_{args.detector}_{args.channel}.csv"
    df = load_manual_labels(labels_path)
    comp = report_completeness(df)
    events = build_reconciled_events(df, comp["split_pairs"])

    print("\n" + "=" * 70)
    print("[A] FSM-only (사람이 안 본 것)")
    print("=" * 70)
    A = build_listing_A(df, events)
    print(A.to_string(index=False))

    print("\n" + "=" * 70)
    print("[B] 사람만 잡음 (FSM 놓침), peak_count 내림차순")
    print("=" * 70)
    B = build_listing_B(df, events)
    print(B.to_string(index=False))
    n_gt500 = int((B["peak_count"] > 500).sum())
    print(f"\npeak_count > 500 인 것: {n_gt500}개 / 총 {len(B)}개")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    A.to_csv(out_dir / "A_fsm_only_no_manual.csv", index=False)
    B.to_csv(out_dir / "B_fsm_missed_manual.csv", index=False)
    print(f"\n[list_fsm_diffs] 저장 -> {out_dir}/A_fsm_only_no_manual.csv, B_fsm_missed_manual.csv")

    print("\n" + "=" * 70)
    print("[C] peak>500인데 FSM이 놓친 이벤트 -- threshold vs count 예비 진단 (w7k7 on0.1)")
    print("=" * 70)
    cnt, _ = load_channel_series(args.detector, args.channel)
    top2 = B[B["peak_count"] > 500].sort_values("peak_count", ascending=False).head(2)
    events_by_id = events.set_index("event_id")
    for _, r in top2.iterrows():
        row = events_by_id.loc[r["event_id"]].copy()
        row["event_id"] = r["event_id"]
        diagnose_threshold(cnt, w=7, k=7.0, onset_floor=0.1, event_row=row, label="event")


if __name__ == "__main__":
    main()
