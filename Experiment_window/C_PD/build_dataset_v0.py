"""
build_dataset_v0.py
====================
manual_labels_{detector}_{channel}.csv (check_manual_labels.py로 검증된 손라벨)로부터
예측 모델 착수용 데이터셋 v0을 만든다.

재사용 (재구현 없음 -- import만):
  check_manual_labels.py              : load_manual_labels, build_reconciled_events,
      report_completeness(의 split_pairs 병합 로직) -- event 구간 재구성에 그대로 재사용
  label_events_gui.py                 : load_channel_series (count 로드)
  fsm_count_spe_quietoff_mad_poes.py  : compute_rolling_bg (bg_median/std)
  diag_rolling_threshold_poes.py의 _draw_zscore_panel과 동일 공식: z=(count-bg_median)/bg_std
      (그 함수 자체는 plotting 전용이라 import 대신 동일 공식만 재사용 -- 계산 로직은
      compute_rolling_bg 결과를 그대로 씀, 새로 추정하는 부분 없음)

출력 2종 (--out-dir, 기본 manual_labels/dataset_v0/):
  timeseries.parquet : 15분 격자 전체(라벨 있는 채널 count 시계열 전 구간) --
      time, count, zscore, label(0/1), event_id(-1=quiet), ambiguous_peak, peak_below_bg
  windows.parquet     : 위 timeseries에서 2궤도(기본 14샘플) 슬라이딩 윈도우로 만든
      학습 샘플 -- window_end_time, z_lag13..z_lag0, label, event_id, ambiguous_peak,
      peak_below_bg (event_id는 GroupKFold 등 이벤트 단위 분할용으로 유지)

사용:
  python build_dataset_v0.py --detector metop03 --channel omni_p6 --window 14 --bg-window-days 7
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "POES"))
sys.path.insert(0, str(HERE / "POES" / "count_FSM"))

from label_events_gui import load_channel_series           # noqa: E402
from check_manual_labels import load_manual_labels, report_completeness, build_reconciled_events  # noqa: E402
import fsm_count_spe_quietoff_mad_poes as fsm_engine        # noqa: E402


def build_event_spans(events: pd.DataFrame, peak_lt2_ids: set) -> pd.DataFrame:
    """완결 이벤트 테이블 -> [event_id, onset_time, end_time, ambiguous_peak, peak_below_bg]."""
    complete = events.dropna(subset=["onset_time", "end_time"]).copy()
    complete["ambiguous_peak"] = complete["event_id"].isin(peak_lt2_ids)
    return complete[["event_id", "onset_time", "end_time", "ambiguous_peak"]]


def make_timeseries(cnt: pd.Series, bg: pd.DataFrame, spans: pd.DataFrame) -> pd.DataFrame:
    std = bg["bg_std"].reindex(cnt.index).ffill().replace(0, np.nan)
    med = bg["bg_median"].reindex(cnt.index).ffill()
    z = (cnt - med) / std

    label = pd.Series(0, index=cnt.index, dtype=int)
    event_id = pd.Series(-1, index=cnt.index, dtype=int)
    ambiguous = pd.Series(False, index=cnt.index, dtype=bool)

    # onset_time 오름차순으로 적용 -- 겹치는 구간은 나중(늦게 시작한) 이벤트가 이미
    # 찍힌 앞 이벤트의 event_id를 덮어쓰지 않도록, 먼저 온 이벤트부터 채우고 겹치는
    # 뒤 이벤트는 label/ambiguous만 OR로 반영(그 구간의 '대표 event_id'는 먼저 시작한
    # 이벤트로 고정 -- 1-c 겹침 리포트와 별개로, 데이터셋에서는 임의 규칙이 필요해서
    # '더 먼저 시작한 이벤트에 귀속' 규칙을 명시적으로 채택).
    for _, r in spans.sort_values("onset_time").iterrows():
        mask = (cnt.index >= r["onset_time"]) & (cnt.index <= r["end_time"])
        label.loc[mask] = 1
        ambiguous.loc[mask] = ambiguous.loc[mask] | bool(r["ambiguous_peak"])
        unassigned = mask & (event_id == -1)
        event_id.loc[unassigned] = int(r["event_id"])

    ts = pd.DataFrame({
        "time": cnt.index, "count": cnt.values, "zscore": z.values,
        "label": label.values, "event_id": event_id.values, "ambiguous_peak": ambiguous.values,
    }).set_index("time")
    return ts


def make_windows(ts: pd.DataFrame, window: int) -> pd.DataFrame:
    z = ts["zscore"].values
    n = len(ts)
    if n < window:
        return pd.DataFrame()
    # sliding_window_view: 행 i = z[i-window+1 .. i] (라벨/이벤트id는 윈도우 끝 시점 기준)
    windows = np.lib.stride_tricks.sliding_window_view(z, window)  # shape (n-window+1, window)
    end_idx = np.arange(window - 1, n)
    cols = {f"z_lag{window-1-j}": windows[:, j] for j in range(window)}
    out = pd.DataFrame(cols)
    out.insert(0, "window_end_time", ts.index[end_idx])
    out["label"] = ts["label"].values[end_idx]
    out["event_id"] = ts["event_id"].values[end_idx]
    out["ambiguous_peak"] = ts["ambiguous_peak"].values[end_idx]
    n_before = len(out)
    out = out.dropna().reset_index(drop=True)
    n_dropped = n_before - len(out)
    print(f"[dataset_v0] 윈도우 {n_before}개 생성, NaN(z-score 미확보 구간) {n_dropped}개 제외 "
          f"-> 최종 {len(out)}개")
    return out


def main():
    ap = argparse.ArgumentParser(description="manual_labels -> 예측 모델 착수용 데이터셋 v0")
    ap.add_argument("--detector", default="metop03")
    ap.add_argument("--channel", default="omni_p6")
    ap.add_argument("--window", type=int, default=14, help="2궤도 ~= 206분 / 15분 샘플 ~= 14")
    ap.add_argument("--bg-window-days", type=int, default=7, help="rolling bg 창(일) -- check_manual_labels와 동일 w 권장")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    labels_path = HERE / "manual_labels" / f"manual_labels_{args.detector}_{args.channel}.csv"
    df = load_manual_labels(labels_path)
    comp = report_completeness(df)
    events = build_reconciled_events(df, comp["split_pairs"])

    cnt, _ = load_channel_series(args.detector, args.channel)
    bg = fsm_engine.compute_rolling_bg(cnt, args.bg_window_days, None, fsm_engine.BG_UPDATE_FREQ)

    peak_lt2_ids = set(events.loc[events["peak_count"] < 2, "event_id"])
    spans = build_event_spans(events, peak_lt2_ids)
    print(f"\n[dataset_v0] 이벤트 구간 {len(spans)}개 (ambiguous_peak={int(spans['ambiguous_peak'].sum())}개), "
          f"count 시계열 {len(cnt)}행에 라벨링")

    ts = make_timeseries(cnt, bg, spans)
    n_pos = int((ts["label"] == 1).sum())
    print(f"[dataset_v0] timeseries: label=1 {n_pos}행 ({n_pos/len(ts)*100:.2f}%), "
          f"label=0 {len(ts)-n_pos}행")

    windows = make_windows(ts, args.window)
    n_pos_w = int((windows["label"] == 1).sum())
    print(f"[dataset_v0] windows: label=1 {n_pos_w}개 ({n_pos_w/len(windows)*100:.2f}%), "
          f"label=0 {len(windows)-n_pos_w}개, 고유 event_id {windows.loc[windows['event_id']>=0,'event_id'].nunique()}개")

    out_dir = Path(args.out_dir) if args.out_dir else HERE / "manual_labels" / "dataset_v0"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts.reset_index().to_parquet(out_dir / f"timeseries_{args.detector}_{args.channel}.parquet", index=False)
    windows.to_parquet(out_dir / f"windows_{args.detector}_{args.channel}.parquet", index=False)
    print(f"\n[dataset_v0] 저장 완료 -> {out_dir}")


if __name__ == "__main__":
    main()
