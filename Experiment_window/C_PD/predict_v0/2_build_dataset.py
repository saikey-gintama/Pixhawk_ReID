"""
2_build_dataset.py
====================
manual_labels_{detector}_{channel}.csv (1_check_labels.py로 검증된 손라벨)로부터
예측 모델 v0(TCN 3클래스: quiet/rising/decreasing) 착수용 데이터셋을 만든다.

재사용 (재구현 없음 -- import만):
  1_check_labels.py                    : load_manual_labels, build_reconciled_events,
      report_completeness(의 split_pairs 병합 로직) -- event 구간 재구성에 그대로 재사용
  0_label_events_gui.py                : load_channel_series (count 로드)
  fsm_count_spe_quietoff_mad_poes.py   : compute_rolling_bg (bg_median/std)
  diag_rolling_threshold_poes.py의 _draw_zscore_panel과 동일 공식: z=(count-bg_median)/bg_std
      (그 함수 자체는 plotting 전용이라 import 대신 동일 공식만 재사용 -- 계산 로직은
      compute_rolling_bg 결과를 그대로 씀, 새로 추정하는 부분 없음)

라벨 (3상태, onset/peak/end 3점에서 규칙으로 생성):
  quiet=0      : 이벤트 구간 밖 전부
  rising=1     : onset ~ peak (peak 포함)
  decreasing=2 : peak ~ end (peak 미포함, end 포함)
  겹치는 이벤트(빈도 낮음, quality_check/overlapping_pairs.csv 참고)는 onset_time
  오름차순으로 라벨을 덮어써 나중(더 늦게 시작한) 이벤트의 상태가 우선한다 --
  물리적으로 새 이벤트의 rising이 이전 이벤트의 decreasing 꼬리보다 "현재 상태"를
  더 잘 대표한다고 보기 때문. event_id 귀속(그룹핑용)은 반대로 먼저 온 이벤트가
  선점(첫 이벤트 소유권 유지)한다 -- 이 비대칭은 기존 이진 라벨 스크립트의 관례를
  그대로 유지.

z-score 발산 가드 (중요 -- 조용한 저위도 구간에서 bg_std->0 으로 z가 발산하는 함정,
CUSUM 실험 때 겪은 것과 같은 계열):
  bg_std_safe = max(bg_std, Z_EPS)
  z = clip((count-bg_median)/bg_std_safe, -Z_CLIP, Z_CLIP)

이벤트 중심 크롭 + episode 그룹핑 (불균형 완화 + GroupKFold 시간 누수 방지):
  각 이벤트의 [onset-PAD_BEFORE_DAYS, end+PAD_AFTER_DAYS] 구간만 windows.parquet에
  남기고 이벤트에서 먼 quiet는 버린다. 인접한 두 이벤트의 padded 구간이 겹치면
  (예: 이벤트 간격이 2*10일보다 좁음) 하나의 episode로 병합한다 -- 슬라이딩
  윈도우(stride=1)는 인접 시각끼리 13/14 샘플을 공유하는 사실상의 near-duplicate라,
  병합하지 않으면 같은 episode의 시간적으로 인접한 윈도우가 GroupKFold에서
  train/val 양쪽에 걸쳐 들어가는 시간 누수가 생긴다. episode_id는 label용
  event_id(이벤트 구간 소유권, quiet=-1)와 별개로 windows.parquet에만 존재하며
  GroupKFold의 group 컬럼으로 쓴다(크롭 필터를 거친 모든 윈도우는 episode_id>=0).

출력 2종 (--out-dir, 기본 predict_v0/dataset_v0/):
  timeseries.parquet : 15분 격자 전체(라벨 있는 채널 count 시계열 전 구간, 크롭 없음) --
      time, count, zscore, label(0/1/2), event_id(-1=quiet), ambiguous_peak
  windows.parquet     : 위 timeseries에서 이벤트 중심 크롭 구간만, 2궤도(기본 14샘플)
      슬라이딩 윈도우(stride=1)로 만든 학습 샘플 -- window_end_time, z_lag13..z_lag0,
      label, event_id, episode_id, ambiguous_peak

사용:
  python 2_build_dataset.py --detector metop03 --channel omni_p6 --window 14 --bg-window-days 7
"""
from __future__ import annotations
import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent   # C_PD/predict_v0/
C_PD = HERE.parent                        # C_PD/ -- POES 등 공용 모듈 위치
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(C_PD / "POES"))
sys.path.insert(0, str(C_PD / "POES" / "count_FSM"))

# 숫자로 시작하는 모듈명은 `import` 문으로 직접 못 써서 importlib로 로드
# (0_label_events_gui.py / 1_check_labels.py 참고).
load_channel_series = importlib.import_module("0_label_events_gui").load_channel_series
_check = importlib.import_module("1_check_labels")
load_manual_labels = _check.load_manual_labels
report_completeness = _check.report_completeness
build_reconciled_events = _check.build_reconciled_events
import fsm_count_spe_quietoff_mad_poes as fsm_engine        # noqa: E402

# ── z-score 발산 가드 (진단 그림 보고 조정 가능하게 상수로 노출) ──────────────
Z_EPS = 1.0     # bg_std 하한(절대값). 조용한 저위도 구간에서 std->0 발산 방지.
Z_CLIP = 10.0   # z-score 절대값 상한.

# ── 케이던스/윈도우 (15분 격자 전제 -- 케이던스 바뀌면 재계산 필요) ──────────
WINDOW = 14     # ~2궤도(15분 x 14 = 210분)

# ── 이벤트 중심 크롭(불균형 완화) + episode 병합 폭 ──────────────────────────
PAD_BEFORE_DAYS = 10
PAD_AFTER_DAYS = 10

LABEL_QUIET, LABEL_RISING, LABEL_DECREASING = 0, 1, 2
LABEL_NAMES = {LABEL_QUIET: "quiet", LABEL_RISING: "rising", LABEL_DECREASING: "decreasing"}


def build_event_spans(events: pd.DataFrame, peak_lt2_ids: set) -> pd.DataFrame:
    """완결 이벤트 테이블 -> [event_id, onset_time, peak_time, end_time, ambiguous_peak]."""
    complete = events.dropna(subset=["onset_time", "peak_time", "end_time"]).copy()
    complete["ambiguous_peak"] = complete["event_id"].isin(peak_lt2_ids)
    return complete[["event_id", "onset_time", "peak_time", "end_time", "ambiguous_peak"]]


def make_timeseries(cnt: pd.Series, bg: pd.DataFrame, spans: pd.DataFrame) -> pd.DataFrame:
    std = bg["bg_std"].reindex(cnt.index).ffill()
    med = bg["bg_median"].reindex(cnt.index).ffill()
    std_safe = std.clip(lower=Z_EPS)
    z = ((cnt - med) / std_safe).clip(-Z_CLIP, Z_CLIP)

    label = pd.Series(LABEL_QUIET, index=cnt.index, dtype=int)
    event_id = pd.Series(-1, index=cnt.index, dtype=int)
    ambiguous = pd.Series(False, index=cnt.index, dtype=bool)

    for _, r in spans.sort_values("onset_time").iterrows():
        rising = (cnt.index >= r["onset_time"]) & (cnt.index <= r["peak_time"])
        decreasing = (cnt.index > r["peak_time"]) & (cnt.index <= r["end_time"])
        touched = rising | decreasing
        label.loc[rising] = LABEL_RISING
        label.loc[decreasing] = LABEL_DECREASING
        ambiguous.loc[touched] = ambiguous.loc[touched] | bool(r["ambiguous_peak"])
        unassigned = touched & (event_id == -1)
        event_id.loc[unassigned] = int(r["event_id"])

    ts = pd.DataFrame({
        "time": cnt.index, "count": cnt.values, "zscore": z.values,
        "label": label.values, "event_id": event_id.values, "ambiguous_peak": ambiguous.values,
    }).set_index("time")
    return ts


def build_episodes(spans: pd.DataFrame, pad_before_days: int, pad_after_days: int) -> pd.DataFrame:
    """이벤트별 [onset-pad_before, end+pad_after] 구간을 시간순으로 병합해 episode로 묶는다
    (겹치는 padded 구간은 하나의 episode_id -- GroupKFold 시간 누수 방지, 모듈 docstring 참고)."""
    pad_before = pd.Timedelta(days=pad_before_days)
    pad_after = pd.Timedelta(days=pad_after_days)
    ordered = spans.sort_values("onset_time").copy()
    ordered["crop_start"] = ordered["onset_time"] - pad_before
    ordered["crop_end"] = ordered["end_time"] + pad_after

    episodes = []
    cur_start = cur_end = None
    cur_events: list[int] = []
    for _, r in ordered.iterrows():
        if cur_start is None:
            cur_start, cur_end, cur_events = r["crop_start"], r["crop_end"], [int(r["event_id"])]
        elif r["crop_start"] <= cur_end:
            cur_end = max(cur_end, r["crop_end"])
            cur_events.append(int(r["event_id"]))
        else:
            episodes.append((cur_start, cur_end, cur_events))
            cur_start, cur_end, cur_events = r["crop_start"], r["crop_end"], [int(r["event_id"])]
    if cur_start is not None:
        episodes.append((cur_start, cur_end, cur_events))

    return pd.DataFrame([
        {"episode_id": i, "crop_start": s, "crop_end": e, "n_events": len(ev), "event_ids": ev}
        for i, (s, e, ev) in enumerate(episodes)
    ])


def assign_episode(times: pd.Series, episodes: pd.DataFrame) -> np.ndarray:
    ep_id = np.full(len(times), -1, dtype=int)
    times = pd.DatetimeIndex(times)
    for _, r in episodes.iterrows():
        mask = (times >= r["crop_start"]) & (times <= r["crop_end"])
        ep_id[mask] = r["episode_id"]
    return ep_id


def make_windows(ts: pd.DataFrame, window: int, episodes: pd.DataFrame) -> pd.DataFrame:
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
          f"-> {len(out)}개")

    out["episode_id"] = assign_episode(out["window_end_time"], episodes)
    n_all = len(out)
    out = out[out["episode_id"] >= 0].reset_index(drop=True)
    print(f"[dataset_v0] 이벤트 중심 크롭(+-{PAD_BEFORE_DAYS}/{PAD_AFTER_DAYS}일) 필터: "
          f"{n_all}개 -> {len(out)}개 (먼 quiet {n_all - len(out)}개 제외)")
    return out


def main():
    global Z_EPS, Z_CLIP
    ap = argparse.ArgumentParser(description="manual_labels -> 예측 모델 v0 데이터셋(3클래스)")
    ap.add_argument("--detector", default="metop03")
    ap.add_argument("--channel", default="omni_p6")
    ap.add_argument("--window", type=int, default=WINDOW, help="2궤도 ~= 206분 / 15분 샘플 ~= 14")
    ap.add_argument("--bg-window-days", type=int, default=7, help="rolling bg 창(일) -- 1_check_labels와 동일 w 권장")
    ap.add_argument("--pad-before-days", type=int, default=PAD_BEFORE_DAYS)
    ap.add_argument("--pad-after-days", type=int, default=PAD_AFTER_DAYS)
    ap.add_argument("--z-eps", type=float, default=Z_EPS)
    ap.add_argument("--z-clip", type=float, default=Z_CLIP)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    Z_EPS, Z_CLIP = args.z_eps, args.z_clip

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
    for lab, name in LABEL_NAMES.items():
        n = int((ts["label"] == lab).sum())
        print(f"[dataset_v0] timeseries: label={lab}({name}) {n}행 ({n/len(ts)*100:.2f}%)")

    episodes = build_episodes(spans, args.pad_before_days, args.pad_after_days)
    print(f"[dataset_v0] episode {len(episodes)}개 (이벤트 {len(spans)}개가 겹치는 padded 구간끼리 병합됨, "
          f"병합된 episode={int((episodes['n_events'] > 1).sum())}개)")

    windows = make_windows(ts, args.window, episodes)
    for lab, name in LABEL_NAMES.items():
        n = int((windows["label"] == lab).sum())
        print(f"[dataset_v0] windows: label={lab}({name}) {n}개 ({n/len(windows)*100:.2f}%)")
    print(f"[dataset_v0] windows 고유 event_id(라벨 소유) {windows.loc[windows['event_id']>=0,'event_id'].nunique()}개, "
          f"고유 episode_id(그룹) {windows['episode_id'].nunique()}개")

    out_dir = Path(args.out_dir) if args.out_dir else HERE / "dataset_v0"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts.reset_index().to_parquet(out_dir / f"timeseries_{args.detector}_{args.channel}.parquet", index=False)
    windows.to_parquet(out_dir / f"windows_{args.detector}_{args.channel}.parquet", index=False)
    print(f"\n[dataset_v0] 저장 완료 -> {out_dir}")


if __name__ == "__main__":
    main()
