"""
fsm_count_spe_quiet7_std.py
==========================
Count 단독 SPE onset 탐지 — 전 채널 일괄, 단일 방식 고정판 (quiet7 + std).

방식 비교 실험용 4종 중 하나. 검출 엔진(rolling 배경 → median+k·σ → 연속 초과
구간 → peak_floor 사후필터)은 4종이 모두 동일하며, 배경 산포 추정만 다르다:

  fsm_count_spe_quiet7_mad.py    quiet 7일 선택 + MAD robust σ   (= Löwe식 BL-C2 계열)
  fsm_count_spe_quietoff_mad.py  윈도우 전체     + MAD robust σ   (= 제안 방식 Ours)
  fsm_count_spe_quiet7_std.py    quiet 7일 선택 + 표본 std        (비교 baseline)
  fsm_count_spe_quietoff_std.py  윈도우 전체     + 표본 std        (비교 baseline)

이 파일은 [quiet7 + std]:
  - BG_QUIET_DAYS = 7
  - SIGMA_METHOD  = "std" (sample std (ddof=1))

출력 CSV는 noaa_goes_spe_match.py가 그대로 먹도록 ana6 sweep_event.csv와 동일
스키마. 파일명에 방식 태그가 붙어 4종이 섞이지 않는다.
  fsm_onset_quiet7_std.csv / fsm_event_quiet7_std.csv

자립성: ksem_flux_config / ksem_common 비의존. count 로드만 ksem_io 사용.
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ── ksem_io 경로 (count 로드 전용) ────────────────────────────────
_THIS_DIR         = Path(__file__).parent.resolve()
KSEM_COUNT_DIR    = _THIS_DIR.parent / "KSEM_count"
COUNT_PARQUET_DIR = KSEM_COUNT_DIR / "ksem_cache_parquet"
if str(KSEM_COUNT_DIR) not in sys.path:
    sys.path.insert(0, str(KSEM_COUNT_DIR))
import ksem_io   # noqa: E402

# ══════════════════════════════════════════════════════════════════
# 파라미터 (이 블록만 조정 — 검출 로직 본문엔 숫자를 박지 않는다)
# ══════════════════════════════════════════════════════════════════
RESAMPLE_FREQ      = "15min"
BG_UPDATE_FREQ     = "1D"
MIN_SPE_DURATION_H = 1

# ── 이 파일의 방식 고정 ──────────────────────────────────────────
BG_WINDOW_DAYS = 30        # [day] 배경용 슬라이딩 윈도우 (콜드스타트 실험 시 이 값 변경)
BG_QUIET_DAYS  = 7        # 가장 조용한 7일만 사용
SIGMA_METHOD   = "std"      # sample std (ddof=1)

PROTON_LOGICS   = ["O", "OU", "OUT", "CR"]
ELECTRON_LOGICS = ["F", "FT", "FTU", "FTUO"]
TARGET_LOGICS   = PROTON_LOGICS + ELECTRON_LOGICS
PD_KEYS         = ["PD1", "PD2", "PD3"]
SIDES           = ["A", "B"]

K           = 10
ONSET_FLOOR = 0.5
PEAK_FLOOR  = 2.0
MIN_PTS_PER_CHANNEL = 100


def parse_args():
    """파라미터만 argument로 받는다. 미지정 시 위 기본값 사용.
    출력 폴더/파일명은 TAG + 이 값들에서 자동 생성되므로 직접 지정 불필요."""
    p = argparse.ArgumentParser(
        description=f"FSM count SPE detector [{TAG}] — params override defaults, "
                    f"output names auto-built from TAG+params.")
    p.add_argument("--window", type=int,   default=BG_WINDOW_DAYS,
                   help=f"background sliding window [day] (default {BG_WINDOW_DAYS})")
    p.add_argument("--k",      type=float, default=K,
                   help=f"threshold multiplier median+k*sigma (default {K})")
    p.add_argument("--onset",  type=float, default=ONSET_FLOOR,
                   help=f"onset floor, threshold lower clip (default {ONSET_FLOOR})")
    p.add_argument("--peak",   type=float, default=PEAK_FLOOR,
                   help=f"peak floor, post-filter (default {PEAK_FLOOR})")
    p.add_argument("--quiet-days", type=int, default=(BG_QUIET_DAYS if BG_QUIET_DAYS else 0),
                   help="quiet-day selection N; 0 = use whole window "
                        f"(default {BG_QUIET_DAYS if BG_QUIET_DAYS else 0})")
    return p.parse_args()


def build_runtag(tag, window, k, onset, peak):
    """출력 파일명용 파라미터 문자열. 예: quietoff_mad_w30_k10_on0.5_pk2.0"""
    return f"{tag}_w{window}_k{_numstr(k)}_on{_numstr(onset)}_pk{_numstr(peak)}"  # see _numstr


def _numstr(v):
    """10.0 -> '10', 0.5 -> '0.5' (파일명 깔끔하게)."""
    f = float(v)
    return str(int(f)) if f.is_integer() else str(f)


FSM_OUTPUT_DIR = _THIS_DIR / "fsm_output"
FSM_OUTPUT_DIR.mkdir(exist_ok=True)

TAG = "quiet7_std"   # 출력 파일명 태그


# ══════════════════════════════════════════════════════════════════
# 배경 산포: 표본 표준편차 (비교 baseline)
# ══════════════════════════════════════════════════════════════════
def _sigma(x) -> float:
    """표본 표준편차 σ (ddof=1). 고전적 σ method 계열.

    단발 스파이크가 quiet 표본에 섞이면 폭발하는 비강건 추정. MAD판과의
    대비로 robust 선택의 효과를 격리하기 위한 baseline.
    """
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    return float(np.std(x, ddof=1))


# ══════════════════════════════════════════════════════════════════
# 배경 추정 + 검출 엔진 (4종 공통)
# ══════════════════════════════════════════════════════════════════
def _select_quiet_samples(cnt: pd.Series, t_ref: pd.Timestamp,
                          bg_window_days: int, bg_quiet_days) -> pd.Series:
    """t_ref 이전 bg_window_days 윈도우의 배경 표본.
    bg_quiet_days=None이면 윈도우 전체, 정수면 하루 median이 가장 낮은 N일만."""
    seg = cnt.loc[t_ref - pd.Timedelta(days=bg_window_days):t_ref].dropna()
    if len(seg) < 10:
        return pd.Series(dtype=float)
    if bg_quiet_days is None:
        return seg
    daily = seg.resample("1D").median().dropna()
    if len(daily) < bg_quiet_days:
        return pd.Series(dtype=float)
    quiet_days = daily.nsmallest(bg_quiet_days)
    return seg[seg.index.normalize().isin(quiet_days.index.normalize())]


def compute_rolling_bg(cnt: pd.Series, bg_window_days: int,
                       bg_quiet_days, update_freq: str = "1D") -> pd.DataFrame:
    """rolling 배경 시계열. update_freq마다 갱신 후 ffill.
    반환: DataFrame[bg_median, bg_std]. bg_std는 이 파일의 _sigma."""
    if cnt.empty:
        return pd.DataFrame(columns=["bg_median", "bg_std"])
    update_points = pd.date_range(cnt.index[0], cnt.index[-1], freq=update_freq)
    rows = []
    for t in update_points:
        q = _select_quiet_samples(cnt, t, bg_window_days, bg_quiet_days)
        if len(q) >= 2:
            rows.append((t, float(q.median()), _sigma(q)))
        elif len(q) == 1:
            rows.append((t, float(q.median()), np.nan))
        else:
            rows.append((t, np.nan, np.nan))
    bg = pd.DataFrame(rows, columns=["t", "bg_median", "bg_std"]).set_index("t")
    return bg.reindex(cnt.index.union(bg.index)).ffill().reindex(cnt.index)


def build_threshold(bg: pd.DataFrame, k: float, onset_floor: float) -> pd.Series:
    """임계 = (bg_median + k·σ)를 onset_floor로 하한 클립."""
    th = bg["bg_median"] + k * bg["bg_std"]
    if onset_floor > 0:
        th = th.clip(lower=onset_floor)
    return th


def detect_segments(cnt: pd.Series, thresh_series: pd.Series,
                    bg: pd.DataFrame, min_duration_h: float) -> list:
    """count >= rolling 임계가 min_duration_h 이상 연속인 구간. peak_floor는 호출부에서."""
    th  = thresh_series.reindex(cnt.index).ffill()
    med = bg["bg_median"].reindex(cnt.index).ffill()
    sig = bg["bg_std"].reindex(cnt.index).ffill()
    above = (cnt >= th) & cnt.notna() & th.notna()
    segs = []; in_ev = False; onset = None
    def _close(seg, onset_t, end_t):
        if (end_t - onset_t).total_seconds() / 3600 < min_duration_h:
            return
        segs.append({
            "onset_time": onset_t, "peak_time": seg.idxmax(), "end_time": end_t,
            "onset_count": round(float(cnt.loc[onset_t]), 3),
            "peak_count":  round(float(seg.max()), 3),
            "end_count":   round(float(cnt.loc[end_t]), 3),
            "duration_h":  round((end_t - onset_t).total_seconds() / 3600, 2),
            "bg_median": round(float(med.loc[onset_t]), 4) if pd.notna(med.loc[onset_t]) else np.nan,
            "bg_sigma":  round(float(sig.loc[onset_t]), 4) if pd.notna(sig.loc[onset_t]) else np.nan,
            "threshold": round(float(th.loc[onset_t]), 4),
        })
    for t, a in above.items():
        if a and not in_ev:
            in_ev, onset = True, t
        elif not a and in_ev:
            _close(cnt.loc[onset:t], onset, t); in_ev = False
    if in_ev and onset is not None:
        seg = cnt.loc[onset:]; _close(seg, onset, seg.index[-1])
    return segs


# ══════════════════════════════════════════════════════════════════
def load_count() -> pd.DataFrame:
    df_count, _ = ksem_io.load(COUNT_PARQUET_DIR)
    if not df_count.empty and df_count.index.tz is None:
        df_count.index = df_count.index.tz_localize("UTC")
    return df_count


def main():
    args = parse_args()
    window   = args.window
    k        = args.k
    onset_fl = args.onset
    peak_fl  = args.peak
    quiet_d  = args.quiet_days if args.quiet_days > 0 else None
    runtag   = build_runtag(TAG, window, k, onset_fl, peak_fl)
    out_dir  = FSM_OUTPUT_DIR / runtag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fsm:{TAG}] params: window={window}d k={k} onset={onset_fl} "
          f"peak={peak_fl} quiet_days={quiet_d}")
    print(f"[fsm:{TAG}] runtag: {runtag}")
    print(f"[fsm:{TAG}] Loading count data...")
    df_count = load_count()
    cnt_rs = df_count.resample(RESAMPLE_FREQ).mean()

    onset_rows, event_rows = [], []
    for pd_key in PD_KEYS:
        for side in SIDES:
            for logic in TARGET_LOGICS:
                try:
                    cnt = cnt_rs[pd_key, side, logic].dropna()
                except KeyError:
                    continue
                if len(cnt) < MIN_PTS_PER_CHANNEL:
                    continue
                chan = f"{pd_key}{side}-{logic}"
                print(f"\n[fsm:{TAG}] {chan}  (n={len(cnt)} pts)")
                bg   = compute_rolling_bg(cnt, window, quiet_d, BG_UPDATE_FREQ)
                thr  = build_threshold(bg, k, onset_fl)
                segs = detect_segments(cnt, thr, bg, MIN_SPE_DURATION_H)
                base = {"k": k, "onset_floor": onset_fl, "pd_key": pd_key,
                        "side": side, "logic": logic, "channel": chan}
                for s in segs:
                    onset_rows.append({**base, **s})
                passed = [s for s in segs if s["peak_count"] >= peak_fl]
                for s in passed:
                    event_rows.append({**base, "peak_floor": peak_fl, **s})
                print(f"    segs={len(segs)}  peak>={peak_fl}: {len(passed)}")

    ONSET_COLS = ["k","onset_floor","pd_key","side","logic","channel",
                  "onset_time","peak_time","end_time","onset_count","peak_count",
                  "end_count","duration_h","bg_median","bg_sigma","threshold"]
    EVENT_COLS = ["k","onset_floor","peak_floor","pd_key","side","logic","channel",
                  "onset_time","peak_time","end_time","onset_count","peak_count",
                  "end_count","duration_h","bg_median","bg_sigma","threshold"]
    df_on = pd.DataFrame(onset_rows, columns=ONSET_COLS)
    df_ev = pd.DataFrame(event_rows, columns=EVENT_COLS)
    out_on = out_dir / f"fsm_onset_{runtag}.csv"
    out_ev = out_dir / f"fsm_event_{runtag}.csv"
    df_on.to_csv(out_on, index=False)
    df_ev.to_csv(out_ev, index=False)
    print(f"\n[fsm:{TAG}] onset saved: {out_on}  ({len(df_on)} rows)")
    print(f"[fsm:{TAG}] event saved: {out_ev}  ({len(df_ev)} rows)")
    print(f"[fsm:{TAG}] quiet_days={quiet_d} SIGMA_METHOD={SIGMA_METHOD} "
          f"WINDOW={window}d K={k} floor={onset_fl} peak={peak_fl}")
    if not df_on.empty:
        n_on = df_on.groupby("channel").size().rename("n_onset")
        n_ev = (df_ev.groupby("channel").size().rename("n_event")
                if not df_ev.empty else pd.Series(dtype=int, name="n_event"))
        tbl = pd.concat([n_on, n_ev], axis=1).fillna(0).astype(int).sort_values("n_onset", ascending=False)
        print(f"\n[fsm:{TAG}] per-channel counts:")
        print(tbl.to_string())


if __name__ == "__main__":
    main()
