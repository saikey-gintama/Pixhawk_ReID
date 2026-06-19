"""
fsm_count_spe_quietoff_mad_poes.py
==================================
Count 단독 SPE onset 탐지 — POES/MetOp SEM-2 판 (quietoff + MAD (Ours)).

KSEM 원본 fsm_count_spe_quietoff_mad.py 의 POES 이식판:
  - 검출 엔진 5함수(_sigma, _select_quiet_samples, compute_rolling_bg,
    build_threshold, detect_segments)는 KSEM 원본과 **byte 단위 동일**.
  - 바뀐 것은 입출력 어댑터뿐:
      · io: ksem_io → poes_{sat}_io (위성 인자화: --io)
      · 채널 루프: PD×side×logic → (species,direction,energy) MultiIndex 전체 22채널
      · geo 사후 태깅: onset 시점의 |B|/maglat 조회 → in_saa/onset_Bmag/onset_maglat
        (검출은 전 구간; 마스킹 아님. TPR 보존 + SAA발 false onset 식별)
  - 출력 CSV: KSEM 과 공통 컬럼 동일 + POES 전용 in_saa/onset_Bmag/onset_maglat 末尾 추가.
    noaa_goes_spe_match.py 는 공통 컬럼만 읽으므로 호환 유지.

방식 고정: BG_QUIET_DAYS=None (윈도우 전체), SIGMA_METHOD="mad".

사용:
  python fsm_count_spe_quietoff_mad_poes.py --io poes_metop03_io \
      --cache MetOp03_count/poes_metop03_cache_parquet \
      --window 30 --k 10 --onset 0.5 --peak 2
  python fsm_count_spe_quietoff_mad_poes.py --io poes_noaa19_io \
      --cache NOAA19_count/poes_noaa19_cache_parquet --window 7 --k 10 --onset 0.5 --peak 2
"""

from __future__ import annotations
import sys
import argparse
import importlib
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════
# 파라미터 (이 블록만 조정 — 검출 로직 본문엔 숫자를 박지 않는다)
# ══════════════════════════════════════════════════════════════════
RESAMPLE_FREQ      = "15min"
BG_UPDATE_FREQ     = "1D"
MIN_SPE_DURATION_H = 1

# ── 이 파일의 방식 고정 ──────────────────────────────────────────
BG_WINDOW_DAYS = 30        # [day] 배경용 슬라이딩 윈도우 (콜드스타트 실험 시 변경)
BG_QUIET_DAYS  = None      # None=윈도우 전체 사용
SIGMA_METHOD   = "mad"     # MAD robust sigma

K           = 10
ONSET_FLOOR = 0.5
PEAK_FLOOR  = 2.0
MIN_PTS_PER_CHANNEL = 100
SAA_BMAG_NT = 25000.0      # |B| < 이 값 → in_saa (coords_igrf 와 동일 기준)

TAG = "quietoff_mad"       # 출력 파일명 태그 (KSEM 과 동일 방식 태그)


def _import_io(io_arg: str, cache_dir: str):
    """io 모듈명/경로로 임포트. 캐시 경로 부모들을 sys.path 에 추가."""
    name = PureWindowsPath(io_arg).name if (io_arg and ('/' in io_arg or '\\' in io_arg)) else io_arg
    cache = Path(cache_dir).resolve()
    for cand in (cache if cache.is_dir() else cache.parent,
                 cache.parent, cache.parent.parent,
                 Path.cwd(), Path(__file__).resolve().parent):
        sp = str(cand)
        if sp not in sys.path:
            sys.path.insert(0, sp)
    return importlib.import_module(name)


def parse_args():
    p = argparse.ArgumentParser(
        description=f"FSM count SPE detector POES [{TAG}] — engine byte-identical to KSEM.")
    p.add_argument("--io", required=True, help="io 모듈명/경로 (poes_metop03_io | poes_noaa19_io)")
    p.add_argument("--cache", required=True, help="POES count parquet 캐시 디렉터리")
    p.add_argument("--out", default=None, help="출력 루트 (기본: 캐시 옆 fsm_output)")
    p.add_argument("--window", type=int,   default=BG_WINDOW_DAYS)
    p.add_argument("--k",      type=float, default=K)
    p.add_argument("--onset",  type=float, default=ONSET_FLOOR)
    p.add_argument("--peak",   type=float, default=PEAK_FLOOR)
    p.add_argument("--quiet-days", type=int, default=(BG_QUIET_DAYS if BG_QUIET_DAYS else 0))
    p.add_argument("--no-geo", action="store_true",
                   help="geo 태깅 생략(KSEM 처럼 in_saa 없이). geo 캐시 없을 때.")
    return p.parse_args()


def _numstr(v):
    f = float(v)
    return str(int(f)) if f.is_integer() else str(f)


def build_runtag(tag, window, k, onset, peak):
    return f"{tag}_w{window}_k{_numstr(k)}_on{_numstr(onset)}_pk{_numstr(peak)}"


# ══════════════════════════════════════════════════════════════════
# 배경 추정 + 검출 엔진 (KSEM 원본과 byte-동일 — 절대 수정 금지)
# ══════════════════════════════════════════════════════════════════
def _sigma(x) -> float:
    """robust σ = 1.4826 · median(|x - median(x)|).

    단발 스파이크에 강건. 저카운트 채널(OU/OUT)의 quiet 표본에 스파이크가
    섞여도 임계가 폭발하지 않는다. 표본 과반이 동일값이면 0이 될 수 있고,
    그 경우 검출은 floor에 의존(의도된 동작).
    """
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


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
# 입출력 어댑터 (POES 전용)
# ══════════════════════════════════════════════════════════════════
def load_count(io, cache_dir: str) -> pd.DataFrame:
    """POES count 캐시 로드 → MultiIndex DataFrame (UTC)."""
    df_count, _ = io.load(cache_dir)
    if not df_count.empty and df_count.index.tz is None:
        df_count.index = df_count.index.tz_localize("UTC")
    return df_count


def load_geo(io, cache_dir: str):
    """geo(lat/lon/alt + Bmag/maglat) 로드. 실패 시 None."""
    try:
        geo = io.get_geo(cache_dir, with_bmag=True)
        if geo is None or geo.empty:
            return None
        if geo.index.tz is None:
            geo.index = geo.index.tz_localize("UTC")
        return geo
    except Exception as e:
        print(f"[fsm:{TAG}] WARN geo 로드 실패({e}) → in_saa 태깅 생략")
        return None


def tag_onset_geo(onset_time, geo) -> dict:
    """onset 시점의 geo 조회 → in_saa/onset_Bmag/onset_maglat.
    검출은 전 구간에서 이미 끝났고, 여기선 '그 시점이 SAA였나'만 사후 태깅."""
    if geo is None:
        return {"in_saa": np.nan, "onset_Bmag": np.nan, "onset_maglat": np.nan}
    # onset_time 에 가장 가까운 geo 샘플 (1분 격자, 약간의 시차 허용)
    try:
        pos = geo.index.get_indexer([onset_time], method="nearest")[0]
    except Exception:
        return {"in_saa": np.nan, "onset_Bmag": np.nan, "onset_maglat": np.nan}
    if pos < 0:
        return {"in_saa": np.nan, "onset_Bmag": np.nan, "onset_maglat": np.nan}
    row = geo.iloc[pos]
    bmag = float(row["Bmag"]) if "Bmag" in geo.columns and pd.notna(row.get("Bmag")) else np.nan
    mlat = float(row["maglat"]) if "maglat" in geo.columns and pd.notna(row.get("maglat")) else np.nan
    in_saa = bool(bmag < SAA_BMAG_NT) if np.isfinite(bmag) else np.nan
    return {"in_saa": in_saa,
            "onset_Bmag": round(bmag, 1) if np.isfinite(bmag) else np.nan,
            "onset_maglat": round(mlat, 1) if np.isfinite(mlat) else np.nan}


def main():
    args = parse_args()
    io = _import_io(args.io, args.cache)
    window   = args.window
    k        = args.k
    onset_fl = args.onset
    peak_fl  = args.peak
    quiet_d  = args.quiet_days if args.quiet_days > 0 else None
    runtag   = build_runtag(TAG, window, k, onset_fl, peak_fl)

    cache = Path(args.cache)
    out_root = Path(args.out) if args.out else (cache.parent / "fsm_output")
    out_dir  = out_root / runtag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fsm:{TAG}] io={args.io} params: window={window}d k={k} onset={onset_fl} "
          f"peak={peak_fl} quiet_days={quiet_d}")
    print(f"[fsm:{TAG}] runtag: {runtag}")
    print(f"[fsm:{TAG}] Loading count data...")
    df_count = load_count(io, args.cache)
    if df_count.empty:
        print("[fsm] 빈 캐시"); return
    cnt_rs = df_count.resample(RESAMPLE_FREQ).mean()

    geo = None if args.no_geo else load_geo(io, args.cache)
    if geo is not None:
        print(f"[fsm:{TAG}] geo 태깅 ON (Bmag/maglat, SAA<|B|{SAA_BMAG_NT:.0f}nT)")

    onset_rows, event_rows = [], []
    for col in df_count.columns:
        species, direction, energy = col
        try:
            cnt = cnt_rs[col].dropna()
        except KeyError:
            continue
        if len(cnt) < MIN_PTS_PER_CHANNEL:
            continue
        chan = io.tuple_to_fname(tuple(col))   # 'pro_tel0_p5'
        print(f"\n[fsm:{TAG}] {chan}  (n={len(cnt)} pts)")
        bg   = compute_rolling_bg(cnt, window, quiet_d, BG_UPDATE_FREQ)
        thr  = build_threshold(bg, k, onset_fl)
        segs = detect_segments(cnt, thr, bg, MIN_SPE_DURATION_H)
        base = {"k": k, "onset_floor": onset_fl, "species": species,
                "direction": direction, "energy": energy, "channel": chan}
        for s in segs:
            geo_tag = tag_onset_geo(s["onset_time"], geo)
            onset_rows.append({**base, **s, **geo_tag})
        passed = [s for s in segs if s["peak_count"] >= peak_fl]
        for s in passed:
            geo_tag = tag_onset_geo(s["onset_time"], geo)
            event_rows.append({**base, "peak_floor": peak_fl, **s, **geo_tag})
        print(f"    segs={len(segs)}  peak>={peak_fl}: {len(passed)}")

    # 공통 컬럼(KSEM 과 동일) + POES 전용 geo 태깅 컬럼 末尾
    GEO_COLS = ["in_saa", "onset_Bmag", "onset_maglat"]
    ONSET_COLS = ["k","onset_floor","species","direction","energy","channel",
                  "onset_time","peak_time","end_time","onset_count","peak_count",
                  "end_count","duration_h","bg_median","bg_sigma","threshold"] + GEO_COLS
    EVENT_COLS = ["k","onset_floor","peak_floor","species","direction","energy","channel",
                  "onset_time","peak_time","end_time","onset_count","peak_count",
                  "end_count","duration_h","bg_median","bg_sigma","threshold"] + GEO_COLS
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
        # in_saa 비율도 함께 (SAA발 onset 진단)
        if "in_saa" in df_on.columns and df_on["in_saa"].notna().any():
            saa_frac = df_on.groupby("channel")["in_saa"].mean().rename("saa_frac").round(2)
            tbl = pd.concat([n_on, n_ev, saa_frac], axis=1).fillna(0)
        else:
            tbl = pd.concat([n_on, n_ev], axis=1).fillna(0)
        tbl = tbl.sort_values("n_onset", ascending=False)
        print(f"\n[fsm:{TAG}] per-channel counts (saa_frac=onset 중 SAA 비율):")
        print(tbl.to_string())


if __name__ == "__main__":
    main()
