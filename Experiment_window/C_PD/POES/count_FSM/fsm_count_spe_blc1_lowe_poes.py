"""
fsm_count_spe_blc1_lowe_poes.py
================================
BL-Löwe: Löwe et al. (2025) nowcasting 방식을 POES/SEM-2 count 에 이식.
본 연구 제안 방식(rolling MAD-robust FSM)의 직접 선행연구 대조군. POES 판.

GK2A 원본(fsm_count_spe_blc1_lowe.py) 의 POES 이식판:
  - 검출 엔진 5함수(_constant_fit, _window_samples, compute_rolling_bg,
    build_threshold, detect_segments)는 GK2A 원본과 **byte 단위 동일**.
  - 바뀐 것은 입출력 어댑터뿐:
      · io: ksem_io 하드코딩 → poes_{sat}_io (위성 인자화: --io)
      · 채널 루프: PD×side×logic → (species,direction,energy) MultiIndex 전체
      · geo 사후 태깅: onset 시점의 |B|/maglat 조회 → in_saa/onset_Bmag/onset_maglat
        (검출은 전 구간; 마스킹 아님. TPR 보존 + SAA발 false onset 식별)
  - 출력 CSV: POES 공통 컬럼(quietoff_mad_poes 와 동일) + geo 태깅 末尾 추가.
    noaa_goes_spe_match_poes.py 는 공통 컬럼만 읽으므로 호환 유지.

Löwe 원식 (MSL/RAD dose rate E 기준):
  - Background B: 직전 5일 constant linear fit (= 0차 polyfit = mean)
  - Threshold T:  T = 1.25 · B
  - bg_sigma 컬럼: NaN (Löwe엔 산포 항 없음)
  - k 컬럼:        0 (스키마 호환용 placeholder)

사용:
  python fsm_count_spe_blc1_lowe_poes.py \
      --io poes_metop03_io \
      --cache MetOp03_count/poes_metop03_cache_parquet
  python fsm_count_spe_blc1_lowe_poes.py \
      --io poes_noaa19_io \
      --cache NOAA19_count/poes_noaa19_cache_parquet \
      --window 5 --mult 1.25   # Löwe 원문 파라미터 그대로
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

# ── 이 파일의 방식 고정 (Löwe et al. 2025) ───────────────────────
BG_WINDOW_DAYS = 5         # [day] Löwe 원문은 직전 5일
LOWE_MULT      = 1.25      # T = 1.25 · B

ONSET_FLOOR = 0.5
PEAK_FLOOR  = 2.0
MIN_PTS_PER_CHANNEL = 100
SAA_BMAG_NT = 25000.0      # |B| < 이 값 → in_saa (coords_igrf 와 동일 기준)

TAG = "blc1_lowe"


def _import_io(io_arg: str, cache_dir: str):
    """io 모듈명/경로로 임포트. 캐시 경로 부모들을 sys.path 에 추가."""
    name = (PureWindowsPath(io_arg).name
            if (io_arg and ("/" in io_arg or "\\" in io_arg)) else io_arg)
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
        description=f"FSM count SPE detector POES [{TAG}] — "
                    f"Lowe et al. (2025) 5d constant-fit background, T=mult*B. "
                    f"engine byte-identical to GK2A blc1_lowe.")
    p.add_argument("--io",     required=True,
                   help="io 모듈명/경로 (poes_metop03_io | poes_noaa19_io)")
    p.add_argument("--cache",  required=True,
                   help="POES count parquet 캐시 디렉터리")
    p.add_argument("--out",    default=None,
                   help="출력 루트 (기본: 캐시 옆 fsm_output)")
    p.add_argument("--window", type=int,   default=BG_WINDOW_DAYS,
                   help=f"background sliding window [day] (default {BG_WINDOW_DAYS}, Lowe=5)")
    p.add_argument("--mult",   type=float, default=LOWE_MULT,
                   help=f"threshold multiplier T=mult*B (default {LOWE_MULT}, Lowe=1.25)")
    p.add_argument("--onset",  type=float, default=ONSET_FLOOR,
                   help=f"onset floor, threshold lower clip (default {ONSET_FLOOR})")
    p.add_argument("--peak",   type=float, default=PEAK_FLOOR,
                   help=f"peak floor, post-filter (default {PEAK_FLOOR})")
    p.add_argument("--no-geo", action="store_true",
                   help="geo 태깅 생략(in_saa 없이). geo 캐시 없을 때.")
    return p.parse_args()


def _numstr(v):
    f = float(v)
    return str(int(f)) if f.is_integer() else str(f)


def build_runtag(tag, window, mult, onset, peak):
    """출력 파일명용 파라미터 문자열. 예: blc1_lowe_w5_m1.25_on0.5_pk2.0"""
    return f"{tag}_w{window}_m{_numstr(mult)}_on{_numstr(onset)}_pk{_numstr(peak)}"


# ══════════════════════════════════════════════════════════════════
# 배경 추정 + 검출 엔진 (GK2A 원본과 byte-동일 — 절대 수정 금지)
# ══════════════════════════════════════════════════════════════════
def _constant_fit(x) -> float:
    """Löwe "constant linear fit (y = a)" 를 그대로 호출.

    np.polyfit(t, x, deg=0) 의 해는 a = mean(x) 로 떨어진다(0차 다항=상수).
    결과는 표본 평균과 동일하지만, 논문 문구(constant linear fit)를 코드가
    1:1로 따르도록 polyfit 으로 구현한다. 시간축은 균일 15분 격자이므로
    등간격 인덱스 t를 쓰며, 0차 피팅에서는 t 값 자체가 결과에 무관하다.
    """
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    if len(x) < 1:
        return np.nan
    t = np.arange(len(x), dtype=float)
    a = np.polyfit(t, x, 0)          # 0차 다항(상수) 피팅 → a == mean(x)
    return float(a[0])


def _window_samples(cnt: pd.Series, t_ref: pd.Timestamp,
                    bg_window_days: int) -> pd.Series:
    """t_ref 이전 bg_window_days 윈도우의 배경 표본 (전체 사용, quiet 선택 없음)."""
    seg = cnt.loc[t_ref - pd.Timedelta(days=bg_window_days):t_ref].dropna()
    if len(seg) < 10:
        return pd.Series(dtype=float)
    return seg


def compute_rolling_bg(cnt: pd.Series, bg_window_days: int,
                       update_freq: str = "1D") -> pd.DataFrame:
    """rolling 배경 시계열. update_freq마다 갱신 후 ffill.
    반환: DataFrame[bg_median, bg_std].
      bg_median 컬럼 = Löwe 배경 B (constant-fit mean). 스키마 호환 위해 이름 유지.
      bg_std    컬럼 = NaN (Löwe엔 산포 항이 없음)."""
    if cnt.empty:
        return pd.DataFrame(columns=["bg_median", "bg_std"])
    update_points = pd.date_range(cnt.index[0], cnt.index[-1], freq=update_freq)
    rows = []
    for t in update_points:
        q = _window_samples(cnt, t, bg_window_days)
        if len(q) >= 1:
            rows.append((t, _constant_fit(q), np.nan))   # B = constant fit, σ 없음
        else:
            rows.append((t, np.nan, np.nan))
    bg = pd.DataFrame(rows, columns=["t", "bg_median", "bg_std"]).set_index("t")
    return bg.reindex(cnt.index.union(bg.index)).ffill().reindex(cnt.index)


def build_threshold(bg: pd.DataFrame, mult: float, onset_floor: float) -> pd.Series:
    """Löwe 임계 = mult · B  (B = bg_median 컬럼의 constant-fit mean).
    onset_floor로 하한 클립 (저카운트 채널에서 B·mult가 바닥일 때 다른 FSM과 정합)."""
    th = mult * bg["bg_median"]
    if onset_floor > 0:
        th = th.clip(lower=onset_floor)
    return th


def detect_segments(cnt: pd.Series, thresh_series: pd.Series,
                    bg: pd.DataFrame, min_duration_h: float) -> list:
    """count >= rolling 임계가 min_duration_h 이상 연속인 구간. peak_floor는 호출부에서.
    (4종 FSM의 detect_segments와 동일 — bg_sigma만 NaN으로 기록됨)"""
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


# ══════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    io = _import_io(args.io, args.cache)
    window   = args.window
    mult     = args.mult
    onset_fl = args.onset
    peak_fl  = args.peak
    runtag   = build_runtag(TAG, window, mult, onset_fl, peak_fl)

    cache = Path(args.cache)
    out_root = Path(args.out) if args.out else (cache.parent / "fsm2_output")
    out_dir  = out_root / runtag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fsm:{TAG}] io={args.io} params: window={window}d mult={mult} "
          f"onset={onset_fl} peak={peak_fl}  (Lowe: 5d constant-fit mean, T={mult}*B)")
    print(f"[fsm:{TAG}] runtag: {runtag}")
    print(f"[fsm:{TAG}] out: {out_dir}")
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
        bg   = compute_rolling_bg(cnt, window, BG_UPDATE_FREQ)
        thr  = build_threshold(bg, mult, onset_fl)
        segs = detect_segments(cnt, thr, bg, MIN_SPE_DURATION_H)
        # k=0: 스키마 호환 placeholder (Löwe엔 k 개념 없음)
        base = {"k": 0, "onset_floor": onset_fl, "species": species,
                "direction": direction, "energy": energy, "channel": chan}
        for s in segs:
            geo_tag = tag_onset_geo(s["onset_time"], geo)
            onset_rows.append({**base, **s, **geo_tag})
        passed = [s for s in segs if s["peak_count"] >= peak_fl]
        for s in passed:
            geo_tag = tag_onset_geo(s["onset_time"], geo)
            event_rows.append({**base, "peak_floor": peak_fl, **s, **geo_tag})
        print(f"    segs={len(segs)}  peak>={peak_fl}: {len(passed)}")

    # 공통 컬럼(quietoff_mad_poes 와 동일) + POES 전용 geo 태깅 末尾
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
    print(f"\n[fsm:{TAG}] event saved: {out_ev}  ({len(df_ev)} rows)")
    print(f"[fsm:{TAG}] METHOD=Lowe constant-fit  WINDOW={window}d MULT={mult} "
          f"floor={onset_fl} peak={peak_fl}")
    if not df_on.empty:
        n_on = df_on.groupby("channel").size().rename("n_onset")
        n_ev = (df_ev.groupby("channel").size().rename("n_event")
                if not df_ev.empty else pd.Series(dtype=int, name="n_event"))
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
