"""
fsm_count_spe_blc1_fixed_poes.py
=================================
BL-C1: 고정임계 SPE onset 검출기 POES 판. GK2A blc1_fixed 의 POES 이식.

GK2A 원본(fsm_count_spe_blc1_fixed.py) 의 POES 이식판:
  - 검출 엔진 3함수(compute_const_bg, build_threshold, detect_segments)는
    GK2A 원본과 **byte 단위 동일**.
  - 바뀐 것은 입출력 어댑터뿐:
      · io: ksem_io 하드코딩 → poes_{sat}_io (위성 인자화: --io)
      · 채널 루프: PD×side×logic → (species,direction,energy) MultiIndex
      · C 결정: CSV per-channel(1순위) → skip+경고(fallback 없음)
      · geo 사후 태깅: onset 시점의 |B|/maglat 조회 → in_saa/onset_Bmag/onset_maglat
  - POES channels에는 KSEM logic명(OU/OUT 등)이 없으므로 CONST_C_BY_LOGIC fallback
    은 POES 버전에서 제거함. CSV가 없거나 채널이 없으면 skip + 경고.
  - 출력 CSV: POES 공통 컬럼(quietoff_mad_poes 와 동일) + geo 태깅 末尾.

C 결정 방식 (--mode):
  const : --const-csv stats CSV의 channel→C_fpr0.05. 없으면 채널 skip+경고.
          POES proton C_fpr0.05 최솟값(pro_tel0_p5=1.46) > onset_floor(0.5)
          → floor 클립이 proton 채널을 건드리지 않음.
  pctl  : C = percentile(채널 전 기간 count, PCTL). 기본 PCTL=95.

사용:
  python fsm_count_spe_blc1_fixed_poes.py \
      --io poes_metop03_io \
      --cache MetOp03_count/poes_metop03_cache_parquet \
      --mode pctl --pctl 95

  python fsm_count_spe_blc1_fixed_poes.py \
      --io poes_metop03_io \
      --cache MetOp03_count/poes_metop03_cache_parquet \
      --mode const \
      --const-csv MetOp03_count/ana_event_output/noaa_spe_event_count_stats.csv
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
MIN_SPE_DURATION_H = 1

DEFAULT_MODE = "pctl"
DEFAULT_PCTL = 95.0

ONSET_FLOOR = 0.5
PEAK_FLOOR  = 2.0
MIN_PTS_PER_CHANNEL = 100
SAA_BMAG_NT = 25000.0

TAG = "blc1_fixed"


# ══════════════════════════════════════════════════════════════════
# io 임포트 헬퍼
# ══════════════════════════════════════════════════════════════════
def _import_io(io_arg: str, cache_dir: str):
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


# ══════════════════════════════════════════════════════════════════
# CSV 로드
# ══════════════════════════════════════════════════════════════════
def load_const_csv(path: str | None) -> dict[str, float]:
    """stats CSV(channel, C_fpr0.05) → {channel: float} 매핑."""
    if not path:
        return {}
    try:
        df = pd.read_csv(path, usecols=lambda c: c in ("channel", "C_fpr0.05"))
        result = {}
        for _, row in df.iterrows():
            try:
                c = float(row["C_fpr0.05"])
                if np.isfinite(c):
                    result[str(row["channel"])] = c
            except (ValueError, KeyError):
                pass
        print(f"[{TAG}] const-csv 로드: {Path(path).name}  ({len(result)}채널)")
        return result
    except Exception as e:
        print(f"[{TAG}] WARN const-csv 로드 실패({e}) → skip 전용 모드")
        return {}


def parse_args():
    p = argparse.ArgumentParser(
        description=f"FSM count SPE detector POES [{TAG}] — fixed-threshold C "
                    f"(rolling 제거 ablation). engine byte-identical to GK2A blc1_fixed.")
    p.add_argument("--io",        required=True,
                   help="io 모듈명/경로 (poes_metop03_io | poes_noaa19_io)")
    p.add_argument("--cache",     required=True,
                   help="POES count parquet 캐시 디렉터리")
    p.add_argument("--out",       default=None,
                   help="출력 루트 (기본: 캐시 옆 fsm2_output)")
    p.add_argument("--mode",      choices=["const", "pctl"], default=DEFAULT_MODE)
    p.add_argument("--pctl",      type=float, default=DEFAULT_PCTL)
    p.add_argument("--const-csv", default=None,
                   help="const 모드 채널별 C_fpr0.05 CSV 경로")
    p.add_argument("--onset",     type=float, default=ONSET_FLOOR)
    p.add_argument("--peak",      type=float, default=PEAK_FLOOR)
    p.add_argument("--no-geo",    action="store_true",
                   help="geo 태깅 생략(in_saa 없이). geo 캐시 없을 때.")
    return p.parse_args()


def _numstr(v):
    f = float(v)
    return str(int(f)) if f.is_integer() else str(f)


def build_runtag(tag, mode, pctl, onset, peak):
    cstr = f"pctl{_numstr(pctl)}" if mode == "pctl" else "const"
    return f"{tag}_{cstr}_on{_numstr(onset)}_pk{_numstr(peak)}"


# ══════════════════════════════════════════════════════════════════
# C 결정 (POES: CSV 1순위, fallback 없음)
# ══════════════════════════════════════════════════════════════════
def resolve_C(channel: str, cnt: pd.Series,
              mode: str, pctl: float,
              const_csv_map: dict | None = None) -> tuple[float, str]:
    """반환: (C, source).  source='csv'|'no_csv'|'pctl'."""
    if mode == "const":
        if const_csv_map and channel in const_csv_map:
            return float(const_csv_map[channel]), "csv"
        return np.nan, "no_csv"
    return float(np.nanpercentile(cnt.values, pctl)), "pctl"


# ══════════════════════════════════════════════════════════════════
# 배경 추정 + 검출 엔진 (GK2A 원본과 byte-동일 — 절대 수정 금지)
# ══════════════════════════════════════════════════════════════════
def compute_const_bg(cnt: pd.Series, C: float) -> pd.DataFrame:
    """고정임계용 '배경' 시계열. bg_median=C 상수, bg_std=NaN."""
    if cnt.empty:
        return pd.DataFrame(columns=["bg_median", "bg_std"])
    return pd.DataFrame(
        {"bg_median": np.full(len(cnt), float(C)),
         "bg_std":    np.full(len(cnt), np.nan)},
        index=cnt.index,
    )


def build_threshold(bg: pd.DataFrame, onset_floor: float) -> pd.Series:
    """임계 = C(=bg_median)를 onset_floor로 하한 클립."""
    th = bg["bg_median"].copy()
    if onset_floor > 0:
        th = th.clip(lower=onset_floor)
    return th


def detect_segments(cnt: pd.Series, thresh_series: pd.Series,
                    bg: pd.DataFrame, min_duration_h: float) -> list:
    """count >= 임계가 min_duration_h 이상 연속인 구간. peak_floor는 호출부에서."""
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
    df_count, _ = io.load(cache_dir)
    if not df_count.empty and df_count.index.tz is None:
        df_count.index = df_count.index.tz_localize("UTC")
    return df_count


def load_geo(io, cache_dir: str):
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
    mode     = args.mode
    pctl     = args.pctl
    onset_fl = args.onset
    peak_fl  = args.peak
    runtag   = build_runtag(TAG, mode, pctl, onset_fl, peak_fl)

    cache = Path(args.cache)
    out_root = Path(args.out) if args.out else (cache.parent / "fsm2_output")
    out_dir  = out_root / runtag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fsm:{TAG}] io={args.io} mode={mode} pctl={pctl} "
          f"onset={onset_fl} peak={peak_fl}")
    print(f"[fsm:{TAG}] runtag: {runtag}")
    print(f"[fsm:{TAG}] out: {out_dir}")

    # ── const-csv 로드 및 검증 출력 ──────────────────────────────
    const_csv_map = load_const_csv(args.const_csv) if mode == "const" else {}

    if mode == "const" and not const_csv_map:
        print(f"[{TAG}] WARN: --const-csv 미지정 또는 로드 실패 "
              f"→ const 모드에서 전 채널 skip됨")

    if mode == "const" and const_csv_map:
        VERIFY_CHANNELS = ["pro_tel0_p5", "pro_tel90_p5", "ele_tel0_e1"]
        print(f"\n[{TAG}] const-csv 로드 검증 (샘플 채널):")
        for ch in VERIFY_CHANNELS:
            if ch in const_csv_map:
                c_val = const_csv_map[ch]
                floor_clipped = c_val < onset_fl
                note = f"  ← floor_clip!" if floor_clipped else ""
                print(f"  {ch:16s} → C_fpr0.05 = {c_val:.4f}{note}")
            else:
                print(f"  {ch:16s} → (CSV 없음, skip)")

    print(f"\n[fsm:{TAG}] Loading count data...")
    df_count = load_count(io, args.cache)
    if df_count.empty:
        print("[fsm] 빈 캐시"); return
    cnt_rs = df_count.resample(RESAMPLE_FREQ).mean()

    geo = None if args.no_geo else load_geo(io, args.cache)
    if geo is not None:
        print(f"[fsm:{TAG}] geo 태깅 ON (Bmag/maglat, SAA<|B|{SAA_BMAG_NT:.0f}nT)")

    onset_rows, event_rows = [], []
    no_csv_channels: list[str] = []

    for col in df_count.columns:
        species, direction, energy = col
        try:
            cnt = cnt_rs[col].dropna()
        except KeyError:
            continue
        if len(cnt) < MIN_PTS_PER_CHANNEL:
            continue
        chan = io.tuple_to_fname(tuple(col))   # 'pro_tel0_p5'
        C, c_src = resolve_C(chan, cnt, mode, pctl, const_csv_map)

        if c_src == "no_csv":
            no_csv_channels.append(chan)
            print(f"[{TAG}] WARN {chan}: CSV에 C_fpr0.05 없음 → skip")
            continue

        if not np.isfinite(C):
            print(f"\n[fsm:{TAG}] {chan}  C 미정의 — skip")
            continue

        th_eff = max(C, onset_fl) if onset_fl > 0 else C
        print(f"\n[fsm:{TAG}] {chan}  (n={len(cnt)} pts)  "
              f"C={C:.4f} [{c_src}] → th={th_eff:.4f}")
        bg   = compute_const_bg(cnt, C)
        thr  = build_threshold(bg, onset_fl)
        segs = detect_segments(cnt, thr, bg, MIN_SPE_DURATION_H)
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

    # ── skip 요약 (no_csv) ────────────────────────────────────────
    if mode == "const":
        if no_csv_channels:
            print(f"\n[{TAG}] ⚠ no_csv skip: {len(no_csv_channels)}채널")
            for ch in no_csv_channels:
                print(f"    {ch}")
        else:
            print(f"\n[{TAG}] no_csv skip: 0채널 (전 채널 CSV 로드 성공)")

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
    print(f"[fsm:{TAG}] METHOD=fixed-C  mode={mode} pctl={pctl} "
          f"floor={onset_fl} peak={peak_fl}")
    if not df_on.empty:
        n_on = df_on.groupby("channel").size().rename("n_onset")
        n_ev = (df_ev.groupby("channel").size().rename("n_event")
                if not df_ev.empty else pd.Series(dtype=int, name="n_event"))
        if "in_saa" in df_on.columns and df_on["in_saa"].notna().any():
            saa_frac = (df_on.groupby("channel")["in_saa"]
                           .mean().rename("saa_frac").round(2))
            tbl = pd.concat([n_on, n_ev, saa_frac], axis=1).fillna(0)
        else:
            tbl = pd.concat([n_on, n_ev], axis=1).fillna(0)
        tbl = tbl.sort_values("n_onset", ascending=False)
        print(f"\n[fsm:{TAG}] per-channel counts:")
        print(tbl.to_string())


if __name__ == "__main__":
    main()
