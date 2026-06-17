"""
ana2_condition_profile_poes.py  (축 2 — 조건별 count 프로파일)
=============================================================
채널 count 를 조건(condition)별로 분류해 분포를 비교한다. *매칭이 아니라*
구간 통계다. 조건:
  quiet   : SAA 아님 & 고위도 아님            (배경 기준)
  saa     : |B| < SAA_BMAG_NT(25000)          (coords_igrf)
  highlat : |maglat| > --highlat-deg (기본 60)
  spe     : NOAA SPE 활성구간 (begin_time ~ max_time, 상승구간만)
  espe    : SWPC ESPE 활성구간 (begin_time ~ max_time)

핵심 지표:
  spe_over_quiet   : SPE median / quiet median   (이벤트가 quiet 대비 몇 배 — 클수록 트리거 신호 良)
  saa_over_spe     : SAA median / SPE median     (>1 이면 SAA가 SEP보다 더 들림 → 트리거 부적합)
  espe_over_quiet  : ESPE median / quiet median

산출:
  condition_profile.csv   채널 × 조건 median/p90/p99/n + 위 지표
  cond_box_<chan>.png     채널별 조건간 분포 박스플롯(log-y)
  (--no-plots 로 그림 생략)

geo 조건(quiet/saa/highlat)과 event 조건(spe/espe)은 동일한 reindex+bool
마스크 패턴으로 처리 — 축3 coord_summary 를 이벤트까지 확장한 형태.

사용:
  python ana2_condition_profile_poes.py --io poes_metop03_io \
      --cache <parquet_dir> --out <out_dir> \
      --spe <noaa_spe_parquet_dir> --espe <swpc_espe_parquet_dir> [--no-plots]
"""
from __future__ import annotations
import argparse, importlib, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _import_module(mod_arg: str, *extra_dirs):
    """모듈명/경로에서 모듈 임포트. 후보 폴더들을 sys.path 에 추가."""
    name = Path(mod_arg).stem if ('/' in mod_arg or '\\' in mod_arg) else mod_arg
    for cand in extra_dirs:
        sp = str(Path(cand).resolve())
        if sp not in sys.path:
            sys.path.insert(0, sp)
    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))
    return importlib.import_module(name)


def _event_mask(index: pd.DatetimeIndex, ev_df: pd.DataFrame) -> pd.Series:
    """이벤트 df(begin_time 인덱스 + max_time 컬럼)의 begin~max 상승구간 합집합 마스크."""
    mask = pd.Series(False, index=index)
    if ev_df is None or ev_df.empty:
        return mask
    for begin, row in ev_df.iterrows():
        mx = row.get("max_time")
        if pd.isna(begin) or pd.isna(mx):
            continue
        b = pd.Timestamp(begin); e = pd.Timestamp(mx)
        if b.tz is None: b = b.tz_localize("UTC")
        if e.tz is None: e = e.tz_localize("UTC")
        if e < b:
            continue
        mask |= (index >= b) & (index <= e)
    return mask


def _stats(s: pd.Series) -> dict:
    v = s.dropna().to_numpy()
    if len(v) == 0:
        return {"med": np.nan, "p90": np.nan, "p99": np.nan, "n": 0}
    return {"med": float(np.median(v)),
            "p90": float(np.percentile(v, 90)),
            "p99": float(np.percentile(v, 99)),
            "n": int(len(v))}


def build_profile(df, geo, spe_mask, espe_mask, io, highlat_deg):
    rows = []
    has_b = "Bmag" in geo.columns
    has_ml = "maglat" in geo.columns
    # geo 조건 마스크 (전 채널 공통, 시간축 기준)
    for col in df.columns:
        name = io.tuple_to_fname(tuple(col))
        s = df[col].dropna()
        if s.empty:
            continue
        g = geo.reindex(s.index)
        saa = (g["Bmag"] < 25000) if has_b else pd.Series(False, index=s.index)
        hl = (g["maglat"].abs() > highlat_deg) if has_ml else pd.Series(False, index=s.index)
        saa = saa.fillna(False); hl = hl.fillna(False)
        quiet = (~saa) & (~hl)
        spe = spe_mask.reindex(s.index).fillna(False)
        espe = espe_mask.reindex(s.index).fillna(False)

        st = {c: _stats(s[m]) for c, m in
              [("quiet", quiet), ("saa", saa), ("highlat", hl),
               ("spe", spe), ("espe", espe)]}
        q = st["quiet"]["med"]
        row = {"channel": name, "species": col[0], "direction": col[1], "energy": col[2]}
        for c in ("quiet", "saa", "highlat", "spe", "espe"):
            row[f"{c}_med"] = round(st[c]["med"], 4) if np.isfinite(st[c]["med"]) else np.nan
            row[f"{c}_p99"] = round(st[c]["p99"], 4) if np.isfinite(st[c]["p99"]) else np.nan
            row[f"{c}_n"] = st[c]["n"]
        def ratio(a):
            return round(a / q, 2) if (np.isfinite(a) and np.isfinite(q) and q > 0) else np.nan
        row["spe_over_quiet"] = ratio(st["spe"]["med"])
        row["espe_over_quiet"] = ratio(st["espe"]["med"])
        row["saa_over_quiet"] = ratio(st["saa"]["med"])
        sp = st["spe"]["med"]
        row["saa_over_spe"] = round(st["saa"]["med"] / sp, 2) \
            if (np.isfinite(sp) and sp > 0) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def plot_box(df, geo, spe_mask, espe_mask, io, name_col, out, highlat_deg):
    has_b = "Bmag" in geo.columns; has_ml = "maglat" in geo.columns
    for col in df.columns:
        name = io.tuple_to_fname(tuple(col))
        s = df[col].dropna()
        if s.empty:
            continue
        g = geo.reindex(s.index)
        saa = (g["Bmag"] < 25000).fillna(False) if has_b else pd.Series(False, index=s.index)
        hl = (g["maglat"].abs() > highlat_deg).fillna(False) if has_ml else pd.Series(False, index=s.index)
        quiet = (~saa) & (~hl)
        spe = spe_mask.reindex(s.index).fillna(False)
        espe = espe_mask.reindex(s.index).fillna(False)
        data, labels = [], []
        for c, m in [("quiet", quiet), ("saa", saa), ("highlat", hl),
                     ("spe", spe), ("espe", espe)]:
            vals = s[m].to_numpy()
            vals = vals[vals > 0]
            if len(vals) > 0:
                data.append(vals); labels.append(f"{c}\n(n={len(vals)})")
        if len(data) < 2:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        bp = ax.boxplot(data, showfliers=False)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_yscale("log")
        ax.set_ylabel("count rate (#/s, >0)")
        ax.set_title(f"{name} — count by condition", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout(); fig.savefig(out / f"cond_box_{name}.png", dpi=110); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="축2 조건별 count 프로파일 (POES)")
    p.add_argument("--io", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--spe", help="NOAA SPE parquet 디렉터리")
    p.add_argument("--espe", help="SWPC ESPE parquet 디렉터리")
    p.add_argument("--spe-io", default="noaa_goes_spe_io")
    p.add_argument("--espe-io", default="swpc_alert_espe_io")
    p.add_argument("--highlat-deg", type=float, default=60.0)
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()

    cache = Path(args.cache)
    io = _import_module(args.io, cache, cache.parent, cache.parent.parent)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    df, meta = io.load(args.cache)
    geo = io.get_geo(args.cache, with_bmag=True)
    if df.empty:
        print("[ERROR] 빈 캐시"); return
    print(f"[ana2] {df.shape[1]} 채널, {len(df):,} rows, "
          f"geo Bmag:{'O' if 'Bmag' in geo else 'X'} maglat:{'O' if 'maglat' in geo else 'X'}")

    # 이벤트 카탈로그 로드 → 상승구간 마스크
    spe_mask = pd.Series(False, index=df.index)
    espe_mask = pd.Series(False, index=df.index)
    if args.spe:
        spe_io = _import_module(args.spe_io, Path(args.spe).parent, Path(args.spe).parent.parent)
        spe_df, _ = spe_io.load(args.spe)
        spe_mask = _event_mask(df.index, spe_df)
        print(f"[ana2] SPE 이벤트 {len(spe_df)}개 → 활성 {int(spe_mask.sum()):,} min")
    if args.espe:
        espe_io = _import_module(args.espe_io, Path(args.espe).parent, Path(args.espe).parent.parent)
        espe_df, _ = espe_io.load(args.espe)
        espe_mask = _event_mask(df.index, espe_df)
        print(f"[ana2] ESPE 이벤트 {len(espe_df)}개 → 활성 {int(espe_mask.sum()):,} min")

    prof = build_profile(df, geo, spe_mask, espe_mask, io, args.highlat_deg)
    prof.to_csv(out / "condition_profile.csv", index=False)
    print(f"[ana2] condition_profile.csv ({len(prof)} 채널)")
    # 요약 출력 (트리거 후보 관점)
    show = ["channel", "quiet_med", "saa_med", "spe_med", "espe_med",
            "spe_over_quiet", "saa_over_spe"]
    show = [c for c in show if c in prof.columns]
    print(prof[show].to_string(index=False))

    if not args.no_plots:
        plot_box(df, geo, spe_mask, espe_mask, io, "channel", out, args.highlat_deg)
        print(f"[ana2] cond_box_* 그림 → {out}")


if __name__ == "__main__":
    main()