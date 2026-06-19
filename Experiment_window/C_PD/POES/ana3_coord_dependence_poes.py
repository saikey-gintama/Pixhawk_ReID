"""
ana3_coord_dependence_poes.py  (축 3 — 좌표/메타 의존성 + 스파이크 원인 규명)
=============================================================================
채널 count 가 *어디서* 변하는지를 geo(lat/lon/alt→Bmag/maglat)로 규명한다.
특히 ana1 에서 드러난 거대 스파이크(max ≫ p99)가 SAA(|B|<25000) 때문인지,
고위도 때문인지, 단발 글리치인지 태깅한다.

산출:
  spike_origin.csv     채널별 상위 극단값(max 포함 top-K)의 geo 태그
                       (in_saa / high_lat / other) 집계 + max 한 점 상세
  maglat_profile.csv   채널 × maglat 빈 median count
  coord_summary.csv    채널별: in_saa median / quiet median / saa_over_quiet 등
  maglat_<chan>.png    채널별 maglat-binned median 프로파일
  heatmap_<chan>.png   채널별 lat-lon 2D 평균 count 히트맵
  (--no-plots 로 그림 생략)

조건 정의:
  SAA      : |B| < SAA_BMAG_NT(25000 nT)      ← coords_igrf
  high_lat : |maglat| > --highlat-deg (기본 60)
  quiet    : SAA 아님 & high_lat 아님 (배경 비교 기준)

사용:
  python ana3_coord_dependence_poes.py --io poes_metop03_io \
      --cache <parquet_dir> --out <out_dir> [--topk 50] [--no-plots]
"""
from __future__ import annotations
import argparse, importlib, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _import_io(io_arg: str, cache_dir: str):
    name = Path(io_arg).stem if ('/' in io_arg or '\\' in io_arg) else io_arg
    cache = Path(cache_dir).resolve()
    for cand in (cache, cache.parent, cache.parent.parent, Path.cwd()):
        sp = str(cand)
        if sp not in sys.path:
            sys.path.insert(0, sp)
    return importlib.import_module(name)


def _align(s: pd.Series, geo: pd.DataFrame) -> pd.DataFrame:
    """채널 Series 와 geo 를 같은 시간축으로 정렬 (이벤트 마스킹과 동일 패턴)."""
    g = geo.reindex(s.index)
    d = pd.DataFrame({"count": s.values}, index=s.index)
    for c in ("Bmag", "maglat", "lat", "lon"):
        if c in g.columns:
            d[c] = g[c].values
    return d


def spike_origin(df, geo, io, topk: int):
    """각 채널 상위 topk 극단값의 geo 태그 집계 + max 한 점 상세."""
    rows = []
    for col in df.columns:
        name = io.tuple_to_fname(tuple(col))
        s = df[col].dropna()
        if s.empty:
            continue
        d = _align(s, geo).dropna(subset=["count"])
        top = d.nlargest(topk, "count")
        in_saa = (top["Bmag"] < 25000).sum() if "Bmag" in top else np.nan
        high_lat = (top["maglat"].abs() > 60).sum() if "maglat" in top else np.nan
        other = topk - (in_saa if np.isfinite(in_saa) else 0) \
                     - (high_lat if np.isfinite(high_lat) else 0)
        mx = d.loc[d["count"].idxmax()]
        rows.append({
            "channel": name,
            "max": round(float(mx["count"]), 2),
            "max_time": d["count"].idxmax(),
            "max_Bmag": round(float(mx.get("Bmag", np.nan)), 1),
            "max_maglat": round(float(mx.get("maglat", np.nan)), 1),
            "max_lat": round(float(mx.get("lat", np.nan)), 1),
            "max_lon": round(float(mx.get("lon", np.nan)), 1),
            "max_in_saa": bool(mx.get("Bmag", 9e9) < 25000),
            f"top{topk}_in_saa": int(in_saa) if np.isfinite(in_saa) else None,
            f"top{topk}_high_lat": int(high_lat) if np.isfinite(high_lat) else None,
            f"top{topk}_other": int(other),
        })
    return pd.DataFrame(rows)


def maglat_profile(df, geo, io, nbins: int = 18):
    """채널 × maglat 빈 median count."""
    edges = np.linspace(-90, 90, nbins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    out = {}
    for col in df.columns:
        name = io.tuple_to_fname(tuple(col))
        s = df[col].dropna()
        d = _align(s, geo).dropna(subset=["count", "maglat"])
        if d.empty:
            continue
        binned = pd.cut(d["maglat"], edges)
        med = d.groupby(binned, observed=False)["count"].median()
        out[name] = med.values
    prof = pd.DataFrame(out, index=np.round(centers, 1))
    prof.index.name = "maglat_bin"
    return prof


def coord_summary(df, geo, io):
    """채널별 SAA vs quiet median 대비."""
    rows = []
    for col in df.columns:
        name = io.tuple_to_fname(tuple(col))
        s = df[col].dropna()
        d = _align(s, geo).dropna(subset=["count"])
        if "Bmag" not in d or d["Bmag"].isna().all():
            continue
        saa = d[d["Bmag"] < 25000]["count"]
        hl = d[d["maglat"].abs() > 60]["count"] if "maglat" in d else pd.Series(dtype=float)
        quiet = d[(d["Bmag"] >= 25000) & (d["maglat"].abs() <= 60)]["count"]
        q_med = quiet.median() if len(quiet) else np.nan
        rows.append({
            "channel": name,
            "quiet_med": round(float(q_med), 4) if np.isfinite(q_med) else np.nan,
            "saa_med": round(float(saa.median()), 4) if len(saa) else np.nan,
            "highlat_med": round(float(hl.median()), 4) if len(hl) else np.nan,
            "saa_over_quiet": round(float(saa.median() / q_med), 2)
                if (len(saa) and np.isfinite(q_med) and q_med > 0) else np.nan,
            "highlat_over_quiet": round(float(hl.median() / q_med), 2)
                if (len(hl) and np.isfinite(q_med) and q_med > 0) else np.nan,
            "n_saa": int(len(saa)), "n_quiet": int(len(quiet)),
        })
    return pd.DataFrame(rows)


def plot_maglat(prof, name, out):
    if name not in prof.columns:
        return
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(prof.index, prof[name], "o-", color="#8e44ad")
    ax.set_xlabel("magnetic latitude (deg)"); ax.set_ylabel("median count rate")
    ax.set_title(f"{name} — maglat profile", fontsize=9)
    ax.axvspan(-90, -60, alpha=0.08, color="red")
    ax.axvspan(60, 90, alpha=0.08, color="red")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out / f"maglat_{name}.png", dpi=110); plt.close(fig)


def plot_heatmap(s, geo, name, out, nlat=36, nlon=72):
    d = _align(s.dropna(), geo).dropna(subset=["count", "lat", "lon"])
    if d.empty:
        return
    lat_edges = np.linspace(-90, 90, nlat + 1)
    lon_edges = np.linspace(-180, 180, nlon + 1)
    grid = np.full((nlat, nlon), np.nan)
    li = np.clip(np.digitize(d["lat"], lat_edges) - 1, 0, nlat - 1)
    oi = np.clip(np.digitize(d["lon"], lon_edges) - 1, 0, nlon - 1)
    tmp = pd.DataFrame({"li": li, "oi": oi, "c": d["count"].values})
    g = tmp.groupby(["li", "oi"])["c"].mean()
    for (a, b), v in g.items():
        grid[a, b] = v
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.pcolormesh(lon_edges, lat_edges, grid, shading="auto",
                       norm=matplotlib.colors.LogNorm(), cmap="viridis")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title(f"{name} — mean count by lat/lon", fontsize=9)
    fig.colorbar(im, ax=ax, label="mean count rate")
    fig.tight_layout(); fig.savefig(out / f"heatmap_{name}.png", dpi=110); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="축3 좌표/SAA 의존성 + 스파이크 규명 (POES)")
    p.add_argument("--io", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--topk", type=int, default=50, help="스파이크 태깅 상위 K (기본 50)")
    p.add_argument("--highlat-deg", type=float, default=60.0)
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()

    io = _import_io(args.io, args.cache)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    df, meta = io.load(args.cache)
    geo = io.get_geo(args.cache, with_bmag=True)
    if df.empty or geo.empty:
        print("[ERROR] 캐시 또는 geo 비어있음"); return
    print(f"[ana3] {df.shape[1]} 채널, geo {geo.shape[0]:,} rows "
          f"(Bmag/maglat: {'있음' if 'Bmag' in geo else '없음'})")

    so = spike_origin(df, geo, io, args.topk)
    so.to_csv(out / "spike_origin.csv", index=False)
    print(f"[ana3] spike_origin.csv ({len(so)} 채널)")
    # 스파이크 원인 요약 출력
    for _, r in so.iterrows():
        tag = "SAA" if r["max_in_saa"] else "non-SAA"
        print(f"  {r['channel']:16s} max={r['max']:>12.1f} @ "
              f"|B|={r['max_Bmag']:>7.0f} maglat={r['max_maglat']:>6.1f} → {tag}")

    prof = maglat_profile(df, geo, io)
    prof.to_csv(out / "maglat_profile.csv")
    cs = coord_summary(df, geo, io)
    cs.to_csv(out / "coord_summary.csv", index=False)
    print(f"[ana3] maglat_profile.csv / coord_summary.csv 저장")

    if not args.no_plots:
        for col in df.columns:
            name = io.tuple_to_fname(tuple(col))
            plot_maglat(prof, name, out)
            plot_heatmap(df[col], geo, name, out)
        print(f"[ana3] maglat_*/heatmap_* 그림 → {out}")


if __name__ == "__main__":
    main()
