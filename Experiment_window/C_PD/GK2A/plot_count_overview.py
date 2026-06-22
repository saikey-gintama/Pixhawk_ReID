"""
plot_count_overview.py
======================
KSEM count raw 시계열을 Loewe et al. (2025) dose-rate 그림처럼 그린다.
 - raw count (가독성용 리샘플 max 옵션)
 - 왼쪽 y축 = count rate (로그), 오른쪽 y축 = event peak flux pfu (로그)
 - NOAA(SPE) = 빨강 화살표, SWPC(ESPE) = 파랑 화살표
 - 채널별 1장씩 (PD x side x logic)
 - 통합 패널: --combine 으로 logic 지정 → 해당 logic 을 AB/PD123 전부 합산해 1장
              여러 logic 은 콤마로 구분 → 각각 별도 1장: --combine "OU,OUT"

io 인터페이스는 ana2_condition_profile_ksem.py 와 동일:
  df, meta  = ksem_io.load(cache)          # df.columns=(pd_key,side,logic), DatetimeIndex
  ev_df, _  = <event_io>.load(dir)         # begin_time index + max_pfu col

사용:
  python plot_count_overview.py \
      --cache KSEM_count/ksem_cache_parquet \
      --out   KSEM_count/plot_output \
      --spe   ../NOAA_GOES/noaa_goes_spe_cache_parquet \
      --espe  ../SWPC_Alert/swpc_espe_cache_parquet \
      --combine "OU,OUT"
"""
from __future__ import annotations
import argparse, importlib, sys
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _import_mod(name_or_path: str, *extra_dirs) -> object:
    """모듈 이름 또는 경로를 받아 importlib 으로 로드."""
    name = (PureWindowsPath(name_or_path).name
            if ("/" in name_or_path or "\\" in name_or_path)
            else name_or_path)
    for d in extra_dirs:
        sp = str(Path(d).resolve())
        if sp not in sys.path:
            sys.path.insert(0, sp)
    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))
    return importlib.import_module(name)


def _event_points(ev_df: pd.DataFrame) -> list[tuple]:
    """(UTC Timestamp, pfu float) 리스트 반환."""
    if ev_df is None or ev_df.empty:
        return []
    out = []
    for begin, row in ev_df.iterrows():
        if pd.isna(begin):
            continue
        b = pd.Timestamp(begin)
        b = b.tz_localize("UTC") if b.tz is None else b.tz_convert("UTC")
        try:
            pfu = float(row.get("max_pfu", np.nan))
        except (TypeError, ValueError):
            pfu = np.nan
        out.append((b, pfu))
    return out


def _load_events(io_name: str, ev_dir: str, *near) -> list[tuple]:
    if not ev_dir:
        return []
    mod = _import_mod(io_name, Path(ev_dir).parent, Path(ev_dir).parent.parent, *near)
    ev_df, _ = mod.load(ev_dir)
    return _event_points(ev_df)


def _resample(s: pd.Series, rule: str | None, agg: str) -> pd.Series:
    if not rule:
        return s
    return getattr(s.resample(rule), agg)()


def _draw(ax_left, series_list, labels, colors, noaa_pts, swpc_pts, ymin, title):
    """이중 y축 그림: 왼=count(로그), 오른=pfu(로그)+화살표."""
    drawn_max = ymin
    xmin = xmax = None

    for s, lab, c in zip(series_list, labels, colors):
        s = s.dropna()
        if s.empty:
            continue
        idx = pd.DatetimeIndex(s.index)
        idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        y = np.where(s.to_numpy(dtype=float) > 0, s.to_numpy(dtype=float), np.nan)
        ax_left.plot(idx, y, lw=0.5, color=c, label=lab, zorder=2)
        m = np.nanmax(y) if np.any(np.isfinite(y)) else ymin
        drawn_max = max(drawn_max, m)
        xmin = idx.min() if xmin is None else min(xmin, idx.min())
        xmax = idx.max() if xmax is None else max(xmax, idx.max())

    ax_left.set_yscale("log")
    ax_left.set_ylabel("count rate (#/s)")
    ax_left.set_xlabel("Time [year]")
    ax_left.set_title(title, fontsize=10)
    ax_left.grid(True, which="major", alpha=0.3)
    ax_left.grid(True, which="minor", alpha=0.10)
    ax_left.set_ylim(0, drawn_max * 8.0)
    if xmin is not None:
        ax_left.set_xlim(xmin, xmax)

    ax_r = ax_left.twinx()
    ax_r.set_yscale("log")
    ax_r.set_ylabel("event peak flux (pfu)")
    all_pfu = [p for _, p in (noaa_pts + swpc_pts) if np.isfinite(p) and p > 0]
    ax_r.set_ylim(0, max(all_pfu) * 5.0 if all_pfu else 1e4)

    def _arrows(points, color):
        for t, pfu in points:
            if xmin is not None and (t < xmin or t > xmax):
                continue
            if not (np.isfinite(pfu) and pfu > 0):
                continue
            ax_r.annotate(
                "", xy=(t, pfu), xytext=(t, pfu * 3.5),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1, fill=False),
                zorder=5)

    _arrows(noaa_pts, "crimson")
    _arrows(swpc_pts, "tab:blue")

    chan_handles = [plt.Line2D([], [], color=c, lw=1.2, label=l)
                   for l, c in zip(labels, colors)]
    ev_handles = []
    if noaa_pts:
        ev_handles.append(plt.Line2D([], [], color="crimson",
                                     marker=r"$\downarrow$", ls="", markersize=10,
                                     label="NOAA SPE"))
    if swpc_pts:
        ev_handles.append(plt.Line2D([], [], color="tab:blue",
                                     marker=r"$\downarrow$", ls="", markersize=10,
                                     label="SWPC ESPE"))
    leg1 = ax_left.legend(handles=chan_handles, loc="upper left",
                          fontsize=8, framealpha=0.9)
    ax_left.add_artist(leg1)
    if ev_handles:
        ax_r.legend(handles=ev_handles, loc="upper right", fontsize=8, framealpha=0.9)


_PALETTE = {
    "OU": "tab:blue", "OUT": "tab:orange", "O": "tab:green",
    "F": "tab:red",   "FT": "tab:purple",  "FTU": "tab:brown",
    "FTUO": "tab:pink", "CR": "gray",
}


def main():
    p = argparse.ArgumentParser(
        description="KSEM raw count overview, dual-axis, NOAA/SWPC arrows")
    p.add_argument("--io",       default="ksem_io",
                   help="ksem_io 모듈 이름 또는 경로 (기본: ksem_io)")
    p.add_argument("--cache",    required=True,
                   help="ksem_cache_parquet 디렉터리")
    p.add_argument("--out",      required=True,
                   help="그림 저장 디렉터리")
    p.add_argument("--spe",      help="NOAA SPE parquet 디렉터리 (빨강 화살표)")
    p.add_argument("--espe",     help="SWPC ESPE parquet 디렉터리 (파랑 화살표)")
    p.add_argument("--spe-io",   default="noaa_goes_spe_io")
    p.add_argument("--espe-io",  default="swpc_alert_espe_io")
    p.add_argument("--resample", default="1D",
                   help="리샘플 rule (1D, 6H, …). 빈값=raw")
    p.add_argument("--agg",      default="max",
                   choices=["max", "mean", "median"])
    p.add_argument("--ymin",     type=float, default=1e-2)
    p.add_argument("--combine",  default="",
                   help="통합 패널: 콤마 구분 logic → 각각 1장. 예 'OU,OUT'")
    p.add_argument("--show-events", default="both",
                   choices=["noaa", "swpc", "both", "none"],
                   help="통합 패널에만 적용되는 이벤트 필터")
    args = p.parse_args()

    cache_path = Path(args.cache)
    io = _import_mod(args.io,
                     cache_path if cache_path.is_dir() else cache_path.parent,
                     cache_path.parent, cache_path.parent.parent,
                     Path.cwd(), Path(__file__).resolve().parent)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df, _meta = io.load(args.cache)
    if df.empty:
        print("[ERROR] 빈 캐시")
        return
    print(f"[plot] {df.shape[1]} 채널, {len(df):,} rows")

    noaa_pts = _load_events(args.spe_io,  args.spe)  if args.spe  else []
    swpc_pts = _load_events(args.espe_io, args.espe) if args.espe else []
    if noaa_pts:
        print(f"[plot] NOAA SPE {len(noaa_pts)}개 (빨강)")
    if swpc_pts:
        print(f"[plot] SWPC ESPE {len(swpc_pts)}개 (파랑)")

    rule = args.resample.strip() or None

    # (1) 채널별 1장씩 — 이벤트 항상 both
    for col in df.columns:
        name = "_".join(col)
        s = _resample(df[col].dropna(), rule, args.agg)
        if s.empty:
            continue
        fig, ax = plt.subplots(figsize=(11, 3.3))
        _draw(ax, [s], [name], ["black"], noaa_pts, swpc_pts,
              args.ymin, f"{name} — raw count overview")
        fig.tight_layout()
        fig.savefig(out / f"count_overview_{name}.png", dpi=130)
        plt.close(fig)
    print(f"[plot] 채널별 {df.shape[1]}장 → {out}")

    # (2) 통합 패널: logic 별 합산 1장씩
    if not args.combine.strip():
        print("[plot] --combine 비어있음 → 통합 패널 생략")
        return

    comb_noaa = noaa_pts if args.show_events in ("noaa", "both") else []
    comb_swpc = swpc_pts if args.show_events in ("swpc", "both") else []

    seen = []
    for tok in args.combine.split(","):
        lg = tok.strip()
        if not lg or lg in seen:
            continue
        seen.append(lg)
        cols = [c for c in df.columns if c[2] == lg]
        if not cols:
            print(f"[plot] combine '{lg}': 채널 없음 — 건너뜀")
            continue
        summed = _resample(df[cols].sum(axis=1, min_count=1).dropna(), rule, args.agg)
        if summed.empty:
            print(f"[plot] combine '{lg}': 합산 결과 비어있음 — 건너뜀")
            continue
        fig, ax = plt.subplots(figsize=(11, 3.6))
        _draw(ax, [summed], [f"{lg} (sum of {len(cols)} ch)"],
              [_PALETTE.get(lg, "black")], comb_noaa, comb_swpc,
              args.ymin, f"{lg} — summed over all PD/side")
        fig.tight_layout()
        fig.savefig(out / f"count_overview_SUM_{lg}.png", dpi=130)
        plt.close(fig)
        print(f"[plot] 합산 '{lg}' → count_overview_SUM_{lg}.png ({len(cols)} 채널)")

    print("done.")


if __name__ == "__main__":
    main()
