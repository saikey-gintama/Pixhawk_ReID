"""
noaa_goes_spe_match.py
======================
검출 이벤트 CSV(예: fsm_onset_*.csv) ↔ NOAA SPE 카탈로그 매칭 + 시각화.

범용 평가기: 위성·채널 무관하게 "검출 이벤트 목록 CSV"를 받아 NOAA SPE
카탈로그(ground truth)와 대조한다. noaa_goes_spe_io 만 import하며,
KSEM 등 특정 데이터셋에 종속되지 않는다(CSV 경로를 인자로 받음).

매칭 정의:
  - hit: 검출 onset_time이 NOAA begin_time ±MATCH_TOL_H 이내에 있으면 일치.
  - POD = (매칭된 NOAA 이벤트 수) / (NOAA 이벤트 수)
  - FAR = (NOAA와 매칭 안 된 검출 수) / (검출 수)

사용 (GK2A/event_MATCHER/ 에서):
  # 단일 CSV — catalog/count-dir 기본값 자동, sweep 파라미터만 지정
  python noaa_goes_spe_match.py \
    --events ../count_FSM/fsm2_output/<tag>/fsm_onset_<tag>.csv

  # fsm*_output 폴더 전체 (onset CSV 자동)
  python noaa_goes_spe_match.py \
    --events-dir ../count_FSM/fsm2_output --kind onset
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

_HERE = Path(__file__).resolve().parent          # GK2A/event_MATCHER/
# noaa_goes_spe_io 는 ../../NOAA_GOES/ 에 위치
sys.path.insert(0, str(_HERE.parent.parent / "NOAA_GOES"))
import noaa_goes_spe_io as spe_io
from _match_core import (
    MATCH_TOL_H, KSEM_ERA, PFU_BINS, PFU_LABELS,
    _name_stem_from_events, match_events, sweep_table,
    _final_table, _discover, _channel_to_parquet, _load_count,
)


def plot_overlay(count_series: pd.Series, cat: pd.DataFrame,
                 det: pd.DataFrame, title: str, out_path: Path,
                 tol_h: float = MATCH_TOL_H):
    """KSEM count(좌축 선) + NOAA 이벤트 pfu 동그라미(우축 log).
    매칭된 NOAA=채운 원, 놓친 NOAA=빈 원. 검출 onset=세로 점선."""
    on = pd.to_datetime(det["onset_time"]).values
    fig, axL = plt.subplots(figsize=(16, 4.5))
    axL.plot(count_series.index, count_series.values,
             color="#2c3e50", lw=0.6, label="KSEM count", zorder=1)
    axL.set_ylabel("KSEM count [15-min]", fontsize=9)
    axL.set_ylim(bottom=0)

    axR = axL.twinx()
    axR.set_yscale("log")
    axR.set_ylabel("NOAA max_pfu", fontsize=9)

    n_hit = n_miss = 0
    for b, mp in zip(cat.index, cat["max_pfu"].values):
        if pd.isna(mp):
            continue
        dh  = (on - b.to_datetime64()) / np.timedelta64(1, "h")
        hit = np.any(np.abs(dh) <= tol_h)
        axR.scatter(b, mp, s=80,
                    facecolor=("#e74c3c" if hit else "none"),
                    edgecolor="#e74c3c", linewidth=1.5, zorder=3)
        n_hit += hit; n_miss += (not hit)

    for o in on:
        axL.axvline(pd.Timestamp(o), ls=":", color="orange",
                    lw=0.8, alpha=0.6, zorder=2)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="#2c3e50", lw=1.0, label="KSEM count"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#e74c3c",
               markeredgecolor="#e74c3c", markersize=9,
               label=f"NOAA detected ({n_hit})"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor="#e74c3c", markeredgewidth=1.5, markersize=9,
               label=f"NOAA missed ({n_miss})"),
        Line2D([0], [0], ls=":", color="orange", lw=1.0, label="detection onset"),
    ]
    axL.legend(handles=legend_handles, fontsize=8, loc="upper left", framealpha=0.9)
    axL.set_title(title, fontsize=10)
    axL.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axL.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[match] overlay saved → {out_path}")


def plot_pod_far_scatter(tbl, title, out_path):
    """채널별 POD vs FAR scatter (logic 그룹 색상). 우: onset_diff boxplot."""

    def _group(logic):
        if logic in ("OU", "OUT"):   return "A"
        if logic in ("FTU", "FTUO"): return "B"
        return "C"

    COLORS = {"A": "#2c6fbb", "B": "#e67e22", "C": "#95a5a6"}
    LABELS = {"A": "A: OU/OUT  (quiet bg, detection-suitable)",
              "B": "B: FTU/FTUO  (intermediate)",
              "C": "C: O/F/FT/CR  (active bg, unsuitable)"}

    df = tbl.copy()
    if "logic" not in df.columns:
        df["logic"] = df["channel"].str.split("-").str[-1]
    df["grp"] = df["logic"].map(_group)

    fig, (ax, axb) = plt.subplots(1, 2, figsize=(11, 5.2),
                                  gridspec_kw={"width_ratios": [3, 1.4]})
    for g in ["C", "B", "A"]:
        sub = df[df["grp"] == g]
        ax.scatter(sub["FAR"], sub["POD"], s=70, c=COLORS[g],
                   edgecolor="white", linewidth=0.6, alpha=0.85,
                   label=LABELS[g], zorder=3)

    out = df[df["channel"] == "PD1A-OU"]
    if len(out):
        ax.scatter(out["FAR"], out["POD"], s=180, marker="*",
                   facecolor="none", edgecolor="black", linewidth=1.3,
                   alpha=0.9, zorder=6, label="PD1A-OU (noise outlier)")

    lab_df = df[df["grp"].isin(["A", "B"])].sort_values("POD", ascending=False)
    try:
        from adjustText import adjust_text
        texts = [ax.text(r["FAR"], r["POD"], r["channel"], fontsize=6.5, zorder=7)
                 for _, r in lab_df.iterrows()]
        adjust_text(texts, ax=ax, expand=(1.3, 1.6),
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.4, alpha=0.6))
    except ImportError:
        for i, (_, r) in enumerate(lab_df.iterrows()):
            dy = 12 if (i % 2 == 0) else -14
            ax.annotate(r["channel"], (r["FAR"], r["POD"]),
                        fontsize=6.5, xytext=(6, dy), textcoords="offset points",
                        ha="left", va="center", zorder=7,
                        arrowprops=dict(arrowstyle="-", color="gray",
                                        lw=0.4, alpha=0.6, shrinkA=0, shrinkB=2))

    ax.set_xlabel("FAR (false alarm rate)", fontsize=10)
    ax.set_ylabel("POD (probability of detection)", fontsize=10)
    ax.set_xlim(-0.05, 1.08); ax.set_ylim(-0.05, 1.08)
    ax.axhline(0.5, ls=":", color="gray", lw=0.7, alpha=0.6)
    ax.axvline(0.5, ls=":", color="gray", lw=0.7, alpha=0.6)
    ax.text(0.02, 0.99, "ideal", fontsize=8, color="green", alpha=0.7, va="top")
    ax.legend(fontsize=7, loc="best", framealpha=0.92)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.25)

    diff_col = "onset_diff_med_h"
    if diff_col in df.columns:
        data, labels, colors = [], [], []
        for g in ["A", "B", "C"]:
            vals = df[df["grp"] == g][diff_col].dropna().values
            if len(vals):
                data.append(vals); labels.append(g); colors.append(COLORS[g])
        if data:
            bp = axb.boxplot(data, tick_labels=labels, patch_artist=True,
                             widths=0.6, showfliers=False)
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c); patch.set_alpha(0.6)
            axb.axhline(0, ls="--", color="black", lw=0.8, alpha=0.6)
            axb.set_ylabel("onset_diff [h]\n(neg = KSEM leads)", fontsize=8)
            axb.set_xlabel("channel group", fontsize=9)
            axb.set_title("onset_diff", fontsize=10)
            axb.grid(True, alpha=0.25, axis="y")
    else:
        axb.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[match] scatter saved -> {out_path}")


def run_one(events_csv, args, cat):
    """단일 검출 CSV → 매칭 표 + scatter + overlay 일괄."""
    name = _name_stem_from_events(events_csv)
    _ev_cols = pd.read_csv(events_csv, nrows=0).columns
    no_peak = "peak_floor" not in _ev_cols
    if no_peak:
        name = f"{name}_onset"
    outdir = Path(args.out) / name
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== [match] {Path(events_csv).name} -> {outdir} ===")

    tbl = sweep_table(Path(events_csv), cat, args.tol)
    tbl.to_csv(outdir / f"noaa_match_summary_all_{name}.csv", index=False)
    print(f"[match] summary saved -> noaa_match_summary_all_{name}.csv ({len(tbl)} rows)")

    fin = _final_table(tbl, no_peak, args)
    fin.to_csv(outdir / f"noaa_match_{name}.csv", index=False)
    print(f"[match] FINAL saved -> noaa_match_{name}.csv")
    print(fin.to_string(index=False))

    plot_pod_far_scatter(fin, f"NOAA match  {name}",
                         outdir / f"fig_noaa_scatter_{name}.png")

    ev = pd.read_csv(events_csv)
    _pcols = ["k", "onset_floor"] + (["peak_floor"] if "peak_floor" in ev.columns else [])
    ev_single = ev[_pcols].drop_duplicates().shape[0] <= 1

    def _draw(channel, parquet_path):
        if ev_single:
            det = ev[ev.channel == channel]
        else:
            _m = (ev.channel == channel) & (ev.k == args.k) & (ev.onset_floor == args.onset)
            if "peak_floor" in ev.columns:
                _m = _m & (ev.peak_floor == args.peak)
            det = ev[_m]
        cnt = _load_count(parquet_path)
        plot_overlay(cnt, cat, det, f"{channel}  {name}",
                     outdir / f"overlay_{channel}_{name}.png", args.tol)

    if args.count_dir:
        cdir = Path(args.count_dir)
        for channel in sorted(ev["channel"].unique()):
            pq = cdir / _channel_to_parquet(channel)
            if pq.exists():
                _draw(channel, pq)
            else:
                print(f"[match] skip {channel}: parquet 없음 ({pq.name})")
    elif args.count and args.channel:
        _draw(args.channel, Path(args.count))


def main():
    ap = argparse.ArgumentParser(description="검출 이벤트 ↔ NOAA SPE 매칭/시각화")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--events", help="검출 CSV 한 개 (단일 실행)")
    src.add_argument("--events-dir",
                     help="fsm*_output 폴더 — 그 아래 fsm_{kind}_*.csv 전부 실행")
    ap.add_argument("--kind", choices=["onset", "event"], default="onset",
                    help="--events-dir 모드에서 고를 CSV 종류 (default onset)")
    ap.add_argument("--catalog",
                    default=str(_HERE.parent.parent / "NOAA_GOES" / "noaa_goes_spe_cache_parquet"),
                    help="NOAA 캐시 경로(parquet 디렉터리 or json)")
    ap.add_argument("--tol", type=float, default=MATCH_TOL_H, help="매칭 허용오차(h)")
    ap.add_argument("--out", default="noaa_match_output", help="출력 루트 폴더")
    ap.add_argument("--count-dir",
                    default=str(_HERE.parent / "KSEM_count" / "ksem_cache_parquet"),
                    help="전 채널 overlay: count parquet 폴더")
    ap.add_argument("--count",   default=None, help="단일 채널 count parquet")
    ap.add_argument("--channel", default=None, help="단일 채널명 (예: PD3A-OU)")
    ap.add_argument("--k",     type=int,   default=None,
                    help="[sweep 전용] FINAL k 지정 (단일조합 파일이면 불필요)")
    ap.add_argument("--onset", type=float, default=None,
                    help="[sweep 전용] FINAL onset_floor 지정")
    ap.add_argument("--peak",  type=float, default=None,
                    help="[sweep 전용] FINAL peak_floor 지정 (event CSV 전용)")
    args = ap.parse_args()

    cat_all, _ = spe_io.load(args.catalog)
    cat = spe_io.filter_by_date(cat_all, *KSEM_ERA)

    if args.events_dir:
        files = _discover(Path(args.events_dir), args.kind)
        if not files:
            raise SystemExit(
                f"[match] {args.events_dir} 아래 fsm_{args.kind}_*.csv 없음")
        print(f"[match] {len(files)}개 {args.kind} CSV 발견 -> 전부 매칭")
        for f in files:
            run_one(f, args, cat)
        print(f"\n[match] 완료: {len(files)}개 -> {args.out}/")
    else:
        run_one(Path(args.events), args, cat)


if __name__ == "__main__":
    main()
