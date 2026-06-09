"""
noaa_goes_spe_match.py
======================
검출 이벤트 CSV(예: ana6_sweep_event.csv) ↔ NOAA SPE 카탈로그 매칭 + 시각화.

범용 평가기: 위성·채널 무관하게 "검출 이벤트 목록 CSV"를 받아 NOAA SPE
카탈로그(ground truth)와 대조한다. noaa_goes_spe_io 만 import하며,
KSEM 등 특정 데이터셋에 종속되지 않는다(CSV 경로를 인자로 받음).

매칭 정의:
  - hit: 검출 onset_time이 NOAA begin_time ±MATCH_TOL_H 이내에 있으면 일치.
    (NOAA onset/end 시각이 불확실하고 KSEM onset이 floor 지연되므로 넉넉히 24h)
  - POD = (매칭된 NOAA 이벤트 수) / (NOAA 이벤트 수)
  - FAR = (NOAA와 매칭 안 된 검출 수) / (검출 수)
  - pfu 구간별 POD: 약한/중간/강한 이벤트의 검출력 분리.

시간차(매칭 쌍에 대해서만):
  - onset_diff_h = KSEM onset_time - NOAA begin_time
  - peak_diff_h  = KSEM peak_time  - NOAA max_time

겹쳐그리기(GOES flux raw 불필요):
  - KSEM count 시계열(좌 y축, 선) 위에 NOAA 이벤트를 begin 시점 pfu 크기
    동그라미(우 y축, log)로. 매칭=채운 원, 놓침=빈 원 → POD가 그림에 보임.

사용:
  # 단일 채널 겹쳐그리기
  python noaa_goes_spe_match.py \
    --events ../KSEM_plot/ana_output/ana6_output/ana6_sweep_event.csv \
    --catalog .  \
    [--count PD3_A_OU.parquet --channel PD3A-OU --k 10 --onset 0.5 --peak 3.0]

  # 전 채널 일괄 (count parquet들이 든 폴더 지정; 채널명→PD_S_LOGIC.parquet 매핑)
  python noaa_goes_spe_match.py \
    --events ana6_sweep_event.csv --catalog . \
    --count-dir ../KSEM_count/ksem_cache_parquet --k 10 --onset 0.5 --peak 3.0
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import noaa_goes_spe_io as spe_io



def _name_stem_from_events(events_path):
    """입력 CSV 파일명에서 fsm_event_/fsm_onset_ 접두만 떼고 나머지 전체를 반환.
    예) fsm_event_quietoff_mad_w30_k10_on0.5_pk2.csv
        -> 'quietoff_mad_w30_k10_on0.5_pk2'
    접두가 없으면 확장자만 뗀 stem 그대로."""
    import re
    stem = Path(events_path).stem
    m = re.match(r"fsm_(?:event|onset)_(.+)", stem)
    return m.group(1) if m else stem


# ── 매칭 파라미터 ─────────────────────────────────────────────────
MATCH_TOL_H   = 24.0                       # NOAA begin ±시간 매칭 허용오차
KSEM_ERA      = ("2019-01-01", "2024-12-31")
PFU_BINS      = [10, 100, 1000, np.inf]    # pfu 구간 경계 (>=10 SPE 정의)
PFU_LABELS    = ["10-100", "100-1k", "1k+"]


# ─────────────────────────────────────────────────────────────────
def match_events(det: pd.DataFrame, noaa: pd.DataFrame,
                 tol_h: float = MATCH_TOL_H) -> dict:
    """
    검출 이벤트(det, onset_time 보유)와 NOAA(begin_time index, max_time/max_pfu)를
    매칭. 반환: POD/FAR, pfu 구간별 POD, 매칭 쌍의 시간차 배열.
    """
    on = pd.to_datetime(det["onset_time"]).values
    pk = pd.to_datetime(det["peak_time"]).values
    nb = noaa.index.values                       # NOAA begin
    nm = pd.to_datetime(noaa["max_time"]).values  # NOAA max
    npfu = noaa["max_pfu"].values

    # 각 NOAA가 검출됐는가 (hit)
    noaa_hit = np.zeros(len(noaa), dtype=bool)
    onset_diff, peak_diff, matched_pfu = [], [], []
    for i, b in enumerate(nb):
        dh = (on - b) / np.timedelta64(1, "h")
        j = np.where(np.abs(dh) <= tol_h)[0]
        if len(j):
            noaa_hit[i] = True
            # 가장 가까운 검출과의 시간차
            jc = j[np.argmin(np.abs(dh[j]))]
            onset_diff.append((on[jc] - b) / np.timedelta64(1, "h"))
            if pd.notna(nm[i]):
                peak_diff.append((pk[jc] - nm[i]) / np.timedelta64(1, "h"))
            matched_pfu.append(npfu[i])

    # 각 검출이 NOAA와 매칭됐는가 (false alarm 판정)
    det_matched = np.zeros(len(det), dtype=bool)
    for i, o in enumerate(on):
        dh = (nb - o) / np.timedelta64(1, "h")
        if np.any(np.abs(dh) <= tol_h):
            det_matched[i] = True

    n_noaa, n_det = len(noaa), len(det)
    pod = noaa_hit.sum() / n_noaa if n_noaa else np.nan
    far = (~det_matched).sum() / n_det if n_det else np.nan

    # pfu 구간별 POD
    pfu_pod = {}
    binned = pd.cut(npfu, PFU_BINS, labels=PFU_LABELS, right=False)
    for lab in PFU_LABELS:
        m = (binned == lab)
        pfu_pod[lab] = (noaa_hit[m].sum(), int(m.sum()))

    return {
        "n_noaa": n_noaa, "n_det": n_det,
        "pod": pod, "far": far,
        "n_hit": int(noaa_hit.sum()), "n_fa": int((~det_matched).sum()),
        "pfu_pod": pfu_pod,
        "onset_diff_h": np.array(onset_diff),
        "peak_diff_h": np.array(peak_diff),
        "matched_pfu": np.array(matched_pfu),
    }


def sweep_table(events_csv: Path, catalog_dir: Path,
                tol_h: float = MATCH_TOL_H) -> pd.DataFrame:
    """
    sweep_event.csv의 모든 (channel × k × onset × peak) 조합에 대해
    POD/FAR/시간차중앙값을 표로 산출.
    """
    ev = pd.read_csv(events_csv)
    noaa_all, _ = spe_io.load(str(catalog_dir))
    noaa = spe_io.filter_by_date(noaa_all, *KSEM_ERA)

    rows = []
    keys = ["channel", "k", "onset_floor", "peak_floor"]
    for (ch, k, onf, pkf), grp in ev.groupby(keys):
        r = match_events(grp, noaa, tol_h)
        rows.append({
            "channel": ch, "k": k, "onset_floor": onf, "peak_floor": pkf,
            "n_det": r["n_det"], "n_hit": r["n_hit"], "n_fa": r["n_fa"],
            "POD": round(r["pod"], 3), "FAR": round(r["far"], 3),
            "onset_diff_med_h": round(float(np.median(r["onset_diff_h"])), 2)
                if len(r["onset_diff_h"]) else np.nan,
            "peak_diff_med_h": round(float(np.median(r["peak_diff_h"])), 2)
                if len(r["peak_diff_h"]) else np.nan,
            **{f"POD_{lab}": f"{h}/{n}" for lab, (h, n) in r["pfu_pod"].items()},
        })
    return pd.DataFrame(rows)


def plot_overlay(count_series: pd.Series, noaa: pd.DataFrame,
                 det: pd.DataFrame, title: str, out_path: Path,
                 tol_h: float = MATCH_TOL_H):
    """
    KSEM count(좌축 선) + NOAA 이벤트 pfu 동그라미(우축 log).
    매칭된 NOAA=채운 원, 놓친 NOAA=빈 원. 검출 onset=세로 점선.
    """
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
    for b, mp in zip(noaa.index, noaa["max_pfu"].values):
        if pd.isna(mp):
            continue
        dh = (on - b.to_datetime64()) / np.timedelta64(1, "h")
        hit = np.any(np.abs(dh) <= tol_h)
        axR.scatter(b, mp, s=80,
                    facecolor=("#e74c3c" if hit else "none"),
                    edgecolor="#e74c3c", linewidth=1.5, zorder=3)
        n_hit += hit
        n_miss += (not hit)

    for o in on:
        axL.axvline(pd.Timestamp(o), ls=":", color="orange",
                    lw=0.8, alpha=0.6, zorder=2)

    # 실제 범례 (proxy artist로 마커 종류 표시)
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


# ─────────────────────────────────────────────────────────────────
def _channel_to_parquet(channel: str) -> str:
    """채널명 'PD3A-OU' → parquet 파일명 'PD3_A_OU.parquet'."""
    # 'PD3A-OU' → pd='PD3', side='A', logic='OU'
    pd_side, logic = channel.split("-")
    pd_key, side = pd_side[:-1], pd_side[-1]
    return f"{pd_key}_{side}_{logic}.parquet"


def _load_count(parquet_path: Path) -> pd.Series:
    cnt = pd.read_parquet(parquet_path)
    if isinstance(cnt, pd.DataFrame):
        cnt = cnt.iloc[:, 0]
    cnt.index = pd.to_datetime(cnt.index, utc=True)
    return cnt.resample("15min").mean().dropna()




def plot_pod_far_scatter(tbl, title, out_path):
    """채널별 POD vs FAR scatter. x=FAR y=POD.
    그룹: A=OU/OUT(파랑, 탐지적합), B=FTU/FTUO(주황, 중간),
          C=O/F/FT/CR(회색, 부적합). std/MAD 기반 3그룹.
    PD1A-OU 는 noise outlier 로 반투명 별 표시.
    옆 패널에 onset_diff 박스플롯(그룹별)."""

    def _group(logic):
        if logic in ("OU", "OUT"):            return "A"
        if logic in ("FTU", "FTUO"):          return "B"
        return "C"                            # O, F, FT, CR
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

    # ── scatter (회색 먼저, 파랑 위로) ──
    for g in ["C", "B", "A"]:
        sub = df[df["grp"] == g]
        ax.scatter(sub["FAR"], sub["POD"], s=70, c=COLORS[g],
                   edgecolor="white", linewidth=0.6, alpha=0.85,
                   label=LABELS[g], zorder=3)

    # ── PD1A-OU outlier: 반투명·축소 별 (뒤 파랑 점 보이게) ──
    out = df[df["channel"] == "PD1A-OU"]
    if len(out):
        ax.scatter(out["FAR"], out["POD"], s=180, marker="*",
                   facecolor="none", edgecolor="black", linewidth=1.3,
                   alpha=0.9, zorder=6, label="PD1A-OU (noise outlier)")

    # ── 채널명 라벨: A·B 그룹만 ──
    # adjustText 가 있으면 자동 분산(겹침 회피), 없으면 위아래 번갈아 폴백.
    lab_df = df[df["grp"].isin(["A", "B"])].sort_values("POD", ascending=False)
    try:
        from adjustText import adjust_text
        texts = [ax.text(r["FAR"], r["POD"], r["channel"], fontsize=6.5, zorder=7)
                 for _, r in lab_df.iterrows()]
        adjust_text(
            texts, ax=ax,
            expand=(1.3, 1.6),
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.4, alpha=0.6))
    except ImportError:
        for i, (_, r) in enumerate(lab_df.iterrows()):
            dy = 12 if (i % 2 == 0) else -14
            ax.annotate(
                r["channel"], (r["FAR"], r["POD"]),
                fontsize=6.5, xytext=(6, dy), textcoords="offset points",
                ha="left", va="center", zorder=7,
                arrowprops=dict(arrowstyle="-", color="gray",
                                lw=0.4, alpha=0.6, shrinkA=0, shrinkB=2))

    ax.set_xlabel("FAR (false alarm rate)", fontsize=10)
    ax.set_ylabel("POD (probability of detection)", fontsize=10)
    ax.set_xlim(-0.05, 1.08); ax.set_ylim(-0.05, 1.08)
    ax.axhline(0.5, ls=":", color="gray", lw=0.7, alpha=0.6)
    ax.axvline(0.5, ls=":", color="gray", lw=0.7, alpha=0.6)
    ax.text(0.02, 0.99, "ideal", fontsize=8, color="green",
            alpha=0.7, va="top")
    ax.legend(fontsize=7, loc="best", framealpha=0.92)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.25)

    # ── 우: onset_diff 박스플롯 (그룹별) ──
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


def main():
    ap = argparse.ArgumentParser(description="검출 이벤트 ↔ NOAA SPE 매칭/시각화")
    ap.add_argument("--events", required=True, help="검출 이벤트 CSV (ana6_sweep_event.csv)")
    ap.add_argument("--catalog", default=".", help="NOAA 캐시 경로(parquet 디렉터리 or json)")
    ap.add_argument("--tol", type=float, default=MATCH_TOL_H, help="매칭 허용오차(h)")
    ap.add_argument("--out", default="noaa_match_output", help="출력 폴더")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--onset", type=float, default=0.5)
    ap.add_argument("--peak", type=float, default=3.0)
    # 겹쳐그리기: 단일 채널(--count + --channel) 또는 전 채널 일괄(--count-dir)
    ap.add_argument("--count", default=None, help="단일 채널 count parquet")
    ap.add_argument("--channel", default=None, help="단일 채널명 (예: PD3A-OU)")
    ap.add_argument("--count-dir", default=None,
                    help="전 채널 일괄: count parquet들이 든 폴더 (채널명→PD_S_LOGIC.parquet 매핑)")
    args = ap.parse_args()

    # 출력 폴더/파일명을 입력 CSV 파일명 그대로 따라감 (덮어쓰기 방지)
    name = _name_stem_from_events(args.events)   # 예: quietoff_mad_w30_k10_on0.5_pk2
    outdir = Path(args.out) / name
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) 전 조합 POD/FAR 표 (27조합 전체 — 조합 무관하게 항상 동일)
    tbl = sweep_table(Path(args.events), Path(args.catalog), args.tol)
    tbl.to_csv(outdir / f"noaa_match_summary_all_{name}.csv", index=False)
    print(f"[match] summary (all combos) saved → "
          f"{outdir}/noaa_match_summary_all_{name}.csv  ({len(tbl)} rows)")

    # FINAL 조합만 추린 채널별 표 (별도 저장 + 콘솔)
    fin = tbl[(tbl.k == args.k) & (tbl.onset_floor == args.onset)
              & (tbl.peak_floor == args.peak)].sort_values("POD", ascending=False)
    fin.to_csv(outdir / f"noaa_match_{name}.csv", index=False)
    print(f"[match] FINAL ({name}) saved -> "
          f"{outdir}/noaa_match_{name}.csv")
    print(fin.to_string(index=False))

    # scatter figure (Figure for paper)
    plot_pod_far_scatter(
        fin, f"NOAA match  {name}",
        outdir / f"fig_noaa_scatter_{name}.png")

    # 2) 겹쳐그리기
    ev = pd.read_csv(args.events)
    noaa_all, _ = spe_io.load(args.catalog)
    noaa = spe_io.filter_by_date(noaa_all, *KSEM_ERA)

    def _draw(channel: str, parquet_path: Path):
        det = ev[(ev.channel == channel) & (ev.k == args.k)
                 & (ev.onset_floor == args.onset) & (ev.peak_floor == args.peak)]
        cnt = _load_count(parquet_path)
        plot_overlay(cnt, noaa, det,
                     f"{channel}  {name}",
                     outdir / f"overlay_{channel}_{name}.png", args.tol)

    if args.count_dir:
        # 전 채널 일괄: CSV에 등장하는 모든 채널에 대해 parquet 찾아 그림
        cdir = Path(args.count_dir)
        for channel in sorted(ev["channel"].unique()):
            pq = cdir / _channel_to_parquet(channel)
            if pq.exists():
                _draw(channel, pq)
            else:
                print(f"[match] skip {channel}: parquet 없음 ({pq.name})")
    elif args.count and args.channel:
        _draw(args.channel, Path(args.count))


if __name__ == "__main__":
    main()