"""
spe_vs_ksem.py  —  NOAA SPE GT 이벤트 vs KSEM Raw Data 응답 진단기
=====================================================================
"GT 이벤트 시점에 KSEM value/background 비율이 실제로 몇 배였나?"

states CSV(threshold 의존)가 아닌 Raw Count 원본 데이터를 직접 읽어
15분 리샘플 후 GT 이벤트 구간별 응답을 분석한다.

분석 항목:
  - 이벤트 구간 내 max(value), max(value/bg_mean)
  - 감지에 필요한 최소 threshold multiplier
  - 임계값별 감지 가능 이벤트 수 시뮬레이션
  - Flare lead time vs KSEM 응답 시간
  - 이벤트 구간 전/중/후 평균 count rate 비교
  - PD별 응답 차이
  - 월별/연도별 패턴
  - 놓친 이벤트 vs 감지된 이벤트 특성 비교

출력:
  spe_vs_ksem_summary.csv    — 이벤트별 상세 테이블
  spe_vs_ksem_overview.png   — 종합 시각화 (6패널)
  spe_vs_ksem_detail.png     — 상세 분석 시각화 (6패널)
  spe_vs_ksem_timeseries/    — 이벤트별 시계열 PNG (옵션)

사용법:
  python spe_vs_ksem.py \\
      --root "D:\\Raw Count" \\
      --events ".\\spe_out_1924\\known_events_1924.json" \\
      --output_dir ".\\spe_vs_ksem_out"

  # 이벤트별 시계열도 저장
  python spe_vs_ksem.py \\
      --root "D:\\Raw Count" \\
      --events ".\\spe_out_1924\\known_events_1924.json" \\
      --output_dir ".\\spe_vs_ksem_out" \\
      --plot_timeseries
"""

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

# ── ksem_bl_c2.py와 동일한 설정 ──────────────────────────────
RESAMPLE_MIN = 15
BACKGROUND_WINDOW_DAYS = 5

PROTON_BINS = (
    #list(range(1, 24)) +   # O (148 keV ~ )
    #list(range(25, 42)) +  # OU  (19.5 MeV ~ )
    list(range(46, 57))    # OUT (19.5 MeV ~ 22.5 MeV)
)
EXCLUDE_BINS = {0, 24, 45, 57, 81, 105, 117, 127}

# 비교할 threshold multiplier 목록
THRESHOLD_MULTS = [1.25, 2, 3, 5, 7, 10, 15, 20, 21, 25, 30, 50]


# ============================================================
# 1.  Raw 로더 (ksem_bl_c2.py 로직 재사용)
# ============================================================

def _parse_bin_idx(col: str, prefix: str) -> Optional[int]:
    try:
        return int(col[len(prefix):].split("(")[0])
    except Exception:
        return None


def load_one_csv(filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(filepath)
    raw["Time"] = pd.to_datetime(raw["Time"])
    raw = raw.set_index("Time").sort_index()

    def extract(prefix: str) -> pd.DataFrame:
        m = {c: _parse_bin_idx(c, prefix)
             for c in raw.columns if c.startswith(prefix)}
        m = {c: i for c, i in m.items() if i is not None}
        df = raw[list(m)].copy()
        df.columns = [m[c] for c in df.columns]
        return df[sorted(df.columns)]

    return extract("A"), extract("B")


def proton_total_from_csv(filepath: Path) -> pd.Series:
    df_A, df_B = load_one_csv(str(filepath))
    bins_A = [b for b in PROTON_BINS if b in df_A.columns and b not in EXCLUDE_BINS]
    bins_B = [b for b in PROTON_BINS if b in df_B.columns and b not in EXCLUDE_BINS]
    return (df_A[bins_A].sum(axis=1) + df_B[bins_B].sum(axis=1)).rename("proton_total")


def ym_folders(root: Path, start_ym: str, end_ym: str) -> List[Path]:
    folders = sorted([f for f in root.iterdir()
                      if f.is_dir() and f.name.isdigit() and len(f.name) == 6])
    return [f for f in folders if start_ym <= f.name <= end_ym]


def load_raw_range(root: Path, start_ym: str, end_ym: str,
                   pd_key: str) -> pd.Series:
    """Raw 1분 데이터 로드 → 15분 리샘플."""
    folders = ym_folders(root, start_ym, end_ym)
    parts = []
    for folder in folders:
        files = sorted([f for f in folder.glob("*.csv")
                        if pd_key.lower() in f.name.lower()])
        for fp in files:
            try:
                parts.append(proton_total_from_csv(fp))
            except Exception as e:
                print(f"  [WARN] {fp.name}: {e}")
    if not parts:
        raise FileNotFoundError(f"{pd_key} 데이터 없음: {root}")
    full = pd.concat(parts).sort_index()
    full = full[~full.index.duplicated(keep="first")]
    # 15분 리샘플
    return full.resample(f"{RESAMPLE_MIN}min").mean().dropna()


# ============================================================
# 2.  Background 추정 (rolling 5일 mean, ksem_bl_c2 동일)
# ============================================================

def compute_rolling_bg(series: pd.Series) -> pd.Series:
    """각 포인트에서 직전 5일치 mean → background Series."""
    window = BACKGROUND_WINDOW_DAYS * 24 * 60 // RESAMPLE_MIN  # 480 pts
    bg = series.rolling(window=window, min_periods=window).mean()
    return bg


# ============================================================
# 3.  이벤트별 분석
# ============================================================

def analyze_event(ev: dict,
                  pd_data: Dict[str, pd.Series],
                  pd_bg:   Dict[str, pd.Series]) -> dict:
    """
    단일 GT 이벤트에 대한 KSEM 응답 분석.

    반환 dict 키:
      onset, peak, end, peak_pfu
      pre_window_*    : onset 전 24h 평균 count
      event_*         : onset~end 구간 통계
      post_window_*   : end 후 24h 평균 count
      max_ratio_*     : PD별 max(value/bg)
      max_ratio_any   : 3 PD 중 최대
      ksem_peak_t_*   : KSEM count 피크 시각 (PD별)
      ksem_rise_min_* : onset → KSEM 피크 시간 (분)
      required_mult   : 감지에 필요한 최소 multiplier
      detectable_at_* : 각 threshold에서 감지 가능 여부
    """
    onset = pd.Timestamp(ev["onset"])
    peak  = pd.Timestamp(ev.get("peak", ev["end"]))
    end   = pd.Timestamp(ev["end"])
    pfu   = ev.get("peak_pfu", None)

    PRE_WIN  = pd.Timedelta(hours=24)
    POST_WIN = pd.Timedelta(hours=24)
    MARGIN   = pd.Timedelta(hours=6)

    row = {
        "onset": str(onset), "peak": str(peak), "end": str(end),
        "peak_pfu": pfu,
        "year": onset.year, "month": onset.month,
        "flare_lead_min": ev.get("flare_lead_min"),
        "xray_class": ev.get("xray_class"),
        "ar_longitude": ev.get("ar_longitude"),
    }

    max_ratios = []

    for pd_key, series in pd_data.items():
        bg = pd_bg[pd_key]

        # 구간 슬라이싱
        pre_seg   = series.loc[onset - PRE_WIN : onset]
        event_seg = series.loc[onset : end]
        post_seg  = series.loc[end   : end + POST_WIN]
        bg_ev     = bg.loc[onset : end]

        if len(event_seg) == 0:
            row[f"{pd_key}_in_data"] = False
            continue
        row[f"{pd_key}_in_data"] = True

        # 사전/사후 평균
        row[f"{pd_key}_pre_mean"]  = round(float(pre_seg.mean()),  1) if len(pre_seg)  else None
        row[f"{pd_key}_post_mean"] = round(float(post_seg.mean()), 1) if len(post_seg) else None

        # 이벤트 구간 통계
        row[f"{pd_key}_event_mean"] = round(float(event_seg.mean()), 1)
        row[f"{pd_key}_event_max"]  = round(float(event_seg.max()),  1)
        row[f"{pd_key}_event_min"]  = round(float(event_seg.min()),  1)
        row[f"{pd_key}_event_std"]  = round(float(event_seg.std()),  1)

        # Background 통계
        bg_vals = bg_ev.dropna()
        bg_mean = float(bg_vals.mean()) if len(bg_vals) else float(pre_seg.mean()) if len(pre_seg) else 1.0
        bg_mean = max(bg_mean, 1.0)
        row[f"{pd_key}_bg_mean"] = round(bg_mean, 1)

        # Ratio
        ratio_ser = event_seg / bg_mean
        max_r     = float(ratio_ser.max())
        max_r_t   = ratio_ser.idxmax()
        row[f"{pd_key}_max_ratio"]    = round(max_r, 3)
        row[f"{pd_key}_max_ratio_t"]  = str(max_r_t)

        # KSEM 피크까지 상승 시간 (onset 기준)
        ksem_peak_t = event_seg.idxmax()
        rise_min = (ksem_peak_t - onset).total_seconds() / 60
        row[f"{pd_key}_ksem_peak_t"]  = str(ksem_peak_t)
        row[f"{pd_key}_ksem_rise_min"] = round(rise_min, 1)

        # SNR 상승 배율 (event_mean / pre_mean)
        if row.get(f"{pd_key}_pre_mean") and row[f"{pd_key}_pre_mean"] > 0:
            row[f"{pd_key}_snr_ratio"] = round(
                row[f"{pd_key}_event_mean"] / row[f"{pd_key}_pre_mean"], 3)

        max_ratios.append(max_r)

    # 3 PD 종합
    if max_ratios:
        row["max_ratio_any"]       = round(max(max_ratios), 3)
        row["max_ratio_mean_pd"]   = round(np.mean(max_ratios), 3)
        row["required_mult"]       = round(max(max_ratios) * 0.95, 3)
        row["in_data"]             = True

        # 각 threshold에서 감지 가능 여부
        for m in THRESHOLD_MULTS:
            row[f"detectable_x{m}"] = bool(max(max_ratios) > m)
    else:
        row["max_ratio_any"]  = None
        row["required_mult"]  = None
        row["in_data"]        = False
        for m in THRESHOLD_MULTS:
            row[f"detectable_x{m}"] = False

    return row


# ============================================================
# 4.  콘솔 리포트
# ============================================================

def print_report(df: pd.DataFrame, pd_keys: List[str]):
    valid = df[df["in_data"] == True].copy()
    no_data = df[df["in_data"] == False]

    print("\n" + "=" * 80)
    print("  spe_vs_ksem  —  NOAA SPE GT 이벤트 vs KSEM Raw 응답 진단")
    print("=" * 80)
    print(f"\n  총 GT 이벤트     : {len(df)}건")
    print(f"  Raw 데이터 있음  : {len(valid)}건")
    print(f"  Raw 데이터 없음  : {len(no_data)}건")

    # ── 이벤트별 상세 표 ──
    print(f"\n{'─'*80}")
    hdr = f"  {'onset':<19} {'pfu':>5}  {'max_ratio':>10}  {'req_×':>7}  {'rise_min':>9}  {'pre→ev':>8}"
    print(hdr)
    print(f"{'─'*80}")

    for _, r in df.sort_values("onset").iterrows():
        if not r.get("in_data", False):
            print(f"  {r['onset']:<19} {str(r.get('peak_pfu','?')):>5}  "
                  f"{'NO DATA':>10}")
            continue

        ratio  = r.get("max_ratio_any", float("nan"))
        req    = r.get("required_mult", float("nan"))

        # PD1 기준 rise time
        rise = r.get("PD1_ksem_rise_min") or r.get("PD2_ksem_rise_min") or r.get("PD3_ksem_rise_min")

        # pre→event 배율 (PD1 우선)
        snr = r.get("PD1_snr_ratio") or r.get("PD2_snr_ratio") or r.get("PD3_snr_ratio")

        ratio_s = f"{ratio:.2f}×" if ratio and not np.isnan(ratio) else "N/A"
        req_s   = f"{req:.2f}×"   if req   and not np.isnan(req)   else "N/A"
        rise_s  = f"{rise:.0f}min"  if rise  else "N/A"
        snr_s   = f"{snr:.2f}×"    if snr   else "N/A"

        print(f"  {r['onset']:<19} {str(r.get('peak_pfu','?')):>5}  "
              f"{ratio_s:>10}  {req_s:>7}  {rise_s:>9}  {snr_s:>8}")

    # ── 놓친 vs 감지된 이벤트 비교 ──
    print(f"\n{'─'*80}")
    print("  [감지된 vs 놓친 이벤트 특성 비교]")
    print(f"{'─'*80}")

    for mult in THRESHOLD_MULTS:
        col = f"detectable_x{mult}"
        det  = valid[valid[col] == True]
        miss = valid[valid[col] == False]
        n_det  = len(det)
        n_miss = len(miss)
        rate   = n_det / len(valid) * 100 if len(valid) else 0
        marker = " ← current" if mult == 21 else ""
        print(f"  ×{mult:<5.2f}  detect={n_det:>2}/{len(valid)}  ({rate:>5.1f}%){marker}")

    # ── 감지된/놓친 이벤트 pfu 비교 ──
    if "detectable_x21" in valid.columns:
        det21  = valid[valid["detectable_x21"] == True]
        miss21 = valid[valid["detectable_x21"] == False]
        print(f"\n  [×21 기준 감지/미감지 특성 비교]")
        print(f"  {'항목':<25} {'감지됨':>12} {'놓침':>12}")
        print(f"  {'─'*50}")

        def cmp(col, fmt=".1f"):
            d = det21[col].dropna()
            m = miss21[col].dropna()
            dv = f"{d.mean():{fmt}}" if len(d) else "N/A"
            mv = f"{m.mean():{fmt}}" if len(m) else "N/A"
            print(f"  {col:<25} {dv:>12} {mv:>12}")

        cmp("peak_pfu", ".0f")
        cmp("max_ratio_any", ".3f")
        for k in pd_keys:
            if f"{k}_ksem_rise_min" in valid.columns:
                cmp(f"{k}_ksem_rise_min", ".0f")
            if f"{k}_snr_ratio" in valid.columns:
                cmp(f"{k}_snr_ratio", ".3f")

    # ── PD별 평균 max_ratio ──
    print(f"\n  [PD별 평균 max_ratio (×1 = background 수준)]")
    for k in pd_keys:
        col = f"{k}_max_ratio"
        if col in valid.columns:
            vals = valid[col].dropna()
            if len(vals):
                print(f"  {k}: mean={vals.mean():.3f}×  "
                      f"min={vals.min():.3f}×  max={vals.max():.3f}×  "
                      f"median={vals.median():.3f}×")

    # ── 시간대별 패턴 ──
    print(f"\n  [이벤트 onset 시간대 분포 (UTC)]")
    valid_copy = valid.copy()
    valid_copy["onset_hour"] = pd.to_datetime(valid_copy["onset"]).dt.hour
    for h_start in range(0, 24, 6):
        h_end = h_start + 6
        cnt = ((valid_copy["onset_hour"] >= h_start) &
               (valid_copy["onset_hour"] < h_end)).sum()
        bar = "█" * cnt
        print(f"  {h_start:02d}-{h_end:02d}UT  {bar} ({cnt})")

    # ── 연도별 ──
    print(f"\n  [연도별 이벤트 수 / 평균 max_ratio]")
    for yr, grp in valid.groupby("year"):
        r_mean = grp["max_ratio_any"].mean()
        print(f"  {yr}: n={len(grp)}  avg_ratio={r_mean:.2f}×")

    print("=" * 80)


# ============================================================
# 5.  종합 시각화
# ============================================================

def plot_overview(df: pd.DataFrame, pd_keys: List[str], path: Path):
    BG    = "#0d1117"; PANEL = "#161b22"
    A1    = "#58a6ff"; A2 = "#f78166"; A3 = "#3fb950"; A4 = "#d2a8ff"
    TEXT  = "#e6edf3"; MUTED = "#8b949e"; GRID = "#21262d"

    valid = df[df["in_data"] == True].copy()
    valid["onset_ts"] = pd.to_datetime(valid["onset"])
    valid = valid.sort_values("onset_ts")

    fig = plt.figure(figsize=(18, 14), facecolor=BG)
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                            hspace=0.50, wspace=0.38,
                            left=0.07, right=0.97, top=0.91, bottom=0.07)

    def sax(ax, title):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=8.5)
        ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)
        ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=7)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
        ax.grid(color=GRID, lw=0.6, alpha=0.8)

    fig.suptitle(
        f"KSEM vs NOAA SPE GT Events  |  n={len(valid)} events with raw data",
        color=TEXT, fontsize=13, fontweight="bold", y=0.97)

    # ── P1 (상단 전체폭): 이벤트별 max ratio 막대 ──
    ax = fig.add_subplot(gs[0, :])
    det_col = "detectable_x21"
    colors  = [A3 if r else A2 for r in valid[det_col]]
    x       = np.arange(len(valid))
    ax.bar(x, valid["max_ratio_any"], color=colors, alpha=0.85, width=0.6, zorder=3)

    # pfu 레이블
    for i, (_, row) in enumerate(valid.iterrows()):
        pfu   = row.get("peak_pfu")
        ratio = row.get("max_ratio_any", 0)
        if pfu and not np.isnan(float(ratio)):
            ax.text(i, float(ratio) + 0.15, f"{pfu}pfu",
                    ha="center", va="bottom", color=MUTED, fontsize=7.5)

    # threshold 라인들
    for mult, col, ls in [(21, A4, "--"), (10, A1, ":"), (3, MUTED, ":")]:
        ax.axhline(mult, color=col, ls=ls, lw=1.5, zorder=4,
                   label=f"×{mult}")
    ax.set_xticks(x)
    ax.set_xticklabels([r["onset"][:10] for _, r in valid.iterrows()],
                       rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("max(value / background)")

    legend_els = [Patch(facecolor=A3, label="Detectable at ×21"),
                  Patch(facecolor=A2, label="Missed at ×21")] + \
                 [Line2D([0], [0], color=c, ls=ls, lw=1.5, label=f"×{m}")
                  for m, c, ls in [(21, A4, "--"), (10, A1, ":"), (3, MUTED, ":")]]
    ax.legend(handles=legend_els, fontsize=8, facecolor=PANEL,
              edgecolor=GRID, labelcolor=TEXT, loc="upper right")
    sax(ax, "Max value/background per GT Event  (green=detectable ×21, red=missed)")

    # ── P2: 임계값별 감지율 ──
    ax = fig.add_subplot(gs[1, 0])
    mults = THRESHOLD_MULTS
    rates = [valid[f"detectable_x{m}"].mean() * 100 for m in mults]
    ax.plot(mults, rates, color=A1, lw=2, marker="o", ms=6, zorder=3)
    ax.axvline(21, color=A4, ls="--", lw=1.5, label="Current ×21", zorder=4)
    ax.fill_between(mults, rates, alpha=0.15, color=A1)
    ax.set_xlabel("Threshold multiplier")
    ax.set_ylabel("Detection Rate (%)")
    ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    ax.set_xscale("log")
    sax(ax, "Threshold vs Detection Rate")

    # ── P3: pfu vs max_ratio scatter ──
    ax = fig.add_subplot(gs[1, 1])
    pfu_v   = valid["peak_pfu"].values.astype(float)
    ratio_v = valid["max_ratio_any"].values.astype(float)
    det_v   = valid[det_col].values
    ax.scatter(pfu_v[det_v],  ratio_v[det_v],  color=A3, s=75,
               alpha=0.9, edgecolors=BG, lw=0.5, zorder=4, label="Detectable")
    ax.scatter(pfu_v[~det_v], ratio_v[~det_v], color=A2, s=75,
               alpha=0.9, edgecolors=BG, lw=0.5, zorder=4, label="Missed")
    ax.axhline(21, color=A4, ls="--", lw=1.5, label="×21 threshold")
    ax.set_xscale("log")
    ax.set_xlabel("GOES Peak Flux (pfu)")
    ax.set_ylabel("KSEM max(value/bg)")
    ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    sax(ax, "GOES Flux vs KSEM Response")

    # ── P4: KSEM 상승 시간 분포 ──
    ax = fig.add_subplot(gs[1, 2])
    rise_cols = [f"{k}_ksem_rise_min" for k in pd_keys
                 if f"{k}_ksem_rise_min" in valid.columns]
    all_rises = pd.concat([valid[c].dropna() for c in rise_cols]).values
    if len(all_rises):
        ax.hist(all_rises / 60, bins=14, color=A3, alpha=0.85,
                edgecolor=BG, zorder=3)
        ax.axvline(np.median(all_rises) / 60, color=A2, ls="--", lw=1.5,
                   label=f"med {np.median(all_rises)/60:.1f}h")
        ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    ax.set_xlabel("KSEM onset → peak (hours)")
    ax.set_ylabel("Count")
    sax(ax, "KSEM Rise Time Distribution")

    # ── P5: SNR ratio 분포 ──
    ax = fig.add_subplot(gs[2, 0])
    snr_cols = [f"{k}_snr_ratio" for k in pd_keys
                if f"{k}_snr_ratio" in valid.columns]
    for k, col in zip(pd_keys, snr_cols):
        vals = valid[col].dropna()
        if len(vals):
            ax.hist(vals, bins=12, alpha=0.6, label=k, edgecolor=BG, zorder=3)
    ax.axvline(1.0, color=MUTED, ls="--", lw=1.2)
    ax.set_xlabel("event_mean / pre_24h_mean")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    sax(ax, "SNR Ratio (event / pre-event 24h)")

    # ── P6: 연도별 max_ratio 박스플롯 ──
    ax = fig.add_subplot(gs[2, 1:])
    years = sorted(valid["year"].unique())
    data  = [valid[valid["year"] == y]["max_ratio_any"].dropna().values
             for y in years]
    bp = ax.boxplot(data, positions=years, widths=0.5, patch_artist=True,
                    medianprops=dict(color=A2, lw=2),
                    boxprops=dict(facecolor=PANEL, edgecolor=A1),
                    whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                    flierprops=dict(marker="o", color=A2, ms=5))
    ax.axhline(21, color=A4, ls="--", lw=1.5, label="×21 threshold")
    ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    ax.set_xlabel("Year"); ax.set_ylabel("max(value/background)")
    sax(ax, "Yearly Distribution of KSEM Response")

    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[Plot1]  -> {path}")


# ============================================================
# 6.  상세 분석 시각화
# ============================================================

def plot_detail(df: pd.DataFrame, pd_keys: List[str], path: Path):
    BG    = "#0d1117"; PANEL = "#161b22"
    A1    = "#58a6ff"; A2 = "#f78166"; A3 = "#3fb950"; A4 = "#d2a8ff"
    TEXT  = "#e6edf3"; MUTED = "#8b949e"; GRID = "#21262d"

    valid = df[df["in_data"] == True].copy()

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), facecolor=BG)
    fig.patch.set_facecolor(BG)
    plt.subplots_adjust(hspace=0.45, wspace=0.35,
                        left=0.07, right=0.97, top=0.91, bottom=0.08)
    fig.suptitle("KSEM Response Detail Analysis", color=TEXT,
                 fontsize=13, fontweight="bold", y=0.97)

    def sax(ax, title):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)
        ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=8)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
        ax.grid(color=GRID, lw=0.6, alpha=0.8)

    # ── D1: pre/event/post count 비교 (PD1) ──
    ax = axes[0, 0]
    k = pd_keys[0]
    pre_col   = f"{k}_pre_mean"
    ev_col    = f"{k}_event_mean"
    post_col  = f"{k}_post_mean"
    sub = valid[[pre_col, ev_col, post_col]].dropna()
    if len(sub):
        x = np.arange(len(sub))
        w = 0.25
        ax.bar(x - w, sub[pre_col],  width=w, color=A1, alpha=0.85, label="Pre-24h")
        ax.bar(x,     sub[ev_col],   width=w, color=A2, alpha=0.85, label="Event")
        ax.bar(x + w, sub[post_col], width=w, color=A3, alpha=0.85, label="Post-24h")
        ax.set_xticks(x)
        ax.set_xticklabels([str(valid.iloc[i]["onset"])[:10]
                            for i in range(len(valid)) if i < len(sub)],
                           rotation=40, ha="right", fontsize=7.5)
        ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    ax.set_ylabel("Count rate")
    sax(ax, f"{k}: Pre / Event / Post mean count")

    # ── D2: PD별 max_ratio 비교 ──
    ax = axes[0, 1]
    ratio_data  = []
    ratio_labels = []
    for k in pd_keys:
        col = f"{k}_max_ratio"
        if col in valid.columns:
            ratio_data.append(valid[col].dropna().values)
            ratio_labels.append(k)
    if ratio_data:
        bp = ax.boxplot(ratio_data, labels=ratio_labels, patch_artist=True,
                        medianprops=dict(color=A2, lw=2),
                        boxprops=dict(facecolor=PANEL, edgecolor=A1),
                        whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                        flierprops=dict(marker="o", color=A2, ms=5))
        ax.axhline(21, color=A4, ls="--", lw=1.5, label="×21")
        ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    ax.set_ylabel("max(value/background)")
    sax(ax, "PD-wise max ratio comparison")

    # ── D3: 월별 발생 / 평균 ratio ──
    ax  = axes[0, 2]
    ax2 = ax.twinx()
    valid["month_int"] = pd.to_datetime(valid["onset"]).dt.month
    mon_cnt   = valid.groupby("month_int").size()
    mon_ratio = valid.groupby("month_int")["max_ratio_any"].mean()
    months    = range(1, 13)
    mon_labels = ["J","F","M","A","M","J","J","A","S","O","N","D"]
    cnt_vals   = [mon_cnt.get(m, 0) for m in months]
    ratio_vals = [mon_ratio.get(m, 0) for m in months]
    ax.bar(months, cnt_vals, color=A1, alpha=0.7, width=0.5, zorder=3)
    ax2.plot(months, ratio_vals, color=A2, lw=2, marker="o", ms=5, zorder=4)
    ax.set_xticks(list(months))
    ax.set_xticklabels(mon_labels)
    ax.set_ylabel("Event count", color=A1)
    ax2.set_ylabel("Avg max ratio", color=A2)
    ax2.tick_params(colors=MUTED, labelsize=8.5)
    sax(ax, "Monthly Distribution & Avg KSEM Response")

    # ── D4: Flare lead time vs KSEM rise time (PD1) ──
    ax = axes[1, 0]
    k  = pd_keys[0]
    flead = valid["flare_lead_min"].astype(float)
    krise = valid[f"{k}_ksem_rise_min"].astype(float) if f"{k}_ksem_rise_min" in valid.columns else pd.Series()
    mask  = ~(flead.isna() | krise.isna()) if len(krise) else pd.Series([False]*len(valid))
    if mask.sum() > 1:
        ax.scatter(flead[mask] / 60, krise[mask] / 60,
                   color=A4, s=70, alpha=0.9, edgecolors=BG, lw=0.5, zorder=3)
        # 대각선 (1:1)
        lim = max(flead[mask].max(), krise[mask].max()) / 60
        ax.plot([0, lim], [0, lim], color=MUTED, ls="--", lw=1.2, zorder=2)
        ax.set_xlabel("Flare → SEP onset (hours)")
        ax.set_ylabel(f"{k} onset → KSEM peak (hours)")
    sax(ax, "Flare Lead Time vs KSEM Rise Time")

    # ── D5: required_mult 히스토그램 ──
    ax = axes[1, 1]
    req = valid["required_mult"].dropna()
    if len(req):
        bins = np.linspace(0, max(req.max() * 1.1, 30), 20)
        ax.hist(req, bins=bins, color=A2, alpha=0.85, edgecolor=BG, zorder=3)
        ax.axvline(21, color=A4, ls="--", lw=1.8, label="Current ×21")
        ax.axvline(req.median(), color=A1, ls="--", lw=1.4,
                   label=f"Median ×{req.median():.1f}")
        pct_below = (req < 21).mean() * 100
        ax.text(0.97, 0.95,
                f"{pct_below:.0f}% of events\nneed < ×21",
                transform=ax.transAxes, color=A3, fontsize=9,
                ha="right", va="top",
                bbox=dict(facecolor=PANEL, edgecolor=GRID,
                          boxstyle="round,pad=0.3"))
        ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    ax.set_xlabel("Required threshold multiplier")
    ax.set_ylabel("Count")
    sax(ax, "Required Threshold to Detect Each Event")

    # ── D6: max_ratio 누적 분포 (CDF) ──
    ax = axes[1, 2]
    ratio_all = valid["max_ratio_any"].dropna().sort_values()
    cdf = np.arange(1, len(ratio_all)+1) / len(ratio_all)
    ax.plot(ratio_all, cdf, color=A1, lw=2.5, zorder=3)
    ax.fill_betweenx(cdf, ratio_all, alpha=0.12, color=A1)
    for mult, col, ls in [(21, A4, "--"), (10, A1, ":"), (5, A3, ":")]:
        pct = (ratio_all < mult).mean() * 100
        ax.axvline(mult, color=col, ls=ls, lw=1.5, zorder=4,
                   label=f"×{mult}: {100-pct:.0f}% detectable")
    ax.set_xlabel("max(value/background)")
    ax.set_ylabel("Cumulative fraction")
    ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    sax(ax, "CDF of KSEM max ratio (GT events)")

    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[Plot2]  -> {path}")


# ============================================================
# 7.  이벤트별 시계열 PNG
# ============================================================

def plot_timeseries_per_event(ev: dict,
                              pd_raw: Dict[str, pd.Series],
                              pd_bg:  Dict[str, pd.Series],
                              out_dir: Path):
    BG   = "#0d1117"; PANEL = "#161b22"
    A1   = "#58a6ff"; A2 = "#f78166"; A3 = "#3fb950"
    TEXT = "#e6edf3"; MUTED = "#8b949e"; GRID = "#21262d"

    onset = pd.Timestamp(ev["onset"])
    end   = pd.Timestamp(ev["end"])
    pfu   = ev.get("peak_pfu", "?")
    margin = pd.Timedelta(hours=24)

    fig, axes = plt.subplots(len(pd_raw), 1,
                             figsize=(14, 3.5 * len(pd_raw)),
                             facecolor=BG, sharex=True)
    if len(pd_raw) == 1:
        axes = [axes]
    fig.suptitle(
        f"GT Event: {ev['onset'][:16]}  |  GOES {pfu} pfu",
        color=TEXT, fontsize=11, fontweight="bold", y=1.01)

    colors = [A1, A2, A3]
    for ax, (k, series), col in zip(axes, pd_raw.items(), colors):
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
        ax.grid(color=GRID, lw=0.5, alpha=0.7)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.yaxis.label.set_color(MUTED)

        seg = series.loc[onset - margin : end + margin]
        bg  = pd_bg[k].loc[onset - margin : end + margin]

        ax.plot(seg.index, seg.values, color=col, lw=1.2, alpha=0.9,
                label=k, zorder=3)
        ax.plot(bg.index, bg.values, color=MUTED, lw=1.2, ls="--",
                alpha=0.7, label="BG (5d mean)", zorder=2)

        # threshold 라인들
        bg_mean = float(bg.dropna().mean()) if len(bg.dropna()) else 1.0
        for mult, ls in [(21, "--"), (10, ":")]:
            ax.axhline(bg_mean * mult, color=MUTED, ls=ls, lw=0.9, alpha=0.6,
                       label=f"×{mult}")

        # onset / end 수직선
        ax.axvspan(onset, end, alpha=0.12, color=A2, zorder=1)
        ax.axvline(onset, color=A2, ls="-", lw=1.5, zorder=4)
        ax.axvline(end,   color=A2, ls=":", lw=1.2, zorder=4)

        ax.set_ylabel("Count rate", color=MUTED)
        ax.legend(fontsize=7.5, facecolor=PANEL, edgecolor=GRID,
                  labelcolor=TEXT, loc="upper right")
        ax.set_title(k, color=TEXT, fontsize=9, pad=4)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=12))
    plt.xticks(rotation=25, ha="right", color=MUTED, fontsize=8)

    fname = onset.strftime("%Y%m%d_%H%M") + f"_{pfu}pfu.png"
    fpath = out_dir / fname
    plt.tight_layout()
    plt.savefig(fpath, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()


# ============================================================
# 8.  MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="NOAA SPE GT 이벤트 vs KSEM Raw Data 응답 진단")
    parser.add_argument("--root",        required=True,
                        help="Raw Count 루트 폴더 (D:\\Raw Count)")
    parser.add_argument("--events",      required=True,
                        help="known_events_{tag}.json")
    parser.add_argument("--output_dir",  default="./spe_vs_ksem_out")
    parser.add_argument("--start",       default="201905",
                        help="분석 시작 YYYYMM (기본: 201905)")
    parser.add_argument("--end",         default="202412",
                        help="분석 종료 YYYYMM (기본: 202412)")
    parser.add_argument("--plot_timeseries", action="store_true",
                        help="이벤트별 시계열 PNG 저장 (시간 소요)")
    args = parser.parse_args()

    root = Path(args.root)
    out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # GT 이벤트 로드
    with open(args.events, encoding="utf-8") as f:
        events = json.load(f)
    print(f"[Events] {len(events)}건  ({args.events})")

    # Raw 데이터 로드 (전체 범위 한 번에)
    pd_keys = ["PD1", "PD2", "PD3"]
    pd_raw: Dict[str, pd.Series] = {}
    pd_bg:  Dict[str, pd.Series] = {}

    for key in pd_keys:
        print(f"[Load] {key} raw data ... ", end="", flush=True)
        try:
            s = load_raw_range(root, args.start, args.end, key)
            pd_raw[key] = s
            pd_bg[key]  = compute_rolling_bg(s)
            print(f"{len(s)}포인트 ({len(s)*15/1440:.1f}일)")
        except FileNotFoundError as e:
            print(f"SKIP ({e})")

    if not pd_raw:
        print("[ERROR] 로드된 PD 데이터 없음. --root 경로 확인.")
        return

    # 이벤트별 분석
    print(f"\n[Analyze] {len(events)}개 이벤트 분석 중...")
    rows = []
    ts_dir = out / "timeseries"
    if args.plot_timeseries:
        ts_dir.mkdir(exist_ok=True)

    for i, ev in enumerate(events):
        row = analyze_event(ev, pd_raw, pd_bg)
        rows.append(row)
        status = "OK" if row.get("in_data") else "NO DATA"
        ratio  = row.get("max_ratio_any", "N/A")
        ratio_s = f"{ratio:.2f}×" if isinstance(ratio, float) else str(ratio)
        print(f"  [{i+1:02d}/{len(events)}] {ev['onset'][:16]}  "
              f"pfu={ev.get('peak_pfu','?'):>5}  "
              f"max_ratio={ratio_s:>8}  {status}")

        if args.plot_timeseries and row.get("in_data"):
            plot_timeseries_per_event(ev, pd_raw, pd_bg, ts_dir)

    result_df = pd.DataFrame(rows)

    # 콘솔 리포트
    print_report(result_df, list(pd_raw.keys()))

    # CSV
    csv_path = out / "spe_vs_ksem_summary.csv"
    result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[Table]  -> {csv_path}")

    # 시각화
    plot_overview(result_df, list(pd_raw.keys()), out / "spe_vs_ksem_overview.png")
    plot_detail(result_df,   list(pd_raw.keys()), out / "spe_vs_ksem_detail.png")

    print("\n[완료]")
    print(f"  python spe_vs_ksem.py \\")
    print(f"      --root \"{args.root}\" \\")
    print(f"      --events \"{args.events}\" \\")
    print(f"      --output_dir \"{args.output_dir}\"")


if __name__ == "__main__":
    main()
