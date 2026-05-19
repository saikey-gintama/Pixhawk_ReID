"""
spe_to_json.py  —  NOAA SPE.txt 완전 분석기 v3
=================================================
출력:
  1. known_events_{tag}.json   — evaluate.py 호환
  2. spe_stats_{tag}.json      — 전체 통계 지표
  3. spe_analysis_{tag}.png    — 종합 시각화 (8패널)
  4. spe_table_{tag}.csv       — 이벤트별 파생 지표 테이블
  5. spe_correlation_{tag}.png — 상관성 분석 시각화

파싱 항목:
  onset, peak, pfu, CME방향, flare_max시각,
  xray_class, optical_class, ar_location, ar_region

파생 지표:
  duration_hr     : onset → peak (hours)
  flare_lead_min  : SEP onset − flare max (min, 양수=SEP after flare)
  ar_longitude    : W=양수, E=음수 (degrees)
  ar_latitude     : N=양수, S=음수
  xray_numeric    : A=0.1×, B=1×, C=10×, M=100×, X=1000× × 수치
  optical_numeric : SF=0, SN/1B=1~2, 2B=3, 3B=4, 4B=5

사용법:
  python spe_to_json.py --input SPE.txt --output_dir ./spe_out
  python spe_to_json.py --input SPE.txt --output_dir ./spe_out \\
      --start_year 2019 --end_year 2024 --buffer_hours 48
"""

import argparse
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from scipy import stats as scipy_stats

DEFAULT_BUFFER_HOURS = 48

MONTH_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
    "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
    "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

OPTICAL_NUM = {
    "SF": 0, "SN": 1, "SB": 1,
    "1F": 1, "1N": 2, "1B": 2,
    "2F": 2, "2N": 3, "2B": 3,
    "3F": 3, "3N": 4, "3B": 4,
    "4F": 4, "4N": 5, "4B": 5,
}


# ============================================================
# 0.  유틸
# ============================================================

def xray_to_numeric(s: str) -> Optional[float]:
    if not s or s.strip() in ("N/A", ""):
        return None
    m = re.match(r"([ABCMX])([\d.]+)", s.strip())
    if not m:
        return None
    base = {"A": 0.1, "B": 1.0, "C": 10.0, "M": 100.0, "X": 1000.0}
    return base.get(m.group(1), 0) * float(m.group(2))


def parse_dt(s: str, year: int) -> Optional[datetime]:
    """'Apr 30/2120' or 'May 01/1700'"""
    m = re.match(r"([A-Za-z]{3})\s+(\d{1,2})/(\d{4})", s.strip())
    if not m:
        return None
    mon = MONTH_MAP.get(m.group(1)[:3].capitalize())
    if not mon:
        return None
    try:
        return datetime(year, mon, int(m.group(2)),
                        int(m.group(3)[:2]), int(m.group(3)[2:]))
    except ValueError:
        return None


def parse_flare_dt(s: str, ref_year: int, ref_month: int,
                   onset: datetime) -> Optional[datetime]:
    """
    CME/Flare 컬럼: 'May 28/2312', 'Apr 30/2114', 'Dec 31/2135'
    연도는 onset 기준으로 추정.
    """
    s = s.strip()
    if s in ("N/A", ""):
        return None
    m = re.match(r"([A-Za-z]{3})\s+(\d{1,2})/(\d{4})", s)
    if m:
        mon = MONTH_MAP.get(m.group(1)[:3].capitalize())
        if not mon:
            return None
        year = onset.year
        # 12월 이벤트인데 flare가 1월 → 전년도
        if onset.month == 1 and mon == 12:
            year -= 1
        # 1월 이벤트인데 flare가 12월 → 전년도
        if onset.month >= 2 and mon == 12 and mon > onset.month:
            year -= 1
        try:
            dt = datetime(year, mon, int(m.group(2)),
                          int(m.group(3)[:2]), int(m.group(3)[2:]))
            # flare가 onset보다 30일 이상 미래면 오류
            if abs((dt - onset).days) > 30:
                return None
            return dt
        except ValueError:
            return None
    return None


def parse_location(s: str) -> Tuple[Optional[float], Optional[float]]:
    """'S09W47' → (lat=-9, lon=+47)  'N18W70' → (lat=+18, lon=+70)"""
    m = re.match(r"([NS])(\d+)([EW])(\d+)", s.strip())
    if not m:
        return None, None
    lat = int(m.group(2)) * (1 if m.group(1) == "N" else -1)
    lon = int(m.group(4)) * (1 if m.group(3) == "W" else -1)
    return float(lat), float(lon)


def parse_xray_optical(s: str) -> Tuple[Optional[str], Optional[str]]:
    """'X2/2B' → ('X2','2B')  'M5.0' → ('M5.0',None)  'N/A' → (None,None)"""
    s = s.strip()
    if s in ("N/A", "", "N/"):
        return None, None
    parts = s.split("/")
    xray = parts[0].strip() or None
    opt  = parts[1].strip() if len(parts) > 1 else None
    opt  = opt or None
    return xray, opt


def parse_cme_dir(s: str) -> Optional[str]:
    """'NW/09 0212' → 'NW'"""
    m = re.match(r"([NSEW]{1,2})/", s.strip())
    return m.group(1) if m else None


def is_cme_field(s: str) -> bool:
    """필드가 CME 컬럼인지 판별 (Mon DD/HHMM 이 아닌 경우)"""
    # CME: 'NW/09 0212', 'SE/26 2024', '/21 1805'
    # Flare: 'May 28/2312', 'Apr 30/2114'
    return bool(re.match(r"[NSEW]{0,2}/", s.strip()))


# ============================================================
# 1.  파서
# ============================================================

def parse_spe_txt(filepath: Path,
                  start_year: Optional[int],
                  end_year:   Optional[int],
                  buffer_hours: int) -> Tuple[list, int]:
    events  = []
    skipped = 0
    current_year = None

    # onset + peak + pfu 고정, 나머지는 2+ 공백 분리
    head_re = re.compile(
        r"^([A-Za-z]{3}\s+\d{1,2}/\d{4})\s+"
        r"([A-Za-z]{3}\s+\d{1,2}/\d{4})\s+"
        r"(\d+)\s*(.*)"
    )
    year_re = re.compile(r"^\s{0,20}(\d{4})\s*$")

    with open(filepath, encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip()

            ym = year_re.match(line)
            if ym:
                current_year = int(ym.group(1))
                continue
            if current_year is None:
                continue

            dm = head_re.match(line)
            if not dm:
                continue

            onset = parse_dt(dm.group(1), current_year)
            peak  = parse_dt(dm.group(2), current_year)
            if onset is None or peak is None:
                continue

            # 연말→연초 넘김
            if peak < onset:
                pn = parse_dt(dm.group(2), current_year + 1)
                if pn is None or pn < onset or (pn - onset).days > 30:
                    print(f"  [SKIP 오기입] {onset:%Y-%m-%d %H:%M}")
                    skipped += 1
                    continue
                peak = pn
            if (peak - onset).days > 30:
                print(f"  [SKIP 비정상] {onset:%Y-%m-%d %H:%M}")
                skipped += 1
                continue

            # 범위 필터
            if start_year and onset.year < start_year:
                continue
            if end_year and onset.year > end_year:
                continue

            # 나머지 컬럼: 공백 2개+ 기준 분리
            rest = dm.group(4).strip()
            fields = re.split(r"\s{2,}", rest) if rest else []
            # 빈 문자열 제거
            fields = [f for f in fields if f.strip()]

            # 필드 순서: [CME?] [Flare] [Importance] [Location] [Region]
            # CME 있으면 5개, 없으면 4개
            idx = 0
            cme_raw = ""
            flare_raw = ""
            importance_raw = ""
            location_raw = ""
            region_raw = ""

            if fields:
                # CME 여부 판별
                if is_cme_field(fields[0]):
                    cme_raw = fields[0]; idx = 1
                else:
                    idx = 0

                if idx < len(fields):
                    flare_raw = fields[idx]; idx += 1
                if idx < len(fields):
                    importance_raw = fields[idx]; idx += 1
                if idx < len(fields):
                    location_raw = fields[idx]; idx += 1
                if idx < len(fields):
                    region_raw = fields[idx]

            # flare max 시각
            flare_dt = parse_flare_dt(flare_raw, onset.year, onset.month, onset)

            # lead time (분): SEP onset − flare max
            flare_lead_min = None
            if flare_dt is not None:
                flare_lead_min = round((onset - flare_dt).total_seconds() / 60.0, 1)

            # X-ray / Optical
            xray_str, optical_str = parse_xray_optical(importance_raw)
            xray_num    = xray_to_numeric(xray_str)
            optical_num = OPTICAL_NUM.get(optical_str) if optical_str else None

            # AR 위치
            ar_lat, ar_lon = parse_location(location_raw)

            # CME 방향
            cme_dir = parse_cme_dir(cme_raw)

            end_dt = peak + timedelta(hours=buffer_hours)

            events.append({
                "onset":           onset.strftime("%Y-%m-%d %H:%M"),
                "peak":            peak.strftime("%Y-%m-%d %H:%M"),
                "end":             end_dt.strftime("%Y-%m-%d %H:%M"),
                "peak_pfu":        int(dm.group(3)),
                "duration_hr":     round((peak - onset).total_seconds() / 3600, 2),
                "flare_max_ut":    flare_dt.strftime("%Y-%m-%d %H:%M") if flare_dt else None,
                "flare_lead_min":  flare_lead_min,
                "xray_class":      xray_str,
                "xray_numeric":    xray_num,
                "optical_class":   optical_str,
                "optical_numeric": optical_num,
                "ar_location":     location_raw or None,
                "ar_latitude":     ar_lat,
                "ar_longitude":    ar_lon,
                "cme_direction":   cme_dir,
                "ar_region":       region_raw or None,
                "year":            onset.year,
                "month":           onset.month,
            })

    return events, skipped


# ============================================================
# 2.  통계
# ============================================================

def compute_stats(events: list, start_year, end_year, buffer_hours) -> dict:
    if not events:
        return {}
    df = pd.DataFrame(events)

    y0 = start_year or int(df["year"].min())
    y1 = end_year   or int(df["year"].max())
    total_days = (datetime(y1, 12, 31) - datetime(y0, 1, 1)).days + 1
    total_pts  = total_days * 24 * 4

    # GT 커버리지
    ts_set = set()
    for ev in events:
        cur = pd.Timestamp(ev["onset"])
        end = pd.Timestamp(ev["end"])
        while cur <= end:
            ts_set.add(cur)
            cur += pd.Timedelta(minutes=15)
    gt_pts = len(ts_set)
    gt_pct = round(gt_pts / total_pts * 100, 3)

    pfu  = df["peak_pfu"].values.astype(float)
    dur  = df["duration_hr"].values.astype(float)
    lead = df["flare_lead_min"].values.astype(float)
    xnum = df["xray_numeric"].values.astype(float)
    lon  = df["ar_longitude"].values.astype(float)
    opt  = df["optical_numeric"].values.astype(float)

    def safe_corr(a, b):
        mask = ~(np.isnan(a) | np.isnan(b))
        if mask.sum() < 3:
            return None, None
        r, p = scipy_stats.pearsonr(a[mask], b[mask])
        return round(float(r), 3), round(float(p), 4)

    r_xray, p_xray = safe_corr(xnum, pfu)
    r_lon,  p_lon  = safe_corr(lon,  pfu)
    r_opt,  p_opt  = safe_corr(opt,  pfu)
    r_lead, p_lead = safe_corr(lead, pfu)

    def nanstat(arr, fn):
        v = arr[~np.isnan(arr)]
        return round(float(fn(v)), 2) if len(v) else None

    return {
        "analysis_period":    f"{y0}~{y1}",
        "total_days":         total_days,
        "n_events":           len(events),
        "events_per_year":    round(len(events) / max(y1 - y0 + 1, 1), 2),

        "duration_hr_mean":   nanstat(dur, np.mean),
        "duration_hr_median": nanstat(dur, np.median),
        "duration_hr_min":    nanstat(dur, np.min),
        "duration_hr_max":    nanstat(dur, np.max),
        "duration_hr_p90":    round(float(np.nanpercentile(dur, 90)), 2),

        "pfu_mean":           round(float(np.nanmean(pfu)), 1),
        "pfu_median":         round(float(np.nanmedian(pfu)), 1),
        "pfu_min":            int(np.nanmin(pfu)),
        "pfu_max":            int(np.nanmax(pfu)),
        "pfu_p90":            round(float(np.nanpercentile(pfu, 90)), 1),
        "n_S1":               int(((pfu >= 10)  & (pfu < 100)).sum()),
        "n_S2":               int(((pfu >= 100) & (pfu < 1000)).sum()),
        "n_S3plus":           int((pfu >= 1000).sum()),

        "n_with_flare":       int((~np.isnan(lead)).sum()),
        "flare_lead_mean":    nanstat(lead, np.mean),
        "flare_lead_median":  nanstat(lead, np.median),
        "flare_lead_min_val": nanstat(lead, np.min),
        "flare_lead_max_val": nanstat(lead, np.max),
        "n_flare_before_sep": int((lead[~np.isnan(lead)] > 0).sum()),
        "n_sep_before_flare": int((lead[~np.isnan(lead)] < 0).sum()),

        "n_with_location":    int((~np.isnan(lon)).sum()),
        "ar_lon_mean":        nanstat(lon, np.mean),
        "ar_lon_west_count":  int((lon[~np.isnan(lon)] > 0).sum()),
        "ar_lon_east_count":  int((lon[~np.isnan(lon)] < 0).sum()),

        "corr_xray_pfu":      r_xray, "p_xray_pfu":    p_xray,
        "corr_lon_pfu":       r_lon,  "p_lon_pfu":     p_lon,
        "corr_optical_pfu":   r_opt,  "p_optical_pfu": p_opt,
        "corr_lead_pfu":      r_lead, "p_lead_pfu":    p_lead,

        "gt_event_pts":       gt_pts,
        "gt_event_pct":       gt_pct,
        "gt_note":            f"15min cadence, overlap-removed. Target ALERT upper bound = {gt_pct}%",

        "yearly_counts":      {str(y): int(c)
                               for y, c in df.groupby("year").size().items()},
        "buffer_hours_used":  buffer_hours,
    }


# ============================================================
# 3.  테이블 CSV
# ============================================================

def save_table(events: list, path: Path):
    cols = [
        "onset", "peak", "end", "peak_pfu",
        "duration_hr", "flare_max_ut", "flare_lead_min",
        "xray_class", "xray_numeric", "optical_class", "optical_numeric",
        "ar_location", "ar_latitude", "ar_longitude",
        "cme_direction", "ar_region",
    ]
    df = pd.DataFrame(events)[cols]
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[Table]  -> {path}")


# ============================================================
# 4.  종합 시각화
# ============================================================

def plot_overview(events: list, stats: dict, path: Path):
    df = pd.DataFrame(events)
    BG = "#0d1117"; PANEL = "#161b22"
    A1 = "#58a6ff"; A2 = "#f78166"; A3 = "#3fb950"; A4 = "#d2a8ff"
    TEXT = "#e6edf3"; MUTED = "#8b949e"; GRID = "#21262d"

    fig = plt.figure(figsize=(18, 13), facecolor=BG)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38,
                            left=0.07, right=0.97, top=0.90, bottom=0.07)

    def sax(ax, title):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=8.5)
        ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)
        ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=7)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
        ax.grid(color=GRID, lw=0.6, alpha=0.8)

    fig.suptitle(
        f"NOAA SPE Catalog  |  {stats['analysis_period']}  |  "
        f"n={stats['n_events']} events  |  "
        f"GT coverage = {stats['gt_event_pct']}%  <- target ALERT upper bound",
        color=TEXT, fontsize=12, fontweight="bold", y=0.97)

    # P1: 연도별 빈도
    ax = fig.add_subplot(gs[0, :2])
    yc = stats["yearly_counts"]
    if yc:
        yrs, cnts = zip(*sorted((int(y), c) for y, c in yc.items()))
        bars = ax.bar(yrs, cnts, color=A1, alpha=0.85, width=0.6, zorder=3)
        for b, c in zip(bars, cnts):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.05,
                    str(c), ha="center", va="bottom", color=TEXT, fontsize=9)
    ax.axhline(stats["events_per_year"], color=A2, ls="--", lw=1.5,
               label=f"avg {stats['events_per_year']:.1f}/yr", zorder=4)
    ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("# events"); sax(ax, "Yearly Event Frequency")

    # P2: pfu 분포
    ax = fig.add_subplot(gs[0, 2])
    pfu = df["peak_pfu"].values
    bins = np.logspace(np.log10(max(pfu.min(), 1)), np.log10(pfu.max()+1), 18)
    ax.hist(pfu, bins=bins, color=A2, alpha=0.85, edgecolor=BG, zorder=3)
    ax.set_xscale("log")
    ax.axvline(100,  color=A1, ls="--", lw=1.2, label="S2 100 pfu")
    ax.axvline(1000, color=A3, ls="--", lw=1.2, label="S3 1000 pfu")
    ax.legend(fontsize=7.5, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    ax.set_xlabel("Peak flux (pfu)"); ax.set_ylabel("Count")
    sax(ax, "Peak Flux Distribution")

    # P3: 상승시간 분포
    ax = fig.add_subplot(gs[1, 0])
    dur = df["duration_hr"].values
    ax.hist(dur, bins=14, color=A3, alpha=0.85, edgecolor=BG, zorder=3)
    ax.axvline(float(np.median(dur)), color=A2, ls="--", lw=1.5,
               label=f"med {np.median(dur):.1f}h")
    ax.axvline(float(np.mean(dur)), color=A1, ls=":", lw=1.5,
               label=f"mean {np.mean(dur):.1f}h")
    ax.legend(fontsize=7.5, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    ax.set_xlabel("Rise time onset->peak (hr)"); ax.set_ylabel("Count")
    sax(ax, "Rise Time Distribution")

    # P4: Flare Lead Time 분포
    ax = fig.add_subplot(gs[1, 1])
    lead = df["flare_lead_min"].dropna().values
    if len(lead) > 0:
        ax.hist(lead, bins=16, color=A4, alpha=0.85, edgecolor=BG, zorder=3)
        ax.axvline(0, color=A2, ls="-", lw=1.5, label="onset = flare max", zorder=4)
        ax.axvline(float(np.median(lead)), color=A1, ls="--", lw=1.3,
                   label=f"med {np.median(lead):.0f} min")
        ax.legend(fontsize=7.5, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
        ax.set_xlabel("SEP onset - Flare max (min)\n(+: SEP after flare)")
        ax.set_ylabel("Count")
    sax(ax, f"Flare->SEP Lead Time  (n={len(lead)})")

    # P5: AR 경도 분포
    ax = fig.add_subplot(gs[1, 2])
    lons = df["ar_longitude"].dropna().values
    if len(lons) > 0:
        ax.hist(lons, bins=14, color=A1, alpha=0.85, edgecolor=BG, zorder=3)
        ax.axvline(0, color=A2, ls="-", lw=1.5, label="Central Meridian")
        if len(lons) >= 2:
            ax.axvline(float(np.median(lons)), color=A3, ls="--", lw=1.3,
                       label=f"med W{np.median(lons):.0f}deg")
        ax.legend(fontsize=7.5, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    ax.set_xlabel("AR Longitude (W=+, E=-) [degrees]")
    ax.set_ylabel("Count")
    sax(ax, f"AR Longitude Distribution  (n={len(lons)})")

    # P6: 강도 분류 파이
    ax = fig.add_subplot(gs[2, 0])
    items = [("S1 10-99",  stats["n_S1"],    A1),
             ("S2 100-999", stats["n_S2"],   A2),
             ("S3+ 1000+", stats["n_S3plus"], A3)]
    nz = [(l, s, c) for l, s, c in items if s > 0]
    if nz:
        ax.pie([s for _, s, _ in nz],
               labels=[f"{l}\n({s})" for l, s, _ in nz],
               colors=[c for _, _, c in nz],
               autopct="%1.0f%%", startangle=90,
               textprops={"color": TEXT, "fontsize": 8.5},
               wedgeprops={"edgecolor": BG, "linewidth": 1.5})
    ax.set_facecolor(PANEL)
    ax.set_title("NOAA Scale Distribution", color=TEXT,
                 fontsize=10, fontweight="bold", pad=7)

    # P7: pfu vs AR 경도 scatter
    ax = fig.add_subplot(gs[2, 1])
    mask = ~(df["ar_longitude"].isna() | df["peak_pfu"].isna())
    if mask.sum() > 1:
        x = df.loc[mask, "ar_longitude"].values
        y = np.log10(df.loc[mask, "peak_pfu"].values)
        sc = ax.scatter(x, y, c=df.loc[mask, "year"].values,
                        cmap="plasma", s=65, alpha=0.85,
                        edgecolors=BG, lw=0.5, zorder=3)
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.ax.tick_params(colors=MUTED, labelsize=7)
        r = stats.get("corr_lon_pfu")
        p = stats.get("p_lon_pfu")
        if r is not None:
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
            ax.text(0.05, 0.95, f"r={r:.3f}  {sig}\np={p:.4f}",
                    transform=ax.transAxes, color=A2, fontsize=9, va="top",
                    bbox=dict(facecolor=PANEL, edgecolor=GRID,
                              boxstyle="round,pad=0.3", alpha=0.9))
        ax.axvline(0, color=MUTED, ls="--", lw=0.8)
    ax.set_xlabel("AR Longitude (W=+, E=-)")
    ax.set_ylabel("log10 Peak flux (pfu)")
    sax(ax, "AR Longitude vs Peak Flux")

    # P8: 이벤트 타임라인
    ax = fig.add_subplot(gs[2, 2])
    for ev in events:
        onset_t = pd.Timestamp(ev["onset"])
        peak_t  = pd.Timestamp(ev["peak"])
        pfu_v   = ev["peak_pfu"]
        col = A3 if pfu_v >= 1000 else (A2 if pfu_v >= 100 else A1)
        ax.plot([onset_t, peak_t], [pfu_v, pfu_v],
                color=col, lw=2.5, alpha=0.85, solid_capstyle="round", zorder=3)
        ax.scatter([onset_t], [pfu_v], color=col, s=22, zorder=4)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("'%y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.set_xlabel("Year"); ax.set_ylabel("Peak flux (pfu)")
    sax(ax, "Event Timeline  (blue=S1 / orange=S2 / green=S3+)")

    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[Plot1]  -> {path}")


# ============================================================
# 5.  상관성 분석 시각화
# ============================================================

def plot_correlation(events: list, stats: dict, path: Path):
    df = pd.DataFrame(events)
    df["log_pfu"] = np.log10(df["peak_pfu"].clip(lower=1))

    BG = "#0d1117"; PANEL = "#161b22"
    A1 = "#58a6ff"; A2 = "#f78166"; A3 = "#3fb950"; A4 = "#d2a8ff"
    TEXT = "#e6edf3"; MUTED = "#8b949e"; GRID = "#21262d"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=BG)
    fig.patch.set_facecolor(BG)
    plt.subplots_adjust(hspace=0.42, wspace=0.35,
                        left=0.09, right=0.97, top=0.91, bottom=0.08)
    fig.suptitle(f"SPE Correlation Analysis  |  {stats['analysis_period']}  |  n={stats['n_events']}",
                 color=TEXT, fontsize=13, fontweight="bold", y=0.97)

    def sax(ax, title):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)
        ax.set_title(title, color=TEXT, fontsize=10.5, fontweight="bold", pad=8)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
        ax.grid(color=GRID, lw=0.6, alpha=0.8)

    def scatter_fit(ax, xcol, ycol, color, xlabel, title, rk, pk, logx=False):
        sub = df[[xcol, ycol]].dropna()
        if len(sub) < 3:
            ax.text(0.5, 0.5, f"Insufficient data\n(n={len(sub)})",
                    ha="center", va="center", color=MUTED,
                    transform=ax.transAxes, fontsize=10)
            sax(ax, title); return
        x = sub[xcol].values.astype(float)
        y = sub[ycol].values.astype(float)
        xp = np.log10(np.clip(x, 1e-9, None)) if logx else x

        ax.scatter(x, y, color=color, s=70, alpha=0.85,
                   edgecolors=BG, lw=0.5, zorder=3)

        # 회귀선
        try:
            m, b, r_val, p_val, _ = scipy_stats.linregress(xp, y)
            xfit = np.linspace(xp.min(), xp.max(), 100)
            yfit = m * xfit + b
            xfit_orig = 10**xfit if logx else xfit
            ax.plot(xfit_orig, yfit, color=A2, lw=2.0, ls="--", zorder=4, alpha=0.9)
            rv = stats.get(rk, round(r_val, 3))
            pv = stats.get(pk, round(p_val, 4))
            if rv is None: rv = round(r_val, 3)
            if pv is None: pv = round(p_val, 4)
            sig = "***" if pv < 0.001 else ("**" if pv < 0.01 else ("*" if pv < 0.05 else "n.s."))
            ax.text(0.05, 0.95,
                    f"r = {rv:.3f}  {sig}\np = {pv:.4f}  n={len(sub)}",
                    transform=ax.transAxes, color=A2, fontsize=9, va="top",
                    bbox=dict(facecolor=PANEL, edgecolor=GRID,
                              boxstyle="round,pad=0.3", alpha=0.9))
        except Exception:
            pass

        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel); ax.set_ylabel("log10 Peak Flux (pfu)")
        sax(ax, title)

    # C1: X-ray class vs pfu
    scatter_fit(axes[0, 0], "xray_numeric", "log_pfu", A1,
                "X-ray Numeric (C=10, M=100, X=1000)",
                "X-ray Class vs Peak Flux",
                "corr_xray_pfu", "p_xray_pfu", logx=True)

    # C2: AR 경도 vs pfu
    scatter_fit(axes[0, 1], "ar_longitude", "log_pfu", A3,
                "AR Longitude (W=+, E=-) [degrees]",
                "AR Longitude vs Peak Flux",
                "corr_lon_pfu", "p_lon_pfu")

    # C3: Optical class vs pfu (boxplot + scatter)
    ax = axes[1, 0]
    sub_opt = df[["optical_numeric", "log_pfu"]].dropna()
    if len(sub_opt) >= 3:
        opt_labels = {0: "SF", 1: "SN/1F", 2: "1N/1B",
                      3: "2N/2B", 4: "3N/3B", 5: "4B+"}
        rng = np.random.default_rng(42)
        for on, grp in sub_opt.groupby("optical_numeric"):
            j = rng.uniform(-0.12, 0.12, len(grp))
            ax.scatter(grp["optical_numeric"].values + j, grp["log_pfu"].values,
                       color=A4, s=55, alpha=0.8, edgecolors=BG, lw=0.4, zorder=3)
        positions = sorted(sub_opt["optical_numeric"].unique())
        groups    = [sub_opt[sub_opt["optical_numeric"] == p]["log_pfu"].values
                     for p in positions]
        ax.boxplot(groups, positions=positions, widths=0.3, patch_artist=True,
                   zorder=4,
                   medianprops=dict(color=A2, lw=2),
                   boxprops=dict(facecolor=PANEL, edgecolor=A1),
                   whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                   flierprops=dict(marker="o", color=A2, ms=4))
        ax.set_xticks(positions)
        ax.set_xticklabels([opt_labels.get(p, str(p)) for p in positions],
                           rotation=15, fontsize=8)
        r = stats.get("corr_optical_pfu")
        p = stats.get("p_optical_pfu")
        if r is not None:
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
            ax.text(0.05, 0.95, f"r = {r:.3f}  {sig}\np = {p:.4f}  n={len(sub_opt)}",
                    transform=ax.transAxes, color=A2, fontsize=9, va="top",
                    bbox=dict(facecolor=PANEL, edgecolor=GRID,
                              boxstyle="round,pad=0.3", alpha=0.9))
    else:
        ax.text(0.5, 0.5, f"Insufficient data\n(n={len(sub_opt)})",
                ha="center", va="center", color=MUTED,
                transform=ax.transAxes, fontsize=10)
    ax.set_xlabel("H-alpha Optical Class")
    ax.set_ylabel("log10 Peak Flux (pfu)")
    sax(ax, "Optical Class vs Peak Flux  (complexity proxy)")

    # C4: Flare Lead Time vs pfu
    ax = axes[1, 1]
    scatter_fit(ax, "flare_lead_min", "log_pfu", A2,
                "SEP onset - Flare max (min)  [+: SEP after flare]",
                "Flare Lead Time vs Peak Flux",
                "corr_lead_pfu", "p_lead_pfu")
    ax.axvline(0, color=MUTED, ls="--", lw=0.9, zorder=2)

    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[Plot2]  -> {path}")


# ============================================================
# 6.  MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="NOAA SPE.txt complete analyzer v3")
    parser.add_argument("--input",        required=True)
    parser.add_argument("--output_dir",   default="./spe_out_1924")
    parser.add_argument("--start_year",   type=int, default=None)
    parser.add_argument("--end_year",     type=int, default=None)
    parser.add_argument("--buffer_hours", type=int, default=DEFAULT_BUFFER_HOURS)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    s = str(args.start_year)[2:] if args.start_year else ""
    e = str(args.end_year)[2:]   if args.end_year   else ""
    tag = (s + e) or "all"

    events, skipped = parse_spe_txt(
        Path(args.input), args.start_year, args.end_year, args.buffer_hours)
    print(f"\n[Parse]  valid={len(events)}  skip={skipped}")
    if not events:
        print("No events found."); return

    # 파싱 검증 샘플
    print("\n[Sample parse check]")
    for ev in events[:3]:
        print(f"  {ev['onset']}  pfu={ev['peak_pfu']:>5}  "
              f"xray={ev['xray_class']}  opt={ev['optical_class']}  "
              f"loc={ev['ar_location']}  lead={ev['flare_lead_min']}min")

    # 저장
    ev_path = out / f"known_events_{tag}.json"
    with open(ev_path, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2, ensure_ascii=False)
    print(f"[Events] -> {ev_path}")

    stats = compute_stats(events, args.start_year, args.end_year, args.buffer_hours)
    st_path = out / f"spe_stats_{tag}.json"
    with open(st_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[Stats]  -> {st_path}")

    save_table(events, out / f"spe_table_{tag}.csv")

    # 콘솔 요약
    print(f"\n{'='*56}")
    print(f"  SPE Analysis  ({stats['analysis_period']})")
    print(f"{'='*56}")
    print(f"  Total events        : {stats['n_events']}")
    print(f"  Events / year       : {stats['events_per_year']}")
    print(f"  Rise time mean/med  : {stats['duration_hr_mean']}h / {stats['duration_hr_median']}h")
    print(f"  Peak pfu mean/max   : {stats['pfu_mean']} / {stats['pfu_max']} pfu")
    print(f"  Scale  S1/S2/S3+    : {stats['n_S1']} / {stats['n_S2']} / {stats['n_S3plus']}")
    print(f"  Flare lead med      : {stats['flare_lead_median']} min  (n={stats['n_with_flare']})")
    print(f"  AR longitude mean   : W{stats['ar_lon_mean']} deg  (n={stats['n_with_location']})")
    print(f"  --- Correlations (Pearson r) ---")
    def fmt(k_r, k_p):
        r, p = stats.get(k_r), stats.get(k_p)
        if r is None: return "N/A (insufficient data)"
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        return f"r={r:.3f}  p={p:.4f}  {sig}"
    print(f"  X-ray   <-> pfu     : {fmt('corr_xray_pfu','p_xray_pfu')}")
    print(f"  Longitude <-> pfu   : {fmt('corr_lon_pfu','p_lon_pfu')}")
    print(f"  Optical <-> pfu     : {fmt('corr_optical_pfu','p_optical_pfu')}")
    print(f"  Lead time <-> pfu   : {fmt('corr_lead_pfu','p_lead_pfu')}")
    print(f"  GT coverage         : {stats['gt_event_pct']}%  <- target ALERT upper bound")
    print(f"{'='*56}")

    plot_overview(events, stats, out / f"spe_analysis_{tag}.png")
    plot_correlation(events, stats, out / f"spe_correlation_{tag}.png")

    print(f"\n[Usage]\n  python evaluate.py \\\n"
          f"      --states  bl_c2_states.csv \\\n"
          f"      --metrics bl_c2_metrics.json \\\n"
          f"      --events  {ev_path}")


if __name__ == "__main__":
    main()