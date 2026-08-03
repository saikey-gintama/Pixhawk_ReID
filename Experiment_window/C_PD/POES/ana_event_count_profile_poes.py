"""
ana_event_count_profile_poes.py
================================
SPE / eSPE 카탈로그 기반 채널별 count 시계열 진단 — POES 판.

KSEM ana_event_count_profile.py 의 POES 이식판.
플롯/통계 로직은 동일, 입력 어댑터만 교체.

[원본 KSEM → POES 변경점]
  1. IO 루트    : KSEM_IO_DIR(KSEM_count/) → POES_IO_DIR(POES/), MetOp03 캐시 사용
  2. io 모듈    : ksem_io → poes_metop03_io (as poes_io)
  3. ERA        : 2019-2024 → 2019-2025
  4. 채널구조   : (PD, side, logic) 3중 → (species, direction, energy) 3중 MultiIndex
  5. 그룹분류   : GROUP_LOGICS(logic 문자열) → GROUP_CHANNELS(채널 튜플, channel_stats.csv 기준)
  6. 패널행     : 4행(A,B,C1,C2) → 3행(A,B,C)
  7. 색상/LS    : PALETTE[PD]/LS_MAP[A|B] → SPECIES_PALETTE[species]/DIR_LS[direction]
  8. epoch 그리드: PD행×logic열 → 그룹행×에너지쌍열, tel0=solid/tel90=dashed

[기능 1] 이벤트별 패널
  - onset 앞 24h ~ peak 뒤 72h(SPE) / 48h(eSPE)
  - 3행(그룹 A/B/C), 각 행 = 그룹 내 전 채널 겹침
  - species=색상, tel0=실선/tel90=점선

[기능 2] Superposed epoch (peak-normalized)
  - 3행(그룹) × 6열(에너지쌍) 그리드
  - 같은 에너지 레벨의 tel0/tel90을 한 셀에 오버레이 (KSEM의 A/B side와 동일 구조)
  - median + IQR 음영, onset(median) 세로선, peak 세로선

[기능 2b] 전 이벤트 raw 겹치기
[기능 3]  이벤트/quiet count 통계 + 고정임계 C 후보 CSV

파일 위치: C_PD/POES/ana_event_count_profile_poes.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── 경로 설정 (이 파일을 C_PD/POES/ 에 두는 전제) ─────────────────
POES_IO_DIR       = Path(__file__).parent          # C_PD/POES/
METOP_DIR         = POES_IO_DIR / "MetOp03_count"

COUNT_PARQUET_DIR = METOP_DIR / "poes_metop03_cache_parquet"

NOAA_IO_DIR           = POES_IO_DIR.parent / "NOAA_GOES"
NOAA_SPE_CATALOG_DIR  = NOAA_IO_DIR / "noaa_goes_spe_cache_parquet"
SWPC_IO_DIR           = POES_IO_DIR.parent / "SWPC_Alert"
SWPC_ESPE_CATALOG_DIR = SWPC_IO_DIR / "swpc_espe_cache_parquet"

POES_ERA = ("2019-01-01", "2025-12-31")

OUT_DIR = POES_IO_DIR / "ana_output"

# ── io 모듈 import ─────────────────────────────────────────────────
for _p in (METOP_DIR, NOAA_IO_DIR, SWPC_IO_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import poes_metop03_io as poes_io      # noqa: E402
import noaa_goes_spe_io as spe_io     # noqa: E402
try:
    import swpc_alert_espe_io as espe_io  # noqa: E402
except ImportError:
    espe_io = None

# ── 파라미터 ───────────────────────────────────────────────────────
RESAMPLE_FREQ         = "15min"
PRE_ONSET_H           = 24
POST_PEAK_NOAA_SPE_H  = 72
POST_PEAK_SWPC_ESPE_H = 48
EPOCH_PRE_H           = 24
EPOCH_POST_H          = 72
EPOCH_BIN_MIN         = 60
NOAA_SPE_MIN_PFU      = 10
SWPC_ESPE_MIN_PFU     = 1000
STAT_NARROW_POST_H    = 12
STAT_PEAK_WIN_H       = 6
STAT_TARGET_FPR       = 0.05

# ── 채널 그룹 (channel_stats.csv 기준, zero_frac/median 분류) ──────
#   A: proton-trigger 후보 — quiet 거의 0, zero_frac 높음
#   B: 중간 배경
#   C: always-on — median 높음, zero_frac ≈ 0
GROUP_CHANNELS: dict[str, list[tuple]] = {
    "A": [
        ("pro", "tel0",  "p5"), ("pro", "tel0",  "p4"),
        ("pro", "tel90", "p5"), ("pro", "tel90", "p4"),
        ("omni", "-", "p6"), ("omni", "-", "p7"),
        ("omni", "-", "p8"), ("omni", "-", "p9"),
    ],
    "B": [
        ("pro", "tel0",  "p6"), ("pro", "tel90", "p6"),
        ("pro", "tel0",  "p3"), ("pro", "tel90", "p3"),
    ],
    "C": [
        ("ele", "tel0",  "e1"), ("ele", "tel0",  "e2"), ("ele", "tel0",  "e3"),
        ("ele", "tel90", "e1"), ("ele", "tel90", "e2"), ("ele", "tel90", "e3"),
        ("pro", "tel0",  "p1"), ("pro", "tel0",  "p2"),
        ("pro", "tel90", "p1"), ("pro", "tel90", "p2"),
    ],
}

# ── 이벤트 패널 행 (3행) ───────────────────────────────────────────
PANEL_ROWS = [
    ("A", GROUP_CHANNELS["A"],
     "Group A: pro_p4/p5, omni_p6~p9  (low quiet bg, high zero_frac)"),
    ("B", GROUP_CHANNELS["B"],
     "Group B: pro_p3/p6  (intermediate bg)"),
    ("C", GROUP_CHANNELS["C"],
     "Group C: ele_e1~e3, pro_p1/p2  (always-on, active bg)"),
]

# ── Superposed epoch 그리드 ────────────────────────────────────────
# nrows = 그룹 수(3), ncols = 그룹별 에너지쌍 최대 수(6)
# 쌍 = (tel0_tpl, tel90_tpl) 또는 (omni_tpl,) 단일
EPOCH_PANELS = [
    ("A", [
        (("pro", "tel0", "p5"), ("pro", "tel90", "p5")),
        (("pro", "tel0", "p4"), ("pro", "tel90", "p4")),
        (("omni", "-", "p6"),),
        (("omni", "-", "p7"),),
        (("omni", "-", "p8"),),
        (("omni", "-", "p9"),),
    ], "Group A"),
    ("B", [
        (("pro", "tel0", "p6"), ("pro", "tel90", "p6")),
        (("pro", "tel0", "p3"), ("pro", "tel90", "p3")),
    ], "Group B"),
    ("C", [
        (("ele", "tel0", "e1"), ("ele", "tel90", "e1")),
        (("ele", "tel0", "e2"), ("ele", "tel90", "e2")),
        (("ele", "tel0", "e3"), ("ele", "tel90", "e3")),
        (("pro", "tel0", "p1"), ("pro", "tel90", "p1")),
        (("pro", "tel0", "p2"), ("pro", "tel90", "p2")),
    ], "Group C"),
]
_EPOCH_NCOLS = max(len(p[1]) for p in EPOCH_PANELS)  # 6

# ── 색상 / 라인스타일 ──────────────────────────────────────────────
SPECIES_PALETTE = {"pro": "#c0392b", "omni": "#e67e22", "ele": "#2980b9"}
DIR_LS          = {"tel0": "-", "tel90": "--", "-": "-"}


def _ch_label(tpl: tuple) -> str:
    """('pro','tel0','p5') → 'pro_tel0_p5' / ('omni','-','p6') → 'omni_p6'."""
    return poes_io.tuple_to_fname(tpl)


# ── 카탈로그 로드 ─────────────────────────────────────────────────
def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is None:
        df.index = pd.to_datetime(df.index, utc=True)
    return df


def _filter_era(df: pd.DataFrame, io_mod) -> pd.DataFrame:
    if hasattr(io_mod, "filter_by_date"):
        return io_mod.filter_by_date(df, *POES_ERA)
    lo = pd.Timestamp(POES_ERA[0], tz="UTC")
    hi = pd.Timestamp(POES_ERA[1], tz="UTC")
    return df[(df.index >= lo) & (df.index <= hi)]


def load_noaa_spe() -> pd.DataFrame:
    df, meta = spe_io.load(str(NOAA_SPE_CATALOG_DIR))
    df = _ensure_utc_index(df)
    df = _filter_era(df, spe_io)
    df = df[df["max_pfu"] >= NOAA_SPE_MIN_PFU].copy()
    df["end_time"] = df["max_time"] + pd.Timedelta(hours=POST_PEAK_NOAA_SPE_H)
    print(f"[catalog] NOAA SPE {POES_ERA[0][:4]}~{POES_ERA[1][:4]}: {len(df)}개  "
          f"(≥{NOAA_SPE_MIN_PFU} pfu)  meta={'ok' if meta else 'none'}")
    return df


def load_swpc_espe() -> pd.DataFrame:
    if espe_io is None:
        print("[catalog] swpc_alert_espe_io 없음 — eSPE 분석 스킵")
        return pd.DataFrame()
    df, meta = espe_io.load(str(SWPC_ESPE_CATALOG_DIR))
    df = _ensure_utc_index(df)
    df = _filter_era(df, espe_io)
    df = df[df["max_pfu"] >= SWPC_ESPE_MIN_PFU].copy()
    df["end_time"] = df["max_time"] + pd.Timedelta(hours=POST_PEAK_SWPC_ESPE_H)
    print(f"[catalog] SWPC eSPE: {len(df)}개  (≥{SWPC_ESPE_MIN_PFU} pfu)  "
          f"meta={'ok' if meta else 'none'}")
    return df


# ── count 로드 ────────────────────────────────────────────────────
def load_count() -> pd.DataFrame:
    print(f"[data] POES count 로드: {COUNT_PARQUET_DIR}")
    df, _ = poes_io.load(COUNT_PARQUET_DIR)
    if not df.empty and df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df.resample(RESAMPLE_FREQ).mean()
    print(f"[data] count shape: {df.shape},  "
          f"기간: {df.index[0].date()} ~ {df.index[-1].date()}")
    return df


# ─────────────────────────────────────────────────────────────────
# 기능 1: 이벤트별 패널 (그룹 A/B/C 3행)
# ─────────────────────────────────────────────────────────────────
def plot_event_panels(df_count: pd.DataFrame,
                      catalog: pd.DataFrame,
                      mode: str,
                      out_prefix: str,
                      panel_subdir: str):
    """이벤트별 count 시계열 패널 — PANEL_ROWS 3행.
    각 행 = 그룹 내 전 채널 겹침. species=색상, tel0=solid/tel90=dashed."""
    panel_dir = OUT_DIR / panel_subdir
    panel_dir.mkdir(parents=True, exist_ok=True)

    label  = "NOAA-SPE" if mode == "noaa_spe" else "SWPC-eSPE"
    post_h = POST_PEAK_NOAA_SPE_H if mode == "noaa_spe" else POST_PEAK_SWPC_ESPE_H
    n_rows = len(PANEL_ROWS)

    for ei, (onset, row) in enumerate(catalog.iterrows()):
        peak     = row["max_time"]
        end_plot = peak + pd.Timedelta(hours=post_h)
        t0       = onset - pd.Timedelta(hours=PRE_ONSET_H)
        t1       = end_plot

        if t0 > df_count.index[-1] or t1 < df_count.index[0]:
            print(f"  [{label} #{ei+1}] 데이터 범위 밖, 스킵")
            continue

        fig, axes = plt.subplots(n_rows, 1,
                                  figsize=(14, 2.8 * n_rows + 0.6),
                                  sharex=True)

        pfu_str   = f"{row['max_pfu']:.0f} pfu"
        flare_str = ""
        if mode == "noaa_spe" and row.get("flare_class", ""):
            flare_str = f"  |  Flare: {row['flare_class']}"
        title = (f"{label} Event #{ei+1}   "
                 f"Onset: {onset.strftime('%Y-%m-%d %H:%M')}   "
                 f"Peak: {peak.strftime('%Y-%m-%d %H:%M')}   "
                 f"Max: {pfu_str}{flare_str}\n"
                 f"End(proxy): peak+{post_h}h  |  "
                 f"pro=red / omni=orange / ele=blue / tel0=solid / tel90=dashed")
        fig.suptitle(title, fontsize=9, y=1.005)

        for ri, (rkey, ch_list, rlabel) in enumerate(PANEL_ROWS):
            ax = axes[ri]
            any_plotted = False

            for ch_tpl in ch_list:
                try:
                    s = df_count[ch_tpl].loc[t0:t1]
                except KeyError:
                    continue
                if s.dropna().empty:
                    continue
                color = SPECIES_PALETTE[ch_tpl[0]]
                ls    = DIR_LS[ch_tpl[1]]
                ax.plot(s.index, s.values,
                        color=color, ls=ls, lw=1.1, alpha=0.8,
                        label=_ch_label(ch_tpl))
                any_plotted = True

            ax.axvline(onset,    ls="--", color="orange", lw=1.2, alpha=0.9, label="onset")
            ax.axvline(peak,     ls=":",  color="red",    lw=1.2, alpha=0.9, label="peak")
            ax.axvline(end_plot, ls="--", color="gray",   lw=0.9, alpha=0.6,
                       label=f"end(+{post_h}h)")
            ax.set_ylim(bottom=0)
            ax.set_ylabel(f"Count\n{rlabel}", fontsize=7.5)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=5, ncol=8, loc="upper left", framealpha=0.7)

            if not any_plotted:
                ax.text(0.5, 0.5, "no data",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=9, color="gray")

        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
        axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
        fig.tight_layout()

        fname = panel_dir / f"{out_prefix}_event_{ei+1:03d}_panel.png"
        fig.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  [{label} #{ei+1}] {onset.date()} / {row['max_pfu']:.0f} pfu"
              f"  → {fname.name}")

    print(f"[panel] {label} {len(catalog)}개 저장 → {panel_dir}")


# ─────────────────────────────────────────────────────────────────
# 기능 2: Superposed Epoch
# ─────────────────────────────────────────────────────────────────
def build_epoch_matrix(cnt: pd.Series,
                        catalog: pd.DataFrame,
                        h_axis: np.ndarray) -> pd.DataFrame:
    """peak 기준 epoch 행렬. 값 = count / count(at catalog peak)."""
    rows = []
    for onset, row in catalog.iterrows():
        peak   = row["max_time"]
        idx_pk = cnt.index.get_indexer([peak], method="nearest")[0]
        pk_cnt = cnt.iloc[idx_pk] if 0 <= idx_pk < len(cnt) else np.nan
        if not (np.isfinite(pk_cnt) and pk_cnt > 0):
            continue
        vals = []
        for dh in h_axis:
            t   = peak + pd.Timedelta(hours=float(dh))
            idx = cnt.index.get_indexer([t], method="nearest")[0]
            vals.append(cnt.iloc[idx] / pk_cnt if 0 <= idx < len(cnt) else np.nan)
        rows.append(vals)
    return pd.DataFrame(rows, columns=h_axis) if rows else pd.DataFrame()


def plot_superposed_epoch(df_count: pd.DataFrame,
                           catalog: pd.DataFrame,
                           mode: str,
                           out_prefix: str):
    """3행(그룹 A/B/C) × 6열(에너지쌍) superposed epoch.
    tel0=solid/tel90=dashed, IQR shade=tel0 기준 (KSEM A side 대응).
    출력: OUT_DIR/{out_prefix}_superposed_epoch.png"""
    label     = "NOAA-SPE" if mode == "noaa_spe" else "SWPC-eSPE"
    post_h    = POST_PEAK_NOAA_SPE_H if mode == "noaa_spe" else POST_PEAK_SWPC_ESPE_H
    epoch_post = min(EPOCH_POST_H, post_h)

    h_axis = np.arange(-EPOCH_PRE_H,
                        epoch_post + EPOCH_BIN_MIN / 60,
                        EPOCH_BIN_MIN / 60)

    nrows = len(EPOCH_PANELS)
    ncols = _EPOCH_NCOLS
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.8 * ncols, 3.5 * nrows),
                              squeeze=False)
    fig.suptitle(
        f"{label} Superposed Epoch  (peak-normalized, {EPOCH_BIN_MIN}-min bin)\n"
        "line=median  shade=IQR  solid=tel0  dashed=tel90",
        fontsize=11)

    # onset 중앙값 [h before peak] — 카탈로그 공통
    med_onset_h = ((catalog["max_time"] - catalog.index)
                   .dt.total_seconds() / 3600).median()

    for ri, (grp_key, ch_pairs, grp_label) in enumerate(EPOCH_PANELS):
        for ci in range(ncols):
            ax = axes[ri][ci]
            if ci >= len(ch_pairs):
                ax.set_visible(False)
                continue

            ch_pair   = ch_pairs[ci]
            n_plotted = 0
            shade_done = False   # IQR shade는 첫 번째 채널만 (KSEM A-side 동일)

            for ch_tpl in ch_pair:
                try:
                    s = df_count[ch_tpl].dropna()
                except KeyError:
                    continue
                mat = build_epoch_matrix(s, catalog, h_axis)
                if mat.empty:
                    continue
                n   = len(mat)
                h   = mat.columns.values.astype(float)
                med = mat.median(axis=0).values
                q25 = mat.quantile(0.25, axis=0).values
                q75 = mat.quantile(0.75, axis=0).values
                c   = SPECIES_PALETTE[ch_tpl[0]]
                ls  = DIR_LS[ch_tpl[1]]

                ax.plot(h, med, color=c, ls=ls, lw=1.6,
                        label=f"{_ch_label(ch_tpl)} (n={n})")
                if not shade_done:
                    ax.fill_between(h, q25, q75, color=c, alpha=0.15)
                    shade_done = True
                n_plotted += 1

            ax.axvline(0,           ls="--", color="red",    lw=1,   label="Peak")
            ax.axvline(-med_onset_h, ls=":",  color="orange", lw=0.9, alpha=0.7,
                       label="onset(median)")
            ax.axhline(1, ls=":", color="gray", lw=0.8)
            ax.set_xlim(-EPOCH_PRE_H, epoch_post)
            ax.set_xlabel("Time from peak [h]", fontsize=8)
            ax.set_ylabel("Normalized Count", fontsize=8)

            ch0 = ch_pair[0]
            ax.set_title(f"{grp_label} / {ch0[0]}_{ch0[2]}", fontsize=9)
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=7)

            if n_plotted == 0:
                ax.text(0.5, 0.5, "no data",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=9, color="gray")

    fig.tight_layout()
    out_path = OUT_DIR / f"{out_prefix}_superposed_epoch.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[epoch] {label} superposed epoch → {out_path.name}")


# ─────────────────────────────────────────────────────────────────
# 기능 2b: 전 이벤트 raw 겹치기
# ─────────────────────────────────────────────────────────────────
def plot_overplot_raw(df_count: pd.DataFrame,
                      catalog: pd.DataFrame,
                      mode: str,
                      out_prefix: str):
    """peak 기준 count 원값 전 이벤트 겹치기.
    3행(그룹) × 6열(에너지쌍). thick=median, thin=개별 이벤트.
    출력: OUT_DIR/{out_prefix}_overplot_raw.png"""
    label     = "NOAA-SPE" if mode == "noaa_spe" else "SWPC-eSPE"
    post_h    = POST_PEAK_NOAA_SPE_H if mode == "noaa_spe" else POST_PEAK_SWPC_ESPE_H
    epoch_post = min(EPOCH_POST_H, post_h)

    h_axis = np.arange(-EPOCH_PRE_H,
                        epoch_post + EPOCH_BIN_MIN / 60,
                        EPOCH_BIN_MIN / 60)

    med_rise = ((catalog["max_time"] - catalog.index)
                .dt.total_seconds().median() / 3600)

    nrows = len(EPOCH_PANELS)
    ncols = _EPOCH_NCOLS
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.8 * ncols, 3.5 * nrows),
                              squeeze=False)
    fig.suptitle(
        f"{label} all-event overplot  (peak-aligned, raw count)\n"
        "thick=median  thin=individual events  solid=tel0  dashed=tel90",
        fontsize=11)

    for ri, (grp_key, ch_pairs, grp_label) in enumerate(EPOCH_PANELS):
        for ci in range(ncols):
            ax = axes[ri][ci]
            if ci >= len(ch_pairs):
                ax.set_visible(False)
                continue

            ch_pair = ch_pairs[ci]

            for ch_tpl in ch_pair:
                try:
                    s = df_count[ch_tpl].dropna()
                except KeyError:
                    continue

                event_traces = []
                for onset, row in catalog.iterrows():
                    peak = row["max_time"]
                    vals = []
                    for dh in h_axis:
                        t   = peak + pd.Timedelta(hours=float(dh))
                        idx = s.index.get_indexer([t], method="nearest")[0]
                        vals.append(s.iloc[idx] if 0 <= idx < len(s) else np.nan)
                    event_traces.append(vals)

                mat = np.array(event_traces, dtype=float)
                if mat.shape[0] == 0:
                    continue

                c  = SPECIES_PALETTE[ch_tpl[0]]
                ls = DIR_LS[ch_tpl[1]]
                for trace in mat:
                    ax.plot(h_axis, trace, color=c, ls=ls, lw=0.5, alpha=0.2)
                med = np.nanmedian(mat, axis=0)
                ax.plot(h_axis, med, color=c, ls=ls, lw=2.2, alpha=0.95,
                        label=f"{_ch_label(ch_tpl)} med (n={mat.shape[0]})")

            ax.axvline(0,        ls="--", color="red",    lw=1.2, label="Peak")
            ax.axvline(-med_rise, ls=":",  color="orange", lw=1,   alpha=0.8,
                       label="onset(median)")
            ax.set_xlim(-EPOCH_PRE_H, epoch_post)
            ax.set_ylim(bottom=0)
            ax.set_xlabel("Time from peak [h]", fontsize=8)
            ax.set_ylabel("Count (raw)", fontsize=8)
            ch0 = ch_pair[0]
            ax.set_title(f"{grp_label} / {ch0[0]}_{ch0[2]}", fontsize=9)
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=7)

    fig.tight_layout()
    out_path = OUT_DIR / f"{out_prefix}_overplot_raw.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[overplot] {label} raw overplot → {out_path.name}")


# ─────────────────────────────────────────────────────────────────
# 기능 3: 이벤트 vs 비이벤트 count 통계 + 고정임계 C 후보 (CSV)
# ─────────────────────────────────────────────────────────────────
def _build_event_mask(cnt_index: pd.DatetimeIndex,
                      catalog: pd.DataFrame,
                      post_h: float,
                      pre_h: float = 0.0) -> np.ndarray:
    mask = np.zeros(len(cnt_index), dtype=bool)
    for onset, row in catalog.iterrows():
        peak  = row["max_time"]
        start = onset - pd.Timedelta(hours=pre_h)
        end   = peak  + pd.Timedelta(hours=post_h)
        mask |= (cnt_index >= start) & (cnt_index <= end)
    return mask


def _event_peak_levels(cnt: pd.Series,
                       catalog: pd.DataFrame,
                       win_h: float) -> np.ndarray:
    peaks = []
    for onset, row in catalog.iterrows():
        peak = row["max_time"]
        seg  = cnt.loc[peak - pd.Timedelta(hours=win_h):
                       peak + pd.Timedelta(hours=win_h)]
        v = seg.max()
        if np.isfinite(v):
            peaks.append(float(v))
    return np.array(peaks, dtype=float)


def _roc_youden_threshold(event_vals: np.ndarray,
                          quiet_vals: np.ndarray) -> tuple:
    ev = event_vals[np.isfinite(event_vals)]
    qt = quiet_vals[np.isfinite(quiet_vals)]
    if len(ev) < 5 or len(qt) < 5:
        return (np.nan, np.nan, np.nan, np.nan)
    cand = np.unique(np.concatenate([ev, qt]))
    if len(cand) > 2000:
        cand = np.quantile(cand, np.linspace(0, 1, 2000))
    best = (np.nan, -np.inf, np.nan, np.nan)
    n_ev, n_qt = len(ev), len(qt)
    for thr in cand:
        tpr = (ev >= thr).sum() / n_ev
        fpr = (qt >= thr).sum() / n_qt
        J   = tpr - fpr
        if J > best[1]:
            best = (float(thr), float(J), float(tpr), float(fpr))
    return best


def _fpr_constrained_threshold(event_vals: np.ndarray,
                               quiet_vals: np.ndarray,
                               target_fpr: float) -> tuple:
    ev = event_vals[np.isfinite(event_vals)]
    qt = quiet_vals[np.isfinite(quiet_vals)]
    if len(ev) < 5 or len(qt) < 5:
        return (np.nan, np.nan, np.nan)
    cand = np.unique(np.concatenate([ev, qt]))
    if len(cand) > 4000:
        cand = np.quantile(cand, np.linspace(0, 1, 4000))
    n_ev, n_qt = len(ev), len(qt)
    best = (np.nan, -np.inf, np.nan)
    for thr in cand:
        fpr = (qt >= thr).sum() / n_qt
        if fpr <= target_fpr:
            tpr = (ev >= thr).sum() / n_ev
            if tpr > best[1]:
                best = (float(thr), float(tpr), float(fpr))
    if not np.isfinite(best[1]):
        return (np.nan, np.nan, np.nan)
    return best


def save_event_count_stats(df_count: pd.DataFrame,
                           catalog: pd.DataFrame,
                           mode: str,
                           out_prefix: str):
    """채널별 이벤트/quiet count 통계 + C 후보(분위수, ROC-Youden) → CSV."""
    label  = "NOAA-SPE" if mode == "noaa_spe" else "SWPC-eSPE"
    post_h = POST_PEAK_NOAA_SPE_H if mode == "noaa_spe" else POST_PEAK_SWPC_ESPE_H

    all_channels = (GROUP_CHANNELS["A"] + GROUP_CHANNELS["B"] + GROUP_CHANNELS["C"])
    rows = []

    for ch_tpl in all_channels:
        try:
            cnt = df_count[ch_tpl].dropna()
        except KeyError:
            continue
        if len(cnt) < 100:
            continue

        wide_mask   = _build_event_mask(cnt.index, catalog, post_h)
        narrow_mask = _build_event_mask(cnt.index, catalog, STAT_NARROW_POST_H)
        qt_vals     = cnt.values[~wide_mask]
        ev_wide     = cnt.values[wide_mask]
        ev_narrow   = cnt.values[narrow_mask]
        if len(ev_narrow) < 5 or len(qt_vals) < 5:
            continue
        epeak = _event_peak_levels(cnt, catalog, STAT_PEAK_WIN_H)

        qp = {f"quiet_p{p}": float(np.nanpercentile(qt_vals, p))
              for p in (50, 90, 95, 99, 99.9)}
        emed_wide   = float(np.nanmedian(ev_wide))
        emed_narrow = float(np.nanmedian(ev_narrow))
        epeak_med   = float(np.nanmedian(epeak)) if len(epeak) else np.nan
        epeak_p25   = float(np.nanpercentile(epeak, 25)) if len(epeak) else np.nan
        qmed        = float(np.nanmedian(qt_vals))
        ratio_narrow = emed_narrow / qmed if qmed > 0 else np.nan
        ratio_epeak  = epeak_med  / qmed if qmed > 0 else np.nan

        c_roc, j_roc, tpr_roc, fpr_roc = _roc_youden_threshold(ev_narrow, qt_vals)
        c_fpr, tpr_fpr, fpr_act = _fpr_constrained_threshold(
            ev_narrow, qt_vals, STAT_TARGET_FPR)

        species, direction, energy = ch_tpl
        grp = ("A" if ch_tpl in GROUP_CHANNELS["A"]
               else "B" if ch_tpl in GROUP_CHANNELS["B"] else "C")

        rows.append({
            "channel": _ch_label(ch_tpl),
            "species": species, "direction": direction, "energy": energy,
            "group": grp,
            "n_quiet_pts": int(len(qt_vals)),
            "n_event_narrow": int(len(ev_narrow)),
            "n_event_peaks": int(len(epeak)),
            "quiet_median": round(qmed, 4),
            "event_median_wide": round(emed_wide, 4),
            "event_median_narrow": round(emed_narrow, 4),
            "event_peak_median": round(epeak_med, 4) if np.isfinite(epeak_med) else np.nan,
            "event_peak_p25": round(epeak_p25, 4) if np.isfinite(epeak_p25) else np.nan,
            "ratio_narrow": round(ratio_narrow, 3) if np.isfinite(ratio_narrow) else np.nan,
            "ratio_peak": round(ratio_epeak, 3) if np.isfinite(ratio_epeak) else np.nan,
            **{k: round(v, 4) for k, v in qp.items()},
            "C_roc_youden": round(c_roc, 4) if np.isfinite(c_roc) else np.nan,
            "roc_J": round(j_roc, 4) if np.isfinite(j_roc) else np.nan,
            "roc_tpr": round(tpr_roc, 4) if np.isfinite(tpr_roc) else np.nan,
            "roc_fpr": round(fpr_roc, 4) if np.isfinite(fpr_roc) else np.nan,
            f"C_fpr{STAT_TARGET_FPR}": round(c_fpr, 4) if np.isfinite(c_fpr) else np.nan,
            "fpr_tpr": round(tpr_fpr, 4) if np.isfinite(tpr_fpr) else np.nan,
            "fpr_actual": round(fpr_act, 4) if np.isfinite(fpr_act) else np.nan,
        })

    df_stats = pd.DataFrame(rows)
    out_csv  = OUT_DIR / f"{out_prefix}_event_count_stats.csv"
    df_stats.to_csv(out_csv, index=False)
    print(f"[stats] {label} channel stats → {out_csv.name}  ({len(df_stats)} channels)")
    if not df_stats.empty:
        cfpr_col = f"C_fpr{STAT_TARGET_FPR}"
        for grp in ("A", "B", "C"):
            g = df_stats[df_stats["group"] == grp].sort_values("roc_J", ascending=False)
            if g.empty:
                continue
            top = g.head(3)[["channel", "ratio_peak", "quiet_p95",
                             "event_peak_median", "C_roc_youden", "roc_J",
                             cfpr_col, "fpr_tpr"]]
            print(f"[stats] {label} Group {grp} — roc_J 상위 3:")
            print(top.to_string(index=False))
    return df_stats


# ─────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────
def main():
    global COUNT_PARQUET_DIR, OUT_DIR

    ap = argparse.ArgumentParser()
    ap.add_argument("--count", required=True, help="POES count parquet 디렉토리")
    ap.add_argument("--out",   required=True, help="출력 디렉토리")
    args = ap.parse_args()

    COUNT_PARQUET_DIR = Path(args.count)
    OUT_DIR           = Path(args.out)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    noaa_spe_catalog  = load_noaa_spe()
    swpc_espe_catalog = load_swpc_espe()
    df_count          = load_count()

    print("\n=== NOAA SPE (양성자 카탈로그 기준) 전 채널 분석 ===")
    plot_event_panels(df_count, noaa_spe_catalog,
                      mode="noaa_spe", out_prefix="noaa_spe",
                      panel_subdir="panels_noaa_spe")
    plot_superposed_epoch(df_count, noaa_spe_catalog,
                          mode="noaa_spe", out_prefix="noaa_spe")
    plot_overplot_raw(df_count, noaa_spe_catalog,
                      mode="noaa_spe", out_prefix="noaa_spe")
    save_event_count_stats(df_count, noaa_spe_catalog,
                           mode="noaa_spe", out_prefix="noaa_spe")

    if not swpc_espe_catalog.empty:
        print("\n=== SWPC eSPE (전자 카탈로그 기준) 전 채널 분석 ===")
        plot_event_panels(df_count, swpc_espe_catalog,
                          mode="swpc_espe", out_prefix="swpc_espe",
                          panel_subdir="panels_swpc_espe")
        plot_superposed_epoch(df_count, swpc_espe_catalog,
                              mode="swpc_espe", out_prefix="swpc_espe")
        plot_overplot_raw(df_count, swpc_espe_catalog,
                          mode="swpc_espe", out_prefix="swpc_espe")
        save_event_count_stats(df_count, swpc_espe_catalog,
                               mode="swpc_espe", out_prefix="swpc_espe")

    print(f"\n[완료] 출력: {OUT_DIR}")
    print("  panels_noaa_spe/          : NOAA SPE 이벤트별 패널 (3행)")
    print("  noaa_spe_superposed_epoch.png / noaa_spe_overplot_raw.png")
    print("  noaa_spe_event_count_stats.csv  ← BL-C1 C 근거")


if __name__ == "__main__":
    main()
