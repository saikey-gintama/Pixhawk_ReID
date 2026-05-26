"""
ana4_count_response_profile.py
================================
이벤트별 Count 응답 프로파일 분석

SPE 기준:
  NOAA >10 MeV / 10 pfu 채널 없음.
  proxy flux (E8+E9+E10 ΔE-적분) >= SPE_PROXY_THRESH 를 SPE로 정의.
  SPE_PROXY_THRESH는 proxy 분포 p95(~1.2e+04 cm-2 sr-1 s-1)에서 설정.

로직 포함 방침:
  O / OU / OUT 세 로직을 모두 분석에 포함.
  O 로직이 SPE에 반응하지 않는다면(ana2 결과: exceed_q99=0, ratio≈1),
  그 결과를 데이터로 직접 보여주는 것이 분석의 근거가 된다.

PD 방향각 비교:
  PD1(15°, polar) / PD2 / PD3(71°, equatorial) 세 쌍 비교.
  (PD1/PD2), (PD1/PD3), (PD2/PD3) 비율 히스토그램으로 이방성 확인.

출력:
  ana4_superposed_epoch.png  — peak 기준 중첩 epoch (median ± IQR)
  ana4_rise_time_cdf.png     — count > quiet p{SPE_COUNT_THRESH_PCTL} 첫 시점 → peak CDF
  ana4_onset_comparison.png  — flux onset vs count onset 지연 분포 (히스토그램)
  ana4_logic_order.png       — O / OU / OUT onset 지연 scatter (에너지 순서 확인)
  ana4_pd_direction.png      — PD1/PD2/PD3 쌍별 peak count 비율 히스토그램
  ana4_onset_delay_table.csv — (pd_key, side, logic) * {median, mean, std, n}
  ana4_onset_delay_raw.csv   — 이벤트별 원시 지연값

튜닝값 (이 파일 단독):
  PRE_H             : superposed epoch 앞 범위 [h].
  POST_H            : superposed epoch 뒤 범위 [h].
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import NamedTuple
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import ksem_flux_config as cfg
from ksem_flux_config import (
    COUNT_PARQUET_DIR, FLUX_PARQUET_DIR, OUTPUT_DIR,
    PROTON_CHANNELS,
    SPE_PROXY_CHANNELS, SPE_PROXY_THRESH, SPE_PROXY_LABEL,
    COUNT_PROTON_LOGICS, COUNT_PD_KEYS, COUNT_SIDES,
    MIN_SPE_DURATION_H, SPE_COUNT_THRESH_PCTL,
)
import ksem_io
import kma_ksem_flux_io

# ── ana4 단독 튜닝값 ──────────────────────────────────────────────
PRE_H             = 24   # [h] superposed epoch 앞 범위
POST_H            = 48   # [h] superposed epoch 뒤 범위


# ─────────────────────────────────────────────────────────────────
class SPEEvent(NamedTuple):
    onset:     pd.Timestamp
    peak:      pd.Timestamp
    end:       pd.Timestamp
    peak_flux: float


def make_proxy(df_flux: pd.DataFrame) -> pd.Series:
    proxy = pd.Series(0.0, index=df_flux.index)
    for ch in SPE_PROXY_CHANNELS:
        if ch in df_flux.columns:
            lo, hi = PROTON_CHANNELS[ch]
            proxy += df_flux[ch].fillna(0) * (hi - lo)
    proxy[proxy == 0] = np.nan
    return proxy


def detect_events(proxy: pd.Series) -> list[SPEEvent]:
    """
    proxy >= SPE_PROXY_THRESH 가 MIN_SPE_DURATION_H 이상 연속인 구간 탐지.
    ana2와 동일한 기준 사용 (MIN_SPE_DURATION_H는 config 공유값).
    """
    above  = proxy >= SPE_PROXY_THRESH
    evs, in_ev, onset = [], False, None
    for t, v in above.items():
        if v and not in_ev:
            in_ev, onset = True, t
        elif not v and in_ev:
            seg = proxy.loc[onset:t]
            if (t - onset).total_seconds() / 3600 >= MIN_SPE_DURATION_H:
                evs.append(SPEEvent(onset, seg.idxmax(), t, float(seg.max())))
            in_ev = False
    if in_ev and onset:
        seg = proxy.loc[onset:]
        if (seg.index[-1] - onset).total_seconds() / 3600 >= MIN_SPE_DURATION_H:
            evs.append(SPEEvent(onset, seg.idxmax(), seg.index[-1], float(seg.max())))
    print(f"[ana4] SPE events detected: {len(evs)}")
    return evs


def quiet_threshold(cnt: pd.Series, proxy: pd.Series,
                    pctl: int = SPE_COUNT_THRESH_PCTL) -> float:
    """
    COUNT_THRESH_PCTL 퍼센타일 threshold.
    proxy bool mask를 cnt 인덱스에 정렬하여 Quiet 구간 count를 추출.
    """
    is_spe_aligned = (proxy >= SPE_PROXY_THRESH).astype(float).reindex(
        cnt.index, method="nearest")
    if is_spe_aligned.isna().all():
        is_spe_aligned = (proxy >= SPE_PROXY_THRESH).astype(float).reindex(
            cnt.index, method="nearest", tolerance=None)
    q_vals = cnt[is_spe_aligned == 0].dropna()
    return float(np.nanpercentile(q_vals, pctl)) if len(q_vals) > 10 else np.nan


def count_onset(cnt: pd.Series, thresh: float,
                ev: SPEEvent) -> pd.Timestamp | None:
    """
    onset 전 6h ~ peak 후 6h 범위에서 count가 thresh를 첫 초과하는 시점.
    threshold를 한 번도 넘지 않으면 해당 구간의 최댓값 시점으로 대체.
    """
    t_start = ev.onset - pd.Timedelta(hours=6)
    t_end   = ev.peak  + pd.Timedelta(hours=6)
    seg     = cnt.loc[t_start:t_end]
    above   = seg[seg > thresh]
    if len(above):
        return above.index[0]
    seg_ev = cnt.loc[ev.onset:t_end].dropna()
    return seg_ev.idxmax() if len(seg_ev) else None


# ─────────────────────────────────────────────────────────────────
# (1) Superposed Epoch
# ─────────────────────────────────────────────────────────────────
def build_epoch(cnt: pd.Series, events: list[SPEEvent],
                resample_min: int = 30) -> pd.DataFrame:
    """
    peak 기준 epoch 행렬. 행=이벤트, 열=peak 기준 시간[h].
    값 = count / peak_count (peak 기준 정규화).
    """
    h_axis = np.arange(-PRE_H, POST_H + resample_min / 60, resample_min / 60)
    rows   = []
    for ev in events:
        pk_cnt = cnt.loc[ev.peak - pd.Timedelta(hours=2):
                          ev.peak + pd.Timedelta(hours=2)].max()
        if not (np.isfinite(pk_cnt) and pk_cnt > 0):
            continue
        vals = []
        for dh in h_axis:
            t   = ev.peak + pd.Timedelta(hours=float(dh))
            idx = cnt.index.get_indexer([t], method="nearest")[0]
            vals.append(cnt.iloc[idx] / pk_cnt if 0 <= idx < len(cnt) else np.nan)
        rows.append(vals)
    return pd.DataFrame(rows, columns=h_axis) if rows else pd.DataFrame()


def plot_superposed_epoch(df_count: pd.DataFrame, events: list[SPEEvent]):
    """
    3행(PD1/PD2/PD3) × 3열(O/OU/OUT) 패널.
    실선=median, 음영=IQR(25th~75th), A=실선, B=점선.
    """
    nrows, ncols = len(COUNT_PD_KEYS), len(COUNT_PROTON_LOGICS)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5 * ncols, 3.8 * nrows), squeeze=False)
    fig.suptitle(
        "Superposed Epoch  (peak-normalized, 30-min bin)\n"
        "solid=median, shade=IQR, solid line=A, dashed=B", fontsize=11)

    colors = {"PD1": "#2980b9", "PD2": "#27ae60", "PD3": "#8e44ad"}
    ls_map = {"A": "-", "B": "--"}

    for ri, pd_key in enumerate(COUNT_PD_KEYS):
        for ci, logic in enumerate(COUNT_PROTON_LOGICS):
            ax = axes[ri][ci]
            for side in COUNT_SIDES:
                try:
                    s = df_count[pd_key, side, logic].dropna()
                except KeyError:
                    continue
                mat = build_epoch(s, events)
                if mat.empty:
                    continue
                h   = mat.columns.values.astype(float)
                med = mat.median(axis=0).values
                q25 = mat.quantile(0.25, axis=0).values
                q75 = mat.quantile(0.75, axis=0).values
                c   = colors[pd_key]
                ax.plot(h, med, color=c, ls=ls_map[side], lw=1.5,
                        label=f"{side} (n={len(mat)})")
                if side == "A":
                    ax.fill_between(h, q25, q75, color=c, alpha=0.15)
            ax.axvline(0, ls="--", color="red", lw=1, label="Peak")
            ax.axhline(1, ls=":",  color="gray", lw=0.8)
            ax.set_xlim(-PRE_H, POST_H)
            ax.set_xlabel("Hours from Peak", fontsize=8)
            ax.set_ylabel("Normalized Count", fontsize=8)
            ax.set_title(f"{pd_key} / {logic}", fontsize=9)
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)

    fig.tight_layout()
    out = OUTPUT_DIR / "ana4_superposed_epoch.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[ana4] Superposed epoch saved: {out}")


# ─────────────────────────────────────────────────────────────────
# (2) Rise Time CDF
# ─────────────────────────────────────────────────────────────────
def plot_rise_time_cdf(df_count: pd.DataFrame,
                       proxy: pd.Series,
                       events: list[SPEEvent]):
    """
    count > quiet p{COUNT_THRESH_PCTL} 첫 시점 → peak 까지 시간 = rise time.
    CDF: x시간 이내에 peak에 도달한 이벤트 비율.
    O/OU/OUT 모두 포함. O가 반응 없으면 CDF가 평탄하게 나타남.
    """
    nrows = len(COUNT_PD_KEYS)
    ncols = len(COUNT_PROTON_LOGICS)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.5 * ncols, 3.2 * nrows), squeeze=False)
    fig.suptitle(
        f"Rise Time CDF  (count > quiet p{SPE_COUNT_THRESH_PCTL}  first crossing → Peak)\n"
        "CDF: fraction of events reaching peak within x hours", fontsize=11)

    colors_side = {"A": "#e74c3c", "B": "#3498db"}

    for ri, pd_key in enumerate(COUNT_PD_KEYS):
        for ci, logic in enumerate(COUNT_PROTON_LOGICS):
            ax = axes[ri][ci]
            for side in COUNT_SIDES:
                try:
                    cnt = df_count[pd_key, side, logic]
                except KeyError:
                    continue
                th = quiet_threshold(cnt, proxy)
                if not np.isfinite(th):
                    continue
                rt = []
                for ev in events:
                    t_on = count_onset(cnt, th, ev)
                    if t_on is None:
                        continue
                    dt = (ev.peak - t_on).total_seconds() / 3600
                    if 0 < dt < 200:
                        rt.append(dt)
                if not rt:
                    continue
                rt_s = np.sort(rt)
                cdf  = np.arange(1, len(rt_s) + 1) / len(rt_s)
                ax.step(rt_s, cdf, color=colors_side[side], lw=1.5,
                        label=f"{side} (n={len(rt_s)})")
            ax.set_xlabel("Rise Time (h)", fontsize=8)
            ax.set_ylabel("CDF", fontsize=8)
            ax.set_title(f"{pd_key} / {logic}", fontsize=9)
            ax.set_xlim(left=0)
            ax.set_ylim(0, 1.05)
            if ax.get_legend_handles_labels()[1]:
                ax.legend(fontsize=6)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)

    fig.tight_layout()
    out = OUTPUT_DIR / "ana4_rise_time_cdf.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[ana4] Rise time CDF saved: {out}")


# ─────────────────────────────────────────────────────────────────
# (3) Flux onset vs Count onset 비교
# ─────────────────────────────────────────────────────────────────
def compute_onset_delays(df_count: pd.DataFrame,
                         proxy: pd.Series,
                         events: list[SPEEvent]) -> pd.DataFrame:
    """
    delay_min = count_onset - flux_onset [분]
    negative  = count가 flux보다 먼저 상승 (count가 조기 탐지 채널)
    O/OU/OUT 모두 포함. O가 반응 없으면 delay가 무작위 또는 극단값으로 나타남.
    """
    flux_onsets = {}
    for ev in events:
        seg   = proxy.loc[ev.onset - pd.Timedelta(hours=12):ev.peak]
        above = seg[seg >= SPE_PROXY_THRESH]
        flux_onsets[ev.onset] = above.index[0] if len(above) else ev.onset

    records = []
    for pd_key in COUNT_PD_KEYS:
        for side in COUNT_SIDES:
            for logic in COUNT_PROTON_LOGICS:
                try:
                    cnt = df_count[pd_key, side, logic]
                except KeyError:
                    continue
                th = quiet_threshold(cnt, proxy)
                if not np.isfinite(th):
                    print(f"  [ana4] warn: quiet_threshold NaN for "
                          f"{pd_key}-{side}-{logic}, skipping")
                    continue
                for ev in events:
                    c_on = count_onset(cnt, th, ev)
                    f_on = flux_onsets.get(ev.onset)
                    if c_on is None or f_on is None:
                        continue
                    delay = (c_on - f_on).total_seconds() / 60
                    records.append({
                        "pd_key":      pd_key,
                        "side":        side,
                        "logic":       logic,
                        "event_onset": ev.onset,
                        "flux_onset":  f_on,
                        "count_onset": c_on,
                        "delay_min":   round(delay, 1),
                        "peak_flux":   ev.peak_flux,
                    })
    return pd.DataFrame(records)


def plot_onset_comparison(df_delay: pd.DataFrame):
    """delay_min 히스토그램 (logic별 패널, PD별 색)."""
    if df_delay.empty:
        print("[ana4] onset delay: No data")
        return
    ncols = len(COUNT_PROTON_LOGICS)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    colors = {"PD1": "#2980b9", "PD2": "#27ae60", "PD3": "#8e44ad"}

    for ci, logic in enumerate(COUNT_PROTON_LOGICS):
        ax  = axes[ci]
        sub = df_delay[df_delay["logic"] == logic]
        for pd_key in COUNT_PD_KEYS:
            vals = sub[sub["pd_key"] == pd_key]["delay_min"].dropna()
            if vals.empty:
                continue
            ax.hist(vals, bins=15, alpha=0.5, color=colors[pd_key],
                    label=f"{pd_key} med={vals.median():.0f}min", density=True)
        ax.axvline(0, ls="--", color="black", lw=1)
        ax.set_xlabel("Delay (min) [count - flux onset]\nnegative = count rises first",
                      fontsize=8)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_title(f"Logic: {logic}", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Count Onset vs Proxy Flux Onset delay distribution\n"
                 "O / OU / OUT all included", fontsize=11)
    fig.tight_layout()
    out = OUTPUT_DIR / "ana4_onset_comparison.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[ana4] Onset comparison saved: {out}")


# ─────────────────────────────────────────────────────────────────
# (4) Logic onset order (O → OU → OUT)
# ─────────────────────────────────────────────────────────────────
def plot_logic_order(df_delay: pd.DataFrame):
    """
    O / OU / OUT onset 지연 scatter.
    에너지 응답 순서: O(coincidence only)가 낮은 에너지 포함 → 먼저 상승 예상.
    O가 반응 없으면 scatter가 무작위 분포로 나타남.
    diamond = median.
    """
    if df_delay.empty:
        return
    colors_l = {"O": "#e74c3c", "OU": "#f39c12", "OUT": "#27ae60"}
    fig, axes = plt.subplots(1, len(COUNT_PD_KEYS),
                              figsize=(5 * len(COUNT_PD_KEYS), 4))
    fig.suptitle("Logic Onset Order  (O / OU / OUT)\n"
                 "diamond=median | expected order: O first if low-energy response",
                 fontsize=11)

    for ci, pd_key in enumerate(COUNT_PD_KEYS):
        ax  = axes[ci]
        sub = df_delay[df_delay["pd_key"] == pd_key]
        for logic in COUNT_PROTON_LOGICS:
            vals = sub[sub["logic"] == logic]["delay_min"].dropna()
            if vals.empty:
                continue
            ax.scatter([logic] * len(vals), vals,
                       color=colors_l[logic], alpha=0.5, s=20, zorder=3)
            ax.plot(logic, vals.median(), "D",
                    color=colors_l[logic], ms=10, zorder=5,
                    label=f"{logic} med={vals.median():.0f}min")
        ax.axhline(0, ls="--", color="gray", lw=0.8)
        ax.set_title(pd_key, fontsize=10)
        ax.set_ylabel("Delay vs flux onset (min)", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(labelsize=8)

    fig.tight_layout()
    out = OUTPUT_DIR / "ana4_logic_order.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[ana4] Logic onset order saved: {out}")


# ─────────────────────────────────────────────────────────────────
# (5) PD 방향각 의존성
# ─────────────────────────────────────────────────────────────────
def plot_pd_direction(df_count: pd.DataFrame, events: list[SPEEvent]):
    """
    PD_COMPARE_KEYS[0](15°, polar) / PD_COMPARE_KEYS[1](71°, equatorial) peak count 비율.
    비율 > 1 → Parker spiral 방향(polar)에서 더 많이 계수.
    O/OU/OUT 모두 포함하여 로직별 이방성 차이를 확인.
    """
    pairs  = [("PD1","PD2"), ("PD1","PD3"), ("PD2","PD3")]
    ncols  = len(COUNT_PROTON_LOGICS)
    nrows  = len(pairs)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5*ncols, 3.5*nrows), squeeze=False)
    fig.suptitle("PD Peak Count Ratio (A-side)\n"
                 "ratio>1: numerator-direction dominant", fontsize=11)

    for ri, (pd0, pd1) in enumerate(pairs):
        for ci, logic in enumerate(COUNT_PROTON_LOGICS):
            ax = axes[ri][ci]
            ratios = []
            for ev in events:
                try:
                    p0 = df_count[pd0, "A", logic].loc[ev.onset:ev.end].max()
                    p1 = df_count[pd1, "A", logic].loc[ev.onset:ev.end].max()
                except KeyError:
                    continue
                if np.isfinite(p0) and np.isfinite(p1) and p1 > 0:
                    ratios.append(p0 / p1)
            if ratios:
                med = float(np.median(ratios))
                ax.hist(ratios, bins=10, color="#9b59b6",
                        alpha=0.75, edgecolor="white")
                ax.axvline(med, ls="--", color="black", lw=1.5,
                           label=f"median={med:.2f}")
                ax.legend(fontsize=8)
            ax.axvline(1.0, ls=":", color="gray", lw=1)
            ax.set_xlabel(f"{pd0}/{pd1} Ratio", fontsize=9)
            ax.set_ylabel("N events", fontsize=9)
            ax.set_title(f"{pd0}/{pd1} | {logic}", fontsize=9)
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = OUTPUT_DIR / "ana4_pd_direction.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[ana4] PD direction dependence saved: {out}")


# ─────────────────────────────────────────────────────────────────
def main():
    print("[ana4] Loading data...")
    df_count, _ = ksem_io.load(COUNT_PARQUET_DIR)
    sensor_data, _ = kma_ksem_flux_io.load(FLUX_PARQUET_DIR)
    df_flux_p = sensor_data.get("proton", pd.DataFrame())

    for df in [df_count, df_flux_p]:
        if not df.empty and df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

    proxy = make_proxy(df_flux_p)

    # proxy 분포 진단 → SPE_PROXY_THRESH 적정값 확인
    pv = proxy.dropna()
    print(f"  [diag] proxy percentiles (dE-integrated, cm-2 sr-1 s-1):")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"         p{p:2d} = {np.nanpercentile(pv, p):.3e}")
    print(f"  [diag] current SPE_PROXY_THRESH = {SPE_PROXY_THRESH:.1e}")
    print(f"  [diag] fraction above thresh    = {(pv >= SPE_PROXY_THRESH).mean():.3f}")

    events = detect_events(proxy)
    if not events:
        print("[ana4] No events: exit")
        return

    plot_superposed_epoch(df_count, events)
    plot_rise_time_cdf(df_count, proxy, events)

    df_delay = compute_onset_delays(df_count, proxy, events)
    plot_onset_comparison(df_delay)
    plot_logic_order(df_delay)
    plot_pd_direction(df_count, events)

    if not df_delay.empty:
        summary = (df_delay.groupby(["pd_key", "side", "logic"])["delay_min"]
                   .agg(["median", "mean", "std", "count"])
                   .reset_index()
                   .rename(columns={"median": "delay_median_min",
                                    "mean":   "delay_mean_min",
                                    "std":    "delay_std_min",
                                    "count":  "n_events"}))
        out_csv = OUTPUT_DIR / "ana4_onset_delay_table.csv"
        summary.to_csv(out_csv, index=False, float_format="%.2f")
        print(f"[ana4] summary table saved: {out_csv}")
        print(summary.to_string(index=False))

        df_delay.to_csv(OUTPUT_DIR / "ana4_onset_delay_raw.csv", index=False)
        print(f"[ana4] Raw delay saved done")


if __name__ == "__main__":
    main()
