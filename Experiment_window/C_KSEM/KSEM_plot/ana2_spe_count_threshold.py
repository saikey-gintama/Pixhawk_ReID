"""
ana2_spe_count_threshold.py
============================
SPE / E_SPE 구간 count 분포 및 임계값 도출

SPE 탐지 기준 (proton):
  proton flux (E8+E9+E10 ΔE-적분) >= SPE_PROXY_THRESH 가
  MIN_SPE_DURATION_H 이상 연속되는 구간을 이벤트로 인정.

E_SPE 탐지 기준 (electron):
  electron flux (E4+E5+E6 ΔE-적분) >= E_SPE_PROXY_THRESH 가
  MIN_E_SPE_DURATION_H 이상 연속되는 구간을 이벤트로 인정.
  E_SPE_PROXY_THRESH=None 이면 flux 분포만 출력하고 electron 블록 스킵.

이벤트 패널 구성:
  mode="proton"   → 행0: proton flux / 행1: O count / 행2: OU count / 행3: OUT count
  mode="electron" → 행0: electron flux / 행1: F count / 행2: FT count /
                    행3: FTU count / 행4: FTUO count
  각 count 행: PD1=파랑, PD2=초록, PD3=보라 / A=실선, B=점선

출력:
  panels_proton/ana2_event_XX_panel.png     (proton SPE 기준)
  panels_electron/ana2_event_XX_panel.png   (electron E_SPE 기준)
  ana2_multiplier_hist.png                  (proton)
  ana2_ab_symmetry.png                      (proton)
  ana2_spe_threshold_table.csv              (proton)
  ana2_electron_multiplier_hist.png         (electron)
  ana2_electron_ab_symmetry.png             (electron)
  ana2_electron_threshold_table.csv         (electron)

튜닝값 (이 파일 단독):
  BG_WINDOW_DAYS : onset 이전 배경 슬라이딩 윈도우 크기.
  BG_QUIET_DAYS  : 윈도우 중 가장 조용한 N일만 median 사용 (Löwe et al. 2025).
  PRE_ONSET_H    : 이벤트 패널 onset 앞 시각화 여백 (분석 범위 아님).
  POST_END_H     : 이벤트 패널 end 뒤 시각화 여백 (분석 범위 아님).
  PERCENTILES    : Quiet 분포에서 추출할 FSM threshold 후보 퍼센타일.
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import ksem_flux_config as cfg
from ksem_flux_config import (
    OUTPUT_DIR,
    SPE_PROXY_THRESH, SPE_PROXY_LABEL,
    E_SPE_PROXY_THRESH, E_SPE_PROXY_LABEL,
    COUNT_PROTON_LOGICS, COUNT_ELECTRON_LOGICS, COUNT_PD_KEYS, COUNT_SIDES,
    SPE_COUNT_THRESH_PCTL, E_SPE_COUNT_THRESH_PCTL,
    BG_WINDOW_DAYS, BG_QUIET_DAYS, COUNT_BG_K,
)
from ksem_common import (
    SPEEvent,
    load_data,
    make_proton_proxy,
    make_electron_proxy,
    detect_proton_events,
    detect_electron_events,
    get_bg_median,
    get_bg_threshold,
    print_proxy_diag,
)

# ── ana2 단독 튜닝값 ──────────────────────────────────────────────
PRE_ONSET_H     = 12   # [h]   이벤트 패널 시각화 앞 여백
POST_END_H      = 12   # [h]   이벤트 패널 시각화 뒤 여백
PERCENTILES     = [50, 60, 70, 80, 90]   # Quiet 퍼센타일 (진단용 참고)
# BG_WINDOW_DAYS / BG_QUIET_DAYS / COUNT_BG_K 는 config에서 공유 (ana3·ana4와 동일)



# ─────────────────────────────────────────────────────────────────
# (1) 이벤트별 시계열 패널
# ─────────────────────────────────────────────────────────────────
def plot_event_panels(df_count: pd.DataFrame,
                      proxy_p: pd.Series,
                      proxy_e: pd.Series | None,
                      events: list[SPEEvent],
                      suffix: str = "proton",
                      mode: str = "proton"):
    """
    mode="proton"   → 행0: proton flux  / 행1~3: O / OU / OUT count
    mode="electron" → 행0: electron flux / 행1~4: F / FT / FTU / FTUO count

    각 count 행: PD1=파랑, PD2=초록, PD3=보라 / A=실선, B=점선
    suffix → 저장 폴더 이름 (panels_proton / panels_electron)
    """
    palette = {"PD1": "#2980b9", "PD2": "#27ae60", "PD3": "#8e44ad"}

    panel_dir = OUTPUT_DIR / f"panels_{suffix}"
    panel_dir.mkdir(exist_ok=True)

    # mode에 따라 flux와 count 로직 결정
    if mode == "proton":
        flux_series  = proxy_p
        flux_label   = SPE_PROXY_LABEL
        flux_thresh  = SPE_PROXY_THRESH
        flux_color   = "k"
        logics       = COUNT_PROTON_LOGICS
        event_label  = "SPE"
    else:
        flux_series  = proxy_e
        flux_label   = E_SPE_PROXY_LABEL
        flux_thresh  = E_SPE_PROXY_THRESH
        flux_color   = "#c0392b"
        logics       = COUNT_ELECTRON_LOGICS
        event_label  = "E_SPE"

    n_rows = 1 + len(logics)   # flux 1행 + count N행

    for ei, ev in enumerate(events):
        t0 = ev.onset - pd.Timedelta(hours=PRE_ONSET_H)
        t1 = ev.end   + pd.Timedelta(hours=POST_END_H)

        fig, axes = plt.subplots(n_rows, 1,
                                  figsize=(13, 2.8 * n_rows), sharex=True)
        if n_rows == 1:
            axes = [axes]

        fig.suptitle(
            f"{event_label} Event #{ei+1}  |  "
            f"Onset: {ev.onset.strftime('%Y-%m-%d %H:%M')}  "
            f"Peak flux: {ev.peak_flux:.2e} cm-2 sr-1 s-1",
            fontsize=10)

        # ── 행0: flux (log scale) ─────────────────────────────────
        ax0  = axes[0]
        seg  = flux_series.loc[t0:t1].replace(0, np.nan)
        ax0.semilogy(seg.index, seg.values, color=flux_color, lw=1.1)
        if flux_thresh is not None:
            ax0.axhline(flux_thresh, ls=":", color="red", lw=1,
                        label=f"thresh = {flux_thresh:.1e}")
        ax0.axvline(ev.onset, ls="--", color="orange", lw=1, alpha=0.8, label="onset")
        ax0.axvline(ev.end,   ls="--", color="gray",   lw=1, alpha=0.8, label="end")
        ax0.set_ylabel(f"{'Proton' if mode=='proton' else 'Electron'} flux\n"
                       f"({flux_label})\n[cm-2 sr-1 s-1]", fontsize=7)
        ax0.legend(fontsize=6, loc="upper right")
        ax0.grid(True, alpha=0.3)

        # ── 행1~N: count (logic별, PD/side별 색/선 구분) ─────────
        for li, logic in enumerate(logics):
            ax = axes[1 + li]
            for pd_key in COUNT_PD_KEYS:
                color = palette[pd_key]
                for side in COUNT_SIDES:
                    ls = "-" if side == "A" else "--"
                    try:
                        s = df_count[pd_key, side, logic].loc[t0:t1]
                        ax.plot(s.index, s.values,
                                color=color, ls=ls, lw=0.9, alpha=0.8,
                                label=f"{pd_key}-{side}")
                    except KeyError:
                        pass

            ax.axvline(ev.onset, ls="--", color="orange", lw=1,   alpha=0.6)
            ax.axvline(ev.peak,  ls=":",  color="red",    lw=1.2, alpha=0.7, label="peak")
            ax.axvline(ev.end,   ls="--", color="gray",   lw=1,   alpha=0.6)
            ax.set_ylim(bottom=0)
            ax.set_ylabel(f"Count [{logic}]", fontsize=8)
            ax.legend(fontsize=5, ncol=6, loc="upper left")
            ax.grid(True, alpha=0.3)

        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
        fig.tight_layout()
        out = panel_dir / f"ana2_event_{ei+1:02d}_panel.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)

    print(f"[ana2] {suffix} event panels {len(events)} saved → {panel_dir}")


# ─────────────────────────────────────────────────────────────────
# (2) 퍼센타일 임계값 + 배율 계산 (proton/electron 공용)
# ─────────────────────────────────────────────────────────────────
def compute_thresholds(df_count: pd.DataFrame,
                       proxy: pd.Series,
                       events: list[SPEEvent],
                       logics: list[str],
                       thresh: float,
                       count_thresh_pctl: int):
    """
    각 (pd_key, side, logic) 채널에 대해 계산:
      quiet_pXX        : Quiet 구간 count 퍼센타일 → FSM threshold 후보
      spe_median_count : SPE/E_SPE 구간 count 중앙값
      event_bg_ratio   : onset~end 구간 count 중앙값 / 조용한 배경 median 배율
      spe_exceed_qxx   : SPE/E_SPE 구간 중 quiet_pxx 초과 비율
    proton:   logics=COUNT_PROTON_LOGICS,   thresh=SPE_PROXY_THRESH
    electron: logics=COUNT_ELECTRON_LOGICS, thresh=E_SPE_PROXY_THRESH
    """
    is_spe   = proxy >= thresh
    is_quiet = ~is_spe

    records      = []
    mult_data    = {}
    ab_diff_data = {}

    for pd_key in COUNT_PD_KEYS:
        for side in COUNT_SIDES:
            for logic in logics:
                try:
                    cnt = df_count[pd_key, side, logic]
                except KeyError:
                    continue

                q_vals  = cnt[is_quiet & cnt.notna()]
                sp_vals = cnt[is_spe   & cnt.notna()]

                pct = {f"quiet_p{p}": float(np.nanpercentile(q_vals, p))
                       if len(q_vals) > 0 else np.nan
                       for p in PERCENTILES}

                ratios, exceed_rates = [], []
                bg_threshs, bg_exceed_rates = [], []
                for ev in events:
                    bg = get_bg_median(cnt, ev.onset, BG_WINDOW_DAYS, BG_QUIET_DAYS)
                    if np.isnan(bg) or bg <= 0:
                        continue
                    event_cnt = cnt.loc[ev.onset:ev.end].median()
                    if np.isfinite(event_cnt):
                        ratios.append(event_cnt / bg)
                    q_thresh_val = pct.get(f"quiet_p{count_thresh_pctl}", np.nan)
                    if np.isfinite(q_thresh_val):
                        seg = cnt.loc[ev.onset:ev.end]
                        if len(seg) > 0:
                            exceed_rates.append((seg > q_thresh_val).mean())
                    # ── 배경 기반 임계값 (ana4 onset 임계값과 동일) ──
                    bg_th = get_bg_threshold(cnt, ev.onset,
                                             BG_WINDOW_DAYS, BG_QUIET_DAYS, COUNT_BG_K)
                    if np.isfinite(bg_th):
                        bg_threshs.append(bg_th)
                        seg = cnt.loc[ev.onset:ev.end]
                        if len(seg) > 0:
                            bg_exceed_rates.append((seg > bg_th).mean())

                records.append({
                    "pd_key": pd_key, "side": side, "logic": logic,
                    **pct,
                    "spe_median_count":
                        float(sp_vals.median()) if len(sp_vals) else np.nan,
                    "event_bg_ratio_median":
                        round(float(np.nanmedian(ratios)), 2) if ratios else np.nan,
                    f"spe_exceed_q{count_thresh_pctl}_rate":
                        round(float(np.nanmean(exceed_rates)), 3) if exceed_rates else np.nan,
                    "bg_thresh_median":
                        round(float(np.nanmedian(bg_threshs)), 2) if bg_threshs else np.nan,
                    "spe_exceed_bgthresh_rate":
                        round(float(np.nanmean(bg_exceed_rates)), 3) if bg_exceed_rates else np.nan,
                    "quiet_n": len(q_vals),
                    "spe_n":   len(sp_vals),
                })
                mult_data[(pd_key, side, logic)] = ratios

        for logic in logics:
            try:
                cA = df_count[pd_key, "A", logic]
                cB = df_count[pd_key, "B", logic]
            except KeyError:
                continue
            denom = (cA + cB).replace(0, np.nan)
            ab_diff_data[(pd_key, logic)] = (cA - cB) / denom

    return pd.DataFrame(records), mult_data, ab_diff_data


def plot_multiplier_hist(mult_data: dict, logics: list[str], suffix: str = ""):
    """이벤트 배율 히스토그램. suffix="" → proton / suffix="_electron" → electron."""
    ncols = len(logics)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    if ncols == 1:
        axes = [axes]
    colors = {"PD1": "#2980b9", "PD2": "#27ae60", "PD3": "#8e44ad"}

    for ci, logic in enumerate(logics):
        ax = axes[ci]
        for pd_key in COUNT_PD_KEYS:
            for side in COUNT_SIDES:
                vals = mult_data.get((pd_key, side, logic), [])
                if vals:
                    ax.hist(vals, bins=15, alpha=0.4, color=colors[pd_key],
                            label=f"{pd_key}-{side}", density=True)
        ax.axvline(1.0, ls="--", color="black", lw=1, label="ratio=1 (no change)")
        ax.set_xlabel(
            f"Event median count / quiet {BG_QUIET_DAYS}-day median"
            f" (of {BG_WINDOW_DAYS}-day window)", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_title(f"Logic: {logic}", fontsize=10)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    label = "Electron E_SPE" if suffix else "Proton SPE"
    title = (
        f"{label} Event Multiplier  "
        f"(onset~end median / quiet {BG_QUIET_DAYS}-day median"
        f" of {BG_WINDOW_DAYS}-day window)"
    )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out = OUTPUT_DIR / f"ana2{suffix}_multiplier_hist.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ana2] multiplier histogram saved: {out}")


def plot_ab_symmetry(ab_diff_data: dict, suffix: str = ""):
    """(A-B)/(A+B) 시계열. suffix="" → proton / suffix="_electron" → electron."""
    keys = list(ab_diff_data.keys())
    if not keys:
        return
    fig, axes = plt.subplots(len(keys), 1,
                              figsize=(14, 2.0 * len(keys)), sharex=True)
    if len(keys) == 1:
        axes = [axes]
    for idx, ((pd_key, logic), s) in enumerate(ab_diff_data.items()):
        ax = axes[idx]
        ax.plot(s.resample("1h").median().index,
                s.resample("1h").median().values,
                lw=0.7, color="#c0392b", alpha=0.85)
        ax.axhline(0, ls="--", color="black", lw=0.8)
        ax.set_ylim(-1.2, 1.2)
        ax.set_ylabel(f"{pd_key}/{logic}", fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    label = "Electron" if suffix else "Proton"
    fig.suptitle(
        f"{label} A/B asymmetry  (A-B)/(A+B)  |  0=perfect symmetry  ±1=one-sided",
        fontsize=11)
    fig.tight_layout()
    out = OUTPUT_DIR / f"ana2{suffix}_ab_symmetry.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[ana2] A/B symmetry saved: {out}")


# ─────────────────────────────────────────────────────────────────
def main():
    print("[ana2] Loading data...")
    df_count, df_flux_p, df_flux_e = load_data()

    # ── Proton flux ───────────────────────────────────────────────
    proxy_p  = make_proton_proxy(df_flux_p)
    print_proxy_diag(proxy_p, SPE_PROXY_THRESH, "proton SPE")
    events_p = detect_proton_events(proxy_p)

    # ── Electron flux ─────────────────────────────────────────────
    proxy_e = make_electron_proxy(df_flux_e)
    print_proxy_diag(proxy_e, E_SPE_PROXY_THRESH, "electron E_SPE")

    if E_SPE_PROXY_THRESH is None:
        print("[ana2] E_SPE_PROXY_THRESH=None → electron 분석 스킵. "
              "위 diag 결과의 p95를 참고해 config에 설정 후 재실행하세요.")
        return

    events_e = detect_electron_events(proxy_e)

    # ── Proton 이벤트 패널 ────────────────────────────────────────
    plot_event_panels(df_count, proxy_p, proxy_e,
                      events_p, suffix="proton", mode="proton")

    # ── Proton threshold 분석 ─────────────────────────────────────
    df_thresh_p, mult_p, ab_p = compute_thresholds(
        df_count, proxy_p, events_p,
        logics=COUNT_PROTON_LOGICS,
        thresh=SPE_PROXY_THRESH,
        count_thresh_pctl=SPE_COUNT_THRESH_PCTL)
    plot_multiplier_hist(mult_p, COUNT_PROTON_LOGICS, suffix="")
    plot_ab_symmetry(ab_p, suffix="")
    out_p = OUTPUT_DIR / "ana2_spe_threshold_table.csv"
    df_thresh_p.to_csv(out_p, index=False, float_format="%.4g")
    print(f"[ana2] proton threshold table saved: {out_p}")
    print(df_thresh_p.to_string(index=False))

    # ── Electron 이벤트 패널 ──────────────────────────────────────
    plot_event_panels(df_count, proxy_p, proxy_e,
                      events_e, suffix="electron", mode="electron")

    # ── Electron threshold 분석 ───────────────────────────────────
    df_thresh_e, mult_e, ab_e = compute_thresholds(
        df_count, proxy_e, events_e,
        logics=COUNT_ELECTRON_LOGICS,
        thresh=E_SPE_PROXY_THRESH,
        count_thresh_pctl=E_SPE_COUNT_THRESH_PCTL)
    plot_multiplier_hist(mult_e, COUNT_ELECTRON_LOGICS, suffix="_electron")
    plot_ab_symmetry(ab_e, suffix="_electron")
    out_e = OUTPUT_DIR / "ana2_electron_threshold_table.csv"
    df_thresh_e.to_csv(out_e, index=False, float_format="%.4g")
    print(f"[ana2] electron threshold table saved: {out_e}")
    print(df_thresh_e.to_string(index=False))


if __name__ == "__main__":
    main()