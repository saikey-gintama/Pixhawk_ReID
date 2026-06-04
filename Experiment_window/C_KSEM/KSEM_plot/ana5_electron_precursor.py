"""
ana5_electron_precursor.py
==========================
Electron precursor 분석 — electron이 proton SPE의 조기 경보가 될 수 있는가?

배경 (Torres et al., 2025):
  근 상대론적 전자(수백 keV~수 MeV)는 proton보다 빠르게 도달하며,
  SEP proton 이벤트의 거의 모든 경우에 함께 관측된다.
  전자 강도 상승이 proton보다 수십 분 먼저 발생하면 조기 경보로 활용 가능.

Lead time 탐지 방식 (Torres et al. 2025 방식):
  단일 onset 시점 대신 cross-correlation을 사용.
  proton proxy_p와 electron proxy_e(또는 count)를 5분 단위로 shift하며
  Pearson r이 최대가 되는 shift = lead time.
  양수 shift = electron이 proton보다 먼저 상승.

분석 구조:
  (1) proton SPE events_p 재탐지 (ana2/4와 동일 로직)
  (2) 각 SPE 이벤트에 대해:
      a) proxy_e vs proxy_p cross-correlation → flux lead time
      b) electron count(F/FT/FTU/FTUO) vs proxy_p cross-correlation → count lead time
  (3) lead time 분포 / lead vs peak_flux scatter / 경보 가능 비율 산출

출력:
  ana5_flux_lead_time_hist.png     electron flux cross-corr lead time 분포
  ana5_count_lead_time_hist.png    electron count cross-corr lead time 분포 (logic별 패널)
  ana5_lead_vs_peak_scatter.png    lead time vs SPE peak_flux scatter (Torres Fig.5 대응)
  ana5_precursor_summary.csv       이벤트별 원시 lead time 전체

튜닝값 (이 파일 단독):
  LEAD_SEARCH_H  : cross-corr shift 탐색 범위 [h]. 양방향 탐색.
  RESAMPLE_MIN   : cross-corr 계산용 리샘플 간격 [분]. 너무 작으면 노이즈 민감.
  MIN_CORR_R     : 유효 cross-corr 최솟값. 이 미만이면 lead time 미신뢰로 처리.
  MAX_LEAD_H     : 유효 lead time 상한 [h]. 초과하면 무관한 상승으로 간주.
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

import ksem_flux_config as cfg
from ksem_flux_config import (
    OUTPUT_DIR,
    SPE_PROXY_THRESH, SPE_PROXY_LABEL,
    E_SPE_PROXY_LABEL,
    COUNT_ELECTRON_LOGICS, COUNT_PD_KEYS, COUNT_SIDES,
)
from ksem_common import (
    SPEEvent,
    load_data,
    make_proton_proxy,
    make_electron_proxy,
    detect_proton_events,
)

# ── ana5 단독 튜닝값 ──────────────────────────────────────────────
LEAD_SEARCH_H = 6      # [h] cross-corr shift 탐색 범위 (양방향, ±LEAD_SEARCH_H)
RESAMPLE_MIN  = 5      # [분] cross-corr 계산용 리샘플 간격
MIN_CORR_R    = 0.5    # 유효 cross-corr 최솟값 (이 미만이면 lead time NaN 처리)
MAX_LEAD_H    = 6      # [h] 유효 lead time 상한


# ────────────────────────────────────────────────────────────────
# Cross-correlation lead time 탐지 (Torres et al. 2025 방식)
# ─────────────────────────────────────────────────────────────────
def crosscorr_lead(ref: np.ndarray,
                   sig: np.ndarray,
                   max_shift: int) -> tuple[float, float]:
    """
    ref(proton)와 sig(electron)의 cross-correlation.
    sig를 shift하며 Pearson r이 최대인 시점을 탐색.

    반환: (best_lag_steps, best_r)
      best_lag_steps > 0 → sig(electron)가 ref(proton)보다 먼저 (lead)
      best_lag_steps < 0 → sig(electron)가 ref(proton)보다 나중 (lag)
    """
    best_r, best_lag = -np.inf, 0
    n = len(ref)
    for shift in range(-max_shift, max_shift + 1):
        if abs(shift) >= n: 
            continue
        if shift > 0:
            r_seg = ref[shift:]
            s_seg = sig[:n - shift]
        elif shift < 0:
            r_seg = ref[:n + shift]
            s_seg = sig[-shift:]
        else:
            r_seg, s_seg = ref, sig

        valid = np.isfinite(r_seg) & np.isfinite(s_seg)
        if valid.sum() < 10:
            continue
        try:
            r, _ = stats.pearsonr(r_seg[valid], s_seg[valid])
        except Exception:
            continue
        if r > best_r:
            best_r, best_lag = r, shift

    return float(best_lag), float(best_r)


def compute_event_lead(proxy_p: pd.Series,
                       sig: pd.Series,
                       ev: SPEEvent) -> tuple[float | None, float | None]:
    """
    단일 이벤트에 대해 cross-correlation lead time 계산.
    탐색 범위: onset - LEAD_SEARCH_H ~ peak + 2h

    반환: (lead_min, best_r)  lead_min=None → 유효하지 않음
    양수 lead_min = sig(electron)이 proton보다 먼저
    """
    t0 = ev.onset - pd.Timedelta(hours=LEAD_SEARCH_H)
    t1 = ev.peak  + pd.Timedelta(hours=2)

    freq = f"{RESAMPLE_MIN}min"
    p = proxy_p.loc[t0:t1].resample(freq).mean()
    s = sig.loc[t0:t1].resample(freq).mean()

    common = p.index.intersection(s.index)
    if len(common) < 10:
        return None, None

    p_arr = np.log1p(p.loc[common].values.astype(float))
    s_arr = np.log1p(s.loc[common].values.astype(float))

    max_shift = int(LEAD_SEARCH_H * 60 / RESAMPLE_MIN)
    best_lag, best_r = crosscorr_lead(p_arr, s_arr, max_shift)

    if best_r < MIN_CORR_R:
        return None, float(best_r)

    lead_min = best_lag * RESAMPLE_MIN   # 양수 = electron 먼저
    if abs(lead_min) > MAX_LEAD_H * 60:
        return None, float(best_r)

    return float(lead_min), float(best_r)


# ─────────────────────────────────────────────────────────────────
def compute_lead_times(df_count: pd.DataFrame,
                       proxy_p: pd.Series,
                       proxy_e: pd.Series,
                       events_p: list[SPEEvent]) -> pd.DataFrame:
    """
    각 SPE 이벤트에 대해 electron flux/count cross-corr lead time 계산.

    lead time 정의 (양수 = electron이 proton보다 먼저):
      flux_lead  : proxy_e vs proxy_p cross-corr
      count_lead : electron count vs proxy_p cross-corr
    """
    records = []
    n_ev = len(events_p)

    for ei, ev in enumerate(events_p):
        if (ei + 1) % 10 == 0 or ei == 0:
            print(f"  [ana5] processing event {ei+1}/{n_ev} ...")

        # electron flux lead (이벤트당 1개)
        flux_lead, flux_r = compute_event_lead(proxy_p, proxy_e, ev)

        # electron count lead (PD/side/logic별)
        for pd_key in COUNT_PD_KEYS:
            for side in COUNT_SIDES:
                for logic in COUNT_ELECTRON_LOGICS:
                    try:
                        cnt = df_count[pd_key, side, logic]
                    except KeyError:
                        continue
                    count_lead, count_r = compute_event_lead(proxy_p, cnt, ev)

                    records.append({
                        "event_onset":   ev.onset,
                        "peak_flux_p":   ev.peak_flux,
                        "flux_lead_min": flux_lead,
                        "flux_corr_r":   flux_r,
                        "pd_key":        pd_key,
                        "side":          side,
                        "logic":         logic,
                        "count_lead_min": count_lead,
                        "count_corr_r":   count_r,
                    })

    df = pd.DataFrame(records)
    print(f"[ana5] lead time records: {len(df)}")
    return df


# ─────────────────────────────────────────────────────────────────
# 플롯
# ─────────────────────────────────────────────────────────────────
def plot_flux_lead_time(df: pd.DataFrame):
    """electron flux cross-corr lead time 분포 히스토그램."""
    flux_df = (df[["event_onset", "flux_lead_min", "flux_corr_r"]]
               .drop_duplicates("event_onset")
               .dropna(subset=["flux_lead_min"]))

    n_total     = df["event_onset"].nunique()
    n_detected  = len(flux_df)
    n_positive  = (flux_df["flux_lead_min"] > 0).sum()
    pct_det     = 100 * n_detected / n_total if n_total > 0 else 0
    pct_pos     = 100 * n_positive / n_total if n_total > 0 else 0

    fig, ax = plt.subplots(figsize=(7, 4))
    if not flux_df.empty:
        ax.hist(flux_df["flux_lead_min"], bins=20,
                color="#e74c3c", alpha=0.75, edgecolor="white")
        med = flux_df["flux_lead_min"].median()
        ax.axvline(med, ls="--", color="black", lw=1.5,
                   label=f"median={med:.0f} min")
    ax.axvline(0, ls=":", color="gray", lw=1, label="0 (simultaneous)")
    ax.set_xlabel("Electron flux lead time (min)\n"
                  "positive = electron first  |  "
                  "method: cross-correlation (Pearson r)", fontsize=9)
    ax.set_ylabel("N events", fontsize=9)
    ax.set_title(
        f"Electron Flux Lead Time  (proxy_e: {E_SPE_PROXY_LABEL})\n"
        f"Detected: {n_detected}/{n_total} ({pct_det:.0f}%)  "
        f"|  Lead>0: {n_positive}/{n_total} ({pct_pos:.0f}%)  "
        f"|  min_r={MIN_CORR_R}  |  ±{LEAD_SEARCH_H}h window",
        fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = OUTPUT_DIR / "ana5_flux_lead_time_hist.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ana5] flux lead time histogram saved: {out}")


def plot_count_lead_time(df: pd.DataFrame):
    """electron count cross-corr lead time 분포 — logic별 패널, PD별 색."""
    valid  = df.dropna(subset=["count_lead_min"])
    ncols  = len(COUNT_ELECTRON_LOGICS)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    colors = {"PD1": "#2980b9", "PD2": "#27ae60", "PD3": "#8e44ad"}

    for ci, logic in enumerate(COUNT_ELECTRON_LOGICS):
        ax  = axes[ci]
        sub = valid[valid["logic"] == logic]
        for pd_key in COUNT_PD_KEYS:
            vals = sub[sub["pd_key"] == pd_key]["count_lead_min"].dropna()
            if vals.empty:
                continue
            n_pos = (vals > 0).sum()
            ax.hist(vals, bins=15, alpha=0.5, color=colors[pd_key],
                    label=f"{pd_key} med={vals.median():.0f}m n+={n_pos}",
                    density=True)
        ax.axvline(0, ls=":", color="gray", lw=1)
        ax.set_xlabel("Count lead time (min)\n"
                      "positive = electron count before proton flux onset", fontsize=8)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_title(f"Logic: {logic}", fontsize=10)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Electron Count Lead Time  (cross-correlation, min_r={MIN_CORR_R})\n"
        f"positive = electron count rises before proton flux onset",
        fontsize=11)
    fig.tight_layout()
    out = OUTPUT_DIR / "ana5_count_lead_time_hist.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ana5] count lead time histogram saved: {out}")


def plot_lead_vs_peak(df: pd.DataFrame):
    """
    lead time vs SPE peak_flux scatter.
    Torres et al. (2025) Figure 5 대응.
    상관계수(flux_corr_r)를 점 크기로 표시 — r이 높을수록 신뢰도 높음.
    """
    flux_df = (df[["event_onset", "flux_lead_min", "flux_corr_r", "peak_flux_p"]]
               .drop_duplicates("event_onset")
               .dropna(subset=["flux_lead_min"]))

    fig, ax = plt.subplots(figsize=(7, 5))
    if not flux_df.empty:
        sizes = (flux_df["flux_corr_r"].clip(0, 1) * 150).values
        sc = ax.scatter(flux_df["peak_flux_p"],
                        flux_df["flux_lead_min"],
                        c=flux_df["flux_lead_min"],
                        s=sizes,
                        cmap="RdYlGn", alpha=0.8, edgecolors="gray", lw=0.5)
        plt.colorbar(sc, ax=ax, label="Lead time (min)")
        ax.text(0.02, 0.02,
                "Point size ∝ cross-corr Pearson r",
                transform=ax.transAxes, fontsize=7, color="gray")

    ax.axhline(0, ls="--", color="black", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(f"SPE peak proton flux\n({SPE_PROXY_LABEL})  [cm-2 sr-1 s-1]",
                  fontsize=9)
    ax.set_ylabel("Electron flux lead time (min)\npositive = electron first", fontsize=9)
    ax.set_title("Electron Flux Lead Time vs SPE Intensity\n"
                 "(Torres et al. 2025, Fig.5 analog  |  cross-correlation method)",
                 fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = OUTPUT_DIR / "ana5_lead_vs_peak_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ana5] lead vs peak scatter saved: {out}")


def print_summary(df: pd.DataFrame):
    """precursor 가능성 요약 출력."""
    n_events    = df["event_onset"].nunique()
    flux_df     = df.drop_duplicates("event_onset")
    n_detected  = flux_df["flux_lead_min"].notna().sum()
    n_positive  = (flux_df["flux_lead_min"] > 0).sum()

    print(f"\n[ana5] ── Precursor Summary ──────────────────────────────")
    print(f"  Method: cross-correlation (Pearson r, {RESAMPLE_MIN}-min bins)")
    print(f"  Search window: ±{LEAD_SEARCH_H}h  |  min_r={MIN_CORR_R}  "
          f"|  max_lead={MAX_LEAD_H}h")
    print(f"  Proton SPE events:          {n_events}")
    print(f"  Flux lead detected (r>={MIN_CORR_R}): {n_detected} "
          f"({100*n_detected/n_events:.0f}%)")
    print(f"  Lead > 0 (electron first):   {n_positive} "
          f"({100*n_positive/n_events:.0f}%)")
    if n_detected > 0:
        med = flux_df["flux_lead_min"].dropna().median()
        print(f"  Median lead time:            {med:.1f} min")
        med_r = flux_df["flux_corr_r"].dropna().median()
        print(f"  Median cross-corr r:         {med_r:.3f}")

    print(f"\n  Electron count lead (by logic, n_positive / median over all PD/side):")
    for logic in COUNT_ELECTRON_LOGICS:
        sub   = df[df["logic"] == logic]["count_lead_min"].dropna()
        n_pos = (sub > 0).sum()
        med   = sub.median() if len(sub) > 0 else np.nan
        med_s = f"{med:.1f}" if np.isfinite(med) else "N/A"
        print(f"    {logic:6s}: n_positive={n_pos:3d}  median={med_s} min")
    print(f"─────────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────
def main():
    from ksem_flux_config import E_SPE_PROXY_THRESH
    if E_SPE_PROXY_THRESH is None:
        print("[ana5] E_SPE_PROXY_THRESH=None → 실행 불가. "
              "ana2 실행 후 config에 E_SPE_PROXY_THRESH를 설정하세요.")
        return

    print("[ana5] Loading data...")
    df_count, df_flux_p, df_flux_e = load_data()

    proxy_p  = make_proton_proxy(df_flux_p)
    proxy_e  = make_electron_proxy(df_flux_e)
    events_p = detect_proton_events(proxy_p)
    if not events_p:
        print("[ana5] No proton SPE events: exit")
        return

    df_lead = compute_lead_times(df_count, proxy_p, proxy_e, events_p)

    plot_flux_lead_time(df_lead)
    plot_count_lead_time(df_lead)
    plot_lead_vs_peak(df_lead)

    out_csv = OUTPUT_DIR / "ana5_precursor_summary.csv"
    df_lead.to_csv(out_csv, index=False)
    print(f"[ana5] precursor summary saved: {out_csv}")

    print_summary(df_lead)


if __name__ == "__main__":
    main()