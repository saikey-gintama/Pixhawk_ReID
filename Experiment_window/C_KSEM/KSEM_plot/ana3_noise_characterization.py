"""
ana3_noise_characterization.py
================================
Quiet period 노이즈 특성 분석

Spike vs Onset 구분:
  spike : Quiet 기간 중 갑작스럽게 튀는 비정상 데이터 포인트.
          ROLLING_WINDOW분 이동 중앙값의 SPIKE_MULTIPLIER배 초과로 정의.
          원인: SEU(Single Event Upset), EMI 등 계측 노이즈.
  onset : SPE/E_SPE 이벤트 시작점 (ana2·ana4에서 별도 정의).
          물리적 입자 증가이며 최소 MIN_*_DURATION_H 이상 지속됨.

분석 구조:
  proton count (O/OU/OUT) — quiet_p 마스크 사용
  electron count (F/FT/FTU/FTUO) — quiet_e 마스크 사용
  각각 독립적으로 Poisson QQ / spike / CR 상관 분석 수행.

출력:
  ana3_poisson_qq.png              proton Poisson QQ
  ana3_spike_analysis.png          proton spike rate / A/B 일치율
  ana3_cr_correlation.png          CR vs proton count Pearson r
  ana3_count_noise_stats.json      proton 노이즈 통계

  ana3_electron_poisson_qq.png     electron Poisson QQ
  ana3_electron_spike_analysis.png electron spike rate / A/B 일치율
  ana3_electron_cr_correlation.png CR vs electron count Pearson r
  ana3_electron_noise_stats.json   electron 노이즈 통계
  (E_SPE_PROXY_THRESH=None 이면 electron quiet 마스크를 proton과 동일하게 사용)

튜닝값 (이 파일 단독):
  SPIKE_MULTIPLIER : 이동 중앙값의 이 배를 초과하면 spike로 판정.
  ROLLING_WINDOW   : spike 판정용 이동 중앙값 윈도우 [분].
"""

from __future__ import annotations
import json
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
    SPE_PROXY_THRESH, E_SPE_PROXY_THRESH,
    COUNT_PROTON_LOGICS, COUNT_ELECTRON_LOGICS, COUNT_PD_KEYS, COUNT_SIDES,
    BG_WINDOW_DAYS, BG_QUIET_DAYS,
)
from ksem_common import (
    load_data,
    make_proton_proxy,
    make_electron_proxy,
    detect_proton_events,
    detect_electron_events,
    get_quiet_bg_samples,
)

# ── ana3 단독 튜닝값 ──────────────────────────────────────────────
SPIKE_MULTIPLIER = 5.0   # rolling median의 이 배를 초과하면 spike
ROLLING_WINDOW   = 60    # [분] spike 판정용 이동 중앙값 윈도우
# BG_WINDOW_DAYS / BG_QUIET_DAYS 는 config에서 공유 (ana2·ana4와 동일)


# ─────────────────────────────────────────────────────────────────

def get_quiet_mask(df_count: pd.DataFrame,
                   proxy: pd.Series,
                   thresh: float,
                   label: str) -> pd.Series:
    """
    count 인덱스 기준 quiet 마스크.
    proxy < thresh 이면 True (quiet).
    thresh=None 이면 전체를 quiet로 간주.
    """
    if thresh is None:
        print(f"[ana3] {label} quiet: thresh=None → 전체를 quiet로 간주")
        return pd.Series(True, index=df_count.index)
    proxy_aligned = proxy.reindex(df_count.index, method="nearest",
                                  tolerance="5min")
    mask = (proxy_aligned < thresh) | proxy_aligned.isna()
    print(f"[ana3] {label} Quiet: {mask.sum():,} / {len(mask):,} points")
    return mask


# ─────────────────────────────────────────────────────────────────
# (1) Poisson QQ
# ─────────────────────────────────────────────────────────────────
def poisson_id(series: pd.Series) -> dict:
    """Index of Dispersion = Var / Mean. ID≈1: Poisson, ID≫1: 과분산."""
    v = series.dropna().values
    v = v[v >= 0]
    if len(v) < 30:
        return {"mean": np.nan, "var": np.nan, "ID": np.nan, "n": len(v)}
    m   = float(np.mean(v))
    var = float(np.var(v, ddof=1))
    return {"mean": m, "var": var,
            "ID": var / m if m > 0 else np.nan,
            "n": int(len(v))}


def plot_qq(df_count: pd.DataFrame,
            quiet_mask: pd.Series,
            logics: list[str],
            suffix: str = "") -> dict:
    """
    Poisson QQ 플롯 (6행 × len(logics)열).
    y=x 대각선 위로 벗어날수록 over-dispersed.
    suffix="" → proton / suffix="_electron" → electron
    반환: poisson_stats dict
    """
    nrows = len(COUNT_PD_KEYS) * len(COUNT_SIDES)
    ncols = len(logics)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.5 * ncols, 3.2 * nrows), squeeze=False)
    label = "Electron  (Quiet period)" if suffix else "Proton  (Quiet period)"
    fig.suptitle(f"Poisson QQ  {label}\n"
                 "y=x: Poisson | above y=x: overdispersed (ID > 1)", fontsize=11)

    poisson_stats = {}
    row = 0
    for pd_key in COUNT_PD_KEYS:
        for side in COUNT_SIDES:
            for ci, logic in enumerate(logics):
                ax = axes[row][ci]
                try:
                    s = df_count[pd_key, side, logic][quiet_mask].dropna()
                except KeyError:
                    ax.set_visible(False)
                    continue

                vals = s.values[s.values >= 0]
                if len(vals) < 20:
                    ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                            ha="center", fontsize=8)
                    continue

                lam   = np.mean(vals)
                probs = np.linspace(0.01, 0.99, 100)
                pq    = stats.poisson.ppf(probs, mu=lam)
                dq    = np.quantile(vals, probs)

                ax.scatter(pq, dq, s=7, alpha=0.5, color="#2980b9")
                lim = max(float(pq[-1]), float(dq[-1])) * 1.05
                ax.plot([0, lim], [0, lim], "r--", lw=1, label="y=x")

                pid    = poisson_id(s)
                id_str = f'{pid["ID"]:.2f}' if np.isfinite(pid["ID"]) else "N/A"
                ax.set_title(f"{pd_key}-{side} / {logic}\nID={id_str}  λ={lam:.1f}",
                             fontsize=7)
                ax.set_xlabel("Poisson Theoretical quantile", fontsize=6)
                ax.set_ylabel("Observed quantile", fontsize=6)
                ax.tick_params(labelsize=5)
                ax.legend(fontsize=5)

                poisson_stats[f"{pd_key}_{side}_{logic}"] = pid
            row += 1

    fig.tight_layout()
    out = OUTPUT_DIR / f"ana3{suffix}_poisson_qq.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[ana3] Poisson QQ saved: {out}")
    return poisson_stats




# ─────────────────────────────────────────────────────────────────
# (1b) 배경 추정 샘플 Poisson QQ
# ─────────────────────────────────────────────────────────────────
def plot_bg_qq(df_count: pd.DataFrame,
               events: list,
               logics: list[str],
               suffix: str = "") -> dict:
    """
    get_bg_median이 실제로 선택한 날짜의 count만 모아 Poisson QQ 작성.

    기존 plot_qq는 proxy < thresh 구간 전체(이벤트 오염 가능)를 사용.
    이 함수는 ana2 배경 추정에 실제로 쓰인 데이터(BG_WINDOW_DAYS 중
    가장 조용한 BG_QUIET_DAYS일)만 사용하므로 배경 순도를 검증할 수 있다.

    출력 파일명:
      ana3_bg_poisson_qq.png        (proton)
      ana3_electron_bg_poisson_qq.png (electron)
    """
    nrows = len(COUNT_PD_KEYS) * len(COUNT_SIDES)
    ncols = len(logics)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.5 * ncols, 3.2 * nrows), squeeze=False)
    label = "Electron" if suffix else "Proton"
    fig.suptitle(
        f"Poisson QQ  {label}  (BG-estimated samples only)\n"
        f"Lowe bg estimation: quietest {BG_QUIET_DAYS} days of {BG_WINDOW_DAYS}-day window",
        fontsize=11)

    bg_stats = {}
    row = 0
    for pd_key in COUNT_PD_KEYS:
        for side in COUNT_SIDES:
            for ci, logic in enumerate(logics):
                ax = axes[row][ci]
                try:
                    cnt_series = df_count[pd_key, side, logic]
                except KeyError:
                    ax.set_visible(False)
                    continue

                # ── 배경 추정에 실제로 쓰인 샘플 추출 ──────────────
                bg_samples = get_quiet_bg_samples(
                    cnt_series, events, BG_WINDOW_DAYS, BG_QUIET_DAYS)

                vals = bg_samples.dropna().values
                vals = vals[vals >= 0]
                key  = f"{pd_key}_{side}_{logic}"

                if len(vals) < 20:
                    ax.text(0.5, 0.5,
                            f"Insufficient\n(n={len(vals)})",
                            transform=ax.transAxes,
                            ha="center", va="center", fontsize=7, color="gray")
                    ax.set_title(f"{pd_key}-{side} / {logic}", fontsize=7)
                    bg_stats[key] = {"mean": np.nan, "var": np.nan,
                                     "ID": np.nan, "n": len(vals)}
                    continue

                lam   = float(np.mean(vals))
                var   = float(np.var(vals, ddof=1))
                ID    = var / lam if lam > 0 else np.nan
                probs = np.linspace(0.01, 0.99, 100)
                pq    = stats.poisson.ppf(probs, mu=lam)
                dq    = np.quantile(vals, probs)

                ax.scatter(pq, dq, s=7, alpha=0.6, color="#27ae60")
                lim = max(float(pq[-1]), float(dq[-1])) * 1.05
                ax.plot([0, lim], [0, lim], "r--", lw=1, label="y=x")

                id_str = f"{ID:.2f}" if np.isfinite(ID) else "N/A"
                ax.set_title(
                    f"{pd_key}-{side} / {logic}\n"
                    f"ID={id_str}  λ={lam:.1f}  n={len(vals):,}",
                    fontsize=7)
                ax.set_xlabel("Poisson Theoretical quantile", fontsize=6)
                ax.set_ylabel("Observed quantile", fontsize=6)
                ax.tick_params(labelsize=5)
                ax.legend(fontsize=5)

                bg_stats[key] = {"mean": lam, "var": var, "ID": ID, "n": int(len(vals))}
            row += 1

    fig.tight_layout()
    out = OUTPUT_DIR / f"ana3{suffix}_bg_poisson_qq.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[ana3] BG Poisson QQ saved: {out}")
    return bg_stats

# ─────────────────────────────────────────────────────────────────
# (2) Spike 분석
# ─────────────────────────────────────────────────────────────────
def spike_analysis(df_count: pd.DataFrame,
                   quiet_mask: pd.Series,
                   logics: list[str]) -> dict:
    """
    Quiet 기간 중 spike 탐지.
    spike: count > SPIKE_MULTIPLIER × ROLLING_WINDOW분 이동 중앙값
    A/B 동시 spike 일치율: AND gate FAR 억제 추정에 사용.
    logics로 proton/electron 공용.
    """
    out = {}
    for pd_key in COUNT_PD_KEYS:
        for logic in logics:
            spA = spB = None
            for side in COUNT_SIDES:
                try:
                    s = df_count[pd_key, side, logic][quiet_mask].dropna()
                except KeyError:
                    continue
                rol   = s.rolling(ROLLING_WINDOW, center=True, min_periods=5).median()
                is_sp = s > SPIKE_MULTIPLIER * rol.replace(0, np.nan)
                rate  = float(is_sp.sum() / len(is_sp)) if len(is_sp) else np.nan
                out[f"{pd_key}_{side}_{logic}"] = {
                    "spike_rate": round(rate, 6), "n_quiet": len(is_sp)}
                if side == "A":
                    spA = is_sp.reindex(df_count.index, fill_value=False)
                else:
                    spB = is_sp.reindex(df_count.index, fill_value=False)

            if spA is not None and spB is not None:
                both   = (spA & spB).sum()
                either = (spA | spB).sum()
                coinc  = float(both / either) if either > 0 else 0.0
                out[f"{pd_key}_AB_{logic}_coincidence"] = {
                    "AB_coincidence_rate": round(coinc, 5),
                    "n_both": int(both), "n_either": int(either)}
    return out


def plot_spike_summary(spike_stats: dict, suffix: str = ""):
    """suffix="" → proton / suffix="_electron" → electron"""
    rate_d  = {k: v["spike_rate"] for k, v in spike_stats.items()
               if "spike_rate" in v}
    coinc_d = {k: v["AB_coincidence_rate"] for k, v in spike_stats.items()
               if "AB_coincidence_rate" in v}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    label = "Electron" if suffix else "Proton"

    ax1.bar(range(len(rate_d)), list(rate_d.values()), color="#e74c3c", alpha=0.75)
    ax1.set_xticks(range(len(rate_d)))
    ax1.set_xticklabels(list(rate_d.keys()), rotation=55, ha="right", fontsize=7)
    ax1.set_ylabel("Spike rate", fontsize=9)
    ax1.set_title(
        f"{label} Spike Rate  "
        f"(>{SPIKE_MULTIPLIER}× rolling-{ROLLING_WINDOW}min median, Quiet period)",
        fontsize=10)
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.bar(range(len(coinc_d)), list(coinc_d.values()), color="#3498db", alpha=0.75)
    ax2.set_xticks(range(len(coinc_d)))
    ax2.set_xticklabels(list(coinc_d.keys()), rotation=40, ha="right", fontsize=8)
    ax2.set_ylabel("A/B Coincidence Rate", fontsize=9)
    ax2.set_title(f"{label} A/B Coincidence Rate  (AND gate FAR reduction estimate)",
                  fontsize=10)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out = OUTPUT_DIR / f"ana3{suffix}_spike_analysis.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[ana3] Spike analysis saved: {out}")


# ─────────────────────────────────────────────────────────────────
# (3) CR 채널 상관성
# ─────────────────────────────────────────────────────────────────
def plot_cr_correlation(df_count: pd.DataFrame,
                        quiet_mask: pd.Series,
                        logics: list[str],
                        suffix: str = ""):
    """
    CR(Cosmic Ray) 채널 vs count Pearson 상관.
    CR 검출 한계(65535 = 16bit 최대값) 초과 포인트 제거.
    r이 높으면 CR 채널을 veto 채널로 활용 가능.
    suffix="" → proton / suffix="_electron" → electron
    """
    nrows = len(COUNT_PD_KEYS)
    ncols = len(logics)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)
    label = "Electron" if suffix else "Proton"
    fig.suptitle(f"CR channel vs {label} Count  (Quiet period)\n"
                 "high Pearson r → CR usable as veto channel", fontsize=11)

    for ri, pd_key in enumerate(COUNT_PD_KEYS):
        for ci, logic in enumerate(logics):
            ax = axes[ri][ci]
            for side in COUNT_SIDES:
                try:
                    cr  = df_count[pd_key, side, "CR"][quiet_mask].dropna()
                    cnt = df_count[pd_key, side, logic][quiet_mask].dropna()
                except KeyError:
                    continue
                idx = cr.index.intersection(cnt.index)
                if len(idx) < 20:
                    continue
                x, y  = cr.loc[idx].values, cnt.loc[idx].values
                valid = x <= 65535   # CR 16bit 검출 한계 초과 제거
                x, y  = x[valid], y[valid]

                ax.scatter(x, y, s=3, alpha=0.2, rasterized=True, label=side)
                r, _ = stats.pearsonr(x, y) if len(x) > 5 else (np.nan, np.nan)
                ofs = 0.0 if side == "A" else 0.1
                ax.text(0.97, 0.97 - ofs, f"{side}: r={r:.2f}",
                        transform=ax.transAxes, fontsize=6, ha="right", va="top")
            ax.set_xlabel("CR count", fontsize=7)
            ax.set_ylabel(f"{logic} count", fontsize=7)
            ax.set_title(f"{pd_key} / {logic}", fontsize=8)
            ax.tick_params(labelsize=6)
            ax.legend(fontsize=6)

    fig.tight_layout()
    out = OUTPUT_DIR / f"ana3{suffix}_cr_correlation.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[ana3] CR correlation saved: {out}")


# ─────────────────────────────────────────────────────────────────
def save_noise_stats(poisson_stats: dict, spike_stats: dict,
                     thresh: float | None, suffix: str = "",
                     bg_poisson_stats: dict | None = None):
    label = "e_spe_proxy_thresh" if suffix else "spe_proxy_thresh"
    out_json = OUTPUT_DIR / f"ana3{suffix}_noise_stats.json"
    payload = {"poisson":    poisson_stats,
               "spike":      spike_stats,
               "config":     {"spike_multiplier":   SPIKE_MULTIPLIER,
                              "rolling_window_min": ROLLING_WINDOW,
                              "bg_window_days":     BG_WINDOW_DAYS,
                              "bg_quiet_days":      BG_QUIET_DAYS,
                              label:                thresh}}
    if bg_poisson_stats is not None:
        payload["bg_poisson"] = bg_poisson_stats
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[ana3] noise stats saved: {out_json}")

    print(f"\n[ana3] Poisson ID summary ({('electron' if suffix else 'proton')}):")
    for k, v in poisson_stats.items():
        id_s = f'{v["ID"]:.2f}' if np.isfinite(v.get("ID", np.nan)) else "N/A"
        print(f"  {k:25s}  ID={id_s}  n={v.get('n', 0):,}")


# ─────────────────────────────────────────────────────────────────
def main():
    print("[ana3] Loading data...")
    df_count, df_flux_p, df_flux_e = load_data()

    # ── Proxy 생성 & 이벤트 탐지 ─────────────────────────────────
    proxy_p  = make_proton_proxy(df_flux_p)
    proxy_e  = make_electron_proxy(df_flux_e)
    events_p = detect_proton_events(proxy_p)
    events_e = detect_electron_events(proxy_e)

    # ── Quiet 마스크 ──────────────────────────────────────────────
    quiet_p = get_quiet_mask(df_count, proxy_p, SPE_PROXY_THRESH,   "proton")
    quiet_e = get_quiet_mask(df_count, proxy_e, E_SPE_PROXY_THRESH, "electron")

    # ── Proton 분석 ───────────────────────────────────────────────
    # (1a) 기존 Quiet 전체 구간 Poisson QQ
    poisson_p = plot_qq(df_count, quiet_p, COUNT_PROTON_LOGICS, suffix="")
    # (1b) 배경 추정 실제 사용 샘플 Poisson QQ (배경 순도 검증)
    bg_p      = plot_bg_qq(df_count, events_p, COUNT_PROTON_LOGICS, suffix="")

    spike_p = spike_analysis(df_count, quiet_p, COUNT_PROTON_LOGICS)
    plot_spike_summary(spike_p, suffix="")
    plot_cr_correlation(df_count, quiet_p, COUNT_PROTON_LOGICS, suffix="")
    save_noise_stats(poisson_p, spike_p, SPE_PROXY_THRESH, suffix="",
                     bg_poisson_stats=bg_p)

    # ── Electron 분석 ─────────────────────────────────────────────
    # (1a) 기존 Quiet 전체 구간 Poisson QQ
    poisson_e = plot_qq(df_count, quiet_e, COUNT_ELECTRON_LOGICS, suffix="_electron")
    # (1b) 배경 추정 실제 사용 샘플 Poisson QQ
    bg_e      = plot_bg_qq(df_count, events_e, COUNT_ELECTRON_LOGICS, suffix="_electron")

    spike_e = spike_analysis(df_count, quiet_e, COUNT_ELECTRON_LOGICS)
    plot_spike_summary(spike_e, suffix="_electron")
    plot_cr_correlation(df_count, quiet_e, COUNT_ELECTRON_LOGICS, suffix="_electron")
    save_noise_stats(poisson_e, spike_e, E_SPE_PROXY_THRESH, suffix="_electron",
                     bg_poisson_stats=bg_e)


if __name__ == "__main__":
    main()