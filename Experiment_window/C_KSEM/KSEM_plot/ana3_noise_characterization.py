"""
ana3_noise_characterization.py
================================
Quiet period 노이즈 특성 분석

Spike vs Onset 구분:
  spike : Quiet 기간 중 갑작스럽게 튀는 비정상 데이터 포인트.
          60분 이동 중앙값의 SPIKE_MULTIPLIER배 초과로 정의.
          원인: SEU(Single Event Upset), EMI 등 계측 노이즈.
  onset : SPE 이벤트 시작점 (ana2·ana4에서 별도 정의).
          물리적 입자 증가이며 최소 MIN_SPE_DURATION_H 이상 지속됨.

전자(electron) 채널:
  배경 추세(ana3_background_trend.png)에서만 사용.
  SPE 판정은 proton proxy만 사용.

출력:
  ana3_poisson_qq.png
    Poisson QQ: y=x 이탈 → 과분산(overdispersed). ID=Var/Mean 표시.
    ID≈1: 순수 랜덤 노이즈 / ID≫1: 클러스터링 또는 다른 신호 혼입.
  ana3_background_trend.png
    월별 median flux — 전자(E1-E6)와 양성자(E8-E10) 채널 분리.
  ana3_spike_analysis.png
    상단: spike rate 막대 (SPIKE_MULTIPLIER × rolling median 초과 비율)
    하단: A/B 동시 spike 일치율 → AND gate FAR 억제 효과 추정
  ana3_cr_correlation.png
    CR 채널 vs O/OU/OUT count 산점도 (Pearson r)
    r이 높으면 CR을 veto 채널로 활용 가능
  ana3_count_noise_stats.json

튜닝값 (이 파일 단독):
  SPIKE_MULTIPLIER : 이동 중앙값의 이 배를 초과하면 spike로 판정.
                     낮추면 민감도 상승(false spike 증가), 높이면 보수적.
  ROLLING_WINDOW   : spike 판정용 이동 중앙값 윈도우 [분].
                     짧으면 지역 변동에 민감, 길면 완만한 상승을 spike로 미인정.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import ksem_flux_config as cfg
from ksem_flux_config import (
    COUNT_PARQUET_DIR, FLUX_PARQUET_DIR, OUTPUT_DIR,
    PROTON_CHANNELS, ELECTRON_CHANNELS,
    SPE_PROXY_CHANNELS, SPE_PROXY_THRESH,
    ELECTRON_BG_CHANNELS, PROTON_BG_CHANNELS,
    COUNT_PROTON_LOGICS, COUNT_PD_KEYS, COUNT_SIDES,
)
import ksem_io
import kma_ksem_flux_io

# ── ana3 단독 튜닝값 ──────────────────────────────────────────────
SPIKE_MULTIPLIER = 5.0   # rolling median의 이 배를 초과하면 spike
ROLLING_WINDOW   = 60    # [분] spike 판정용 이동 중앙값 윈도우


# ─────────────────────────────────────────────────────────────────
def get_quiet_mask(df_count: pd.DataFrame,
                   df_flux_proton: pd.DataFrame) -> pd.Series:
    """count 인덱스 기준 quiet 마스크. proxy < SPE_PROXY_THRESH 이면 True."""
    proxy = pd.Series(0.0, index=df_flux_proton.index)
    for ch in SPE_PROXY_CHANNELS:
        if ch in df_flux_proton.columns:
            lo, hi = PROTON_CHANNELS[ch]
            proxy += df_flux_proton[ch].fillna(0) * (hi - lo)
    proxy[proxy == 0] = np.nan
    proxy_aligned = proxy.reindex(df_count.index, method="nearest",
                                  tolerance="5min")
    mask = (proxy_aligned < SPE_PROXY_THRESH) | proxy_aligned.isna()
    print(f"[ana3] Quiet: {mask.sum():,} / {len(mask):,} points")
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


def plot_qq(df_count: pd.DataFrame, quiet_mask: pd.Series):
    """
    Poisson QQ 플롯 (6행 × 3열).
    y=x 대각선 위로 벗어날수록 over-dispersed.
    """
    nrows = len(COUNT_PD_KEYS) * len(COUNT_SIDES)
    ncols = len(COUNT_PROTON_LOGICS)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.5 * ncols, 3.2 * nrows), squeeze=False)
    fig.suptitle("Poisson QQ  (Quiet period)\n"
                 "y=x: Poisson | above y=x: overdispersed (ID > 1)", fontsize=11)

    row = 0
    for pd_key in COUNT_PD_KEYS:
        for side in COUNT_SIDES:
            for ci, logic in enumerate(COUNT_PROTON_LOGICS):
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
            row += 1

    fig.tight_layout()
    out = OUTPUT_DIR / "ana3_poisson_qq.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[ana3] Poisson QQ saved: {out}")


# ─────────────────────────────────────────────────────────────────
# (2) 배경 장기 추세 — 전자/양성자 분리
# ─────────────────────────────────────────────────────────────────
def plot_background_trend(df_flux_electron: pd.DataFrame,
                          df_flux_proton:   pd.DataFrame,
                          quiet_mask_flux:  pd.Series):
    """
    월별 median flux 추세 (Quiet 기간만).
    상단: 전자(E1~E6) / 하단: 양성자(E8~E10) — 각각 채널별 선.
    SC25 태양활동 상승 추세 확인용.
    """
    fig, (ax_e, ax_p) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    cmap_e = plt.cm.Blues_r
    cmap_p = plt.cm.Oranges_r

    # 전자
    quiet_mask_e = (quiet_mask_flux
                    .reindex(df_flux_electron.index, method="nearest",
                             tolerance=pd.Timedelta("2min"))
                    .astype(float).fillna(1.0).astype(bool))
    has_e_data = False
    for i, ch in enumerate(ELECTRON_BG_CHANNELS):
        if ch not in df_flux_electron.columns:
            continue
        lo, hi = ELECTRON_CHANNELS.get(ch, (0, 0))
        s = df_flux_electron[ch][quiet_mask_e].dropna()
        s = s[s > 0]
        if s.empty:
            continue
        monthly = s.resample("1ME").median().dropna()
        if monthly.empty or (monthly <= 0).all():
            continue
        color = cmap_e(0.2 + 0.6 * i / max(len(ELECTRON_BG_CHANNELS) - 1, 1))
        ax_e.semilogy(monthly.index, monthly.values, lw=1.2, color=color,
                      label=f"{ch} ({lo}-{hi} keV)")
        has_e_data = True
    if not has_e_data:
        ax_e.text(0.5, 0.5, "No valid electron flux data in quiet period",
                  transform=ax_e.transAxes, ha="center", va="center")
    ax_e.set_ylabel("Electron flux median\n[cm-2 sr-1 s-1 keV-1]", fontsize=9)
    ax_e.set_title("Electron channels — Monthly Quiet background trend (SC25)", fontsize=10)
    if has_e_data:
        ax_e.legend(fontsize=7, ncol=3)
    ax_e.grid(True, which="both", alpha=0.3)

    # 양성자
    quiet_mask_p = (quiet_mask_flux
                    .reindex(df_flux_proton.index, method="nearest",
                             tolerance=pd.Timedelta("2min"))
                    .astype(float).fillna(1.0).astype(bool))
    has_p_data = False
    for i, ch in enumerate(PROTON_BG_CHANNELS):
        if ch not in df_flux_proton.columns:
            continue
        lo, hi = PROTON_CHANNELS.get(ch, (0, 0))
        s = df_flux_proton[ch][quiet_mask_p].dropna()
        s = s[s > 0]
        if s.empty:
            continue
        monthly = s.resample("1ME").median().dropna()
        if monthly.empty or (monthly <= 0).all():
            continue
        color = cmap_p(0.2 + 0.6 * i / max(len(PROTON_BG_CHANNELS) - 1, 1))
        ax_p.semilogy(monthly.index, monthly.values, lw=1.2, color=color,
                      label=f"{ch} ({lo//1000:.0f}-{hi//1000:.0f} MeV)")
        has_p_data = True
    if not has_p_data:
        ax_p.text(0.5, 0.5, "No valid proton flux data in quiet period",
                  transform=ax_p.transAxes, ha="center", va="center")
    ax_p.set_ylabel("Proton flux median\n[cm-2 sr-1 s-1 keV-1]", fontsize=9)
    ax_p.set_title("Proton channels — Monthly Quiet background trend (SC25)", fontsize=10)
    if has_p_data:
        ax_p.legend(fontsize=7, ncol=3)
    ax_p.grid(True, which="both", alpha=0.3)
    ax_p.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_p.tick_params(axis="x", rotation=30, labelsize=7)

    fig.suptitle("Quiet background trend — Electron vs Proton", fontsize=12)
    fig.tight_layout()
    out = OUTPUT_DIR / "ana3_background_trend.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[ana3] background trend saved: {out}")


# ─────────────────────────────────────────────────────────────────
# (3) Spike 분석
# ─────────────────────────────────────────────────────────────────
def spike_analysis(df_count: pd.DataFrame,
                   quiet_mask: pd.Series) -> dict:
    """
    Quiet 기간 중 spike 탐지.
    spike 정의: count > SPIKE_MULTIPLIER × ROLLING_WINDOW분 이동 중앙값
    A/B 동시 spike 일치율: AND gate 적용 시 FAR(오경보율) 억제 추정에 사용.
    """
    out = {}
    for pd_key in COUNT_PD_KEYS:
        for logic in COUNT_PROTON_LOGICS:
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


def plot_spike_summary(spike_stats: dict):
    rate_d  = {k: v["spike_rate"] for k, v in spike_stats.items()
               if "spike_rate" in v}
    coinc_d = {k: v["AB_coincidence_rate"] for k, v in spike_stats.items()
               if "AB_coincidence_rate" in v}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    ax1.bar(range(len(rate_d)), list(rate_d.values()), color="#e74c3c", alpha=0.75)
    ax1.set_xticks(range(len(rate_d)))
    ax1.set_xticklabels(list(rate_d.keys()), rotation=55, ha="right", fontsize=7)
    ax1.set_ylabel("Spike rate", fontsize=9)
    ax1.set_title(
        f"Spike Rate  (>{SPIKE_MULTIPLIER}× rolling-{ROLLING_WINDOW}min median, Quiet period)",
        fontsize=10)
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.bar(range(len(coinc_d)), list(coinc_d.values()), color="#3498db", alpha=0.75)
    ax2.set_xticks(range(len(coinc_d)))
    ax2.set_xticklabels(list(coinc_d.keys()), rotation=40, ha="right", fontsize=8)
    ax2.set_ylabel("A/B Coincidence Rate", fontsize=9)
    ax2.set_title("A/B Coincidence Rate  (AND gate FAR reduction estimate)", fontsize=10)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out = OUTPUT_DIR / "ana3_spike_analysis.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[ana3] Spike analysis saved: {out}")


# ─────────────────────────────────────────────────────────────────
# (4) CR 채널 상관성
# ─────────────────────────────────────────────────────────────────
def plot_cr_correlation(df_count: pd.DataFrame, quiet_mask: pd.Series):
    """
    CR(Cosmic Ray) 채널 vs proton count (O/OU/OUT) Pearson 상관.
    r이 높으면 CR 채널을 veto 채널로 활용하여 배경 노이즈를 줄일 수 있음.
    """
    nrows = len(COUNT_PD_KEYS)
    ncols = len(COUNT_PROTON_LOGICS)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)
    fig.suptitle("CR channel vs Proton Count  (Quiet period)\n"
                 "high Pearson r → CR usable as veto channel", fontsize=11)

    for ri, pd_key in enumerate(COUNT_PD_KEYS):
        for ci, logic in enumerate(COUNT_PROTON_LOGICS):
            ax = axes[ri][ci]
            for side in COUNT_SIDES:
                try:
                    cr  = df_count[pd_key, side, "CR"][quiet_mask].dropna()
                    pro = df_count[pd_key, side, logic][quiet_mask].dropna()
                except KeyError:
                    continue
                idx = cr.index.intersection(pro.index)
                if len(idx) < 20:
                    continue
                x, y = cr.loc[idx].values, pro.loc[idx].values
                valid = x <= 65535      # 추가: CR 검출 한계(65535 = 16bit 최대값) 초과 포인트 제거
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
    out = OUTPUT_DIR / "ana3_cr_correlation.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[ana3] CR correlation saved: {out}")


# ─────────────────────────────────────────────────────────────────
def main():
    print("[ana3] Loading data...")
    df_count, _ = ksem_io.load(COUNT_PARQUET_DIR)
    sensor_data, _ = kma_ksem_flux_io.load(FLUX_PARQUET_DIR)
    df_flux_p = sensor_data.get("proton",   pd.DataFrame())
    df_flux_e = sensor_data.get("electron", pd.DataFrame())

    for df in [df_count, df_flux_p, df_flux_e]:
        if not df.empty and df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

    quiet_count = get_quiet_mask(df_count, df_flux_p)

    # flux 인덱스 기준 quiet 마스크 (배경 추세 플롯용)
    proxy_flux = pd.Series(0.0, index=df_flux_p.index)
    for ch in SPE_PROXY_CHANNELS:
        if ch in df_flux_p.columns:
            lo, hi = PROTON_CHANNELS[ch]
            proxy_flux += df_flux_p[ch].fillna(0) * (hi - lo)
    proxy_flux[proxy_flux == 0] = np.nan
    quiet_flux = (proxy_flux < SPE_PROXY_THRESH) | proxy_flux.isna()

    plot_qq(df_count, quiet_count)
    plot_background_trend(df_flux_e, df_flux_p, quiet_flux)

    spike_stats = spike_analysis(df_count, quiet_count)
    plot_spike_summary(spike_stats)

    plot_cr_correlation(df_count, quiet_count)

    # Poisson ID 집계
    poisson_stats = {}
    for pd_key in COUNT_PD_KEYS:
        for side in COUNT_SIDES:
            for logic in COUNT_PROTON_LOGICS:
                try:
                    s = df_count[pd_key, side, logic][quiet_count]
                except KeyError:
                    continue
                poisson_stats[f"{pd_key}_{side}_{logic}"] = poisson_id(s)

    out_json = OUTPUT_DIR / "ana3_count_noise_stats.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"poisson": poisson_stats,
                   "spike":   spike_stats,
                   "config":  {"spike_multiplier":    SPIKE_MULTIPLIER,
                               "rolling_window_min":  ROLLING_WINDOW,
                               "spe_proxy_thresh":    SPE_PROXY_THRESH}},
                  f, indent=2, ensure_ascii=False)
    print(f"[ana3] noise stats saved: {out_json}")

    print("\n[ana3] Poisson ID summary:")
    for k, v in poisson_stats.items():
        id_s = f'{v["ID"]:.2f}' if np.isfinite(v.get("ID", np.nan)) else "N/A"
        print(f"  {k:25s}  ID={id_s}  n={v.get('n', 0):,}")


if __name__ == "__main__":
    main()
