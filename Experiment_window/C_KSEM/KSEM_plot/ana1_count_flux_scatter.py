"""
ana1_count_flux_scatter.py
==========================
Count-Flux 상관관계 분석

목적:
  같은 시각의 KSEM count와 proton flux를 15분 리샘플 후 매칭하여
  멱함수 "count = a * flux^b" 관계를 도출한다.

  E10 단독(3.1-6.0 MeV)과 E8+E9+E10 ΔE-적분 합산(proxy) 두 가지로 분석.

출력:
  ana_output/ana1_count_flux_scatter_E10.png
  ana_output/ana1_count_flux_scatter_proxy.png
  ana_output/ana1_count_flux_fit.csv   ← PD/side/logic별 a, b, R², RMSE

튜닝값 (이 파일 단독):
  RESAMPLE_FREQ : count·flux 평균 윈도우. 좁히면 노이즈 증가, 넓히면 시간해상도 저하.
  TS_MAX        : Theil-Sen 서브샘플 크기. O(n²) 메모리 제한으로 전체 사용 불가.
                  늘리면 추정 정확도 상승, 메모리/시간 비례 증가.
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
    COUNT_PARQUET_DIR, FLUX_PARQUET_DIR, OUTPUT_DIR,
    SPE_PROXY_CHANNELS, SPE_PROXY_THRESH, SPE_PROXY_LABEL,
    SPE_SINGLE_CHANNEL, PROTON_CHANNELS,
    COUNT_PROTON_LOGICS, COUNT_PD_KEYS, COUNT_SIDES,
)
import ksem_io
import kma_ksem_flux_io

# ── ana1 단독 튜닝값 ──────────────────────────────────────────────
RESAMPLE_FREQ = "15min"
TS_MAX        = 5000    # Theil-Sen 서브샘플 상한 (O(n²) 메모리 제약)


# ─────────────────────────────────────────────────────────────────
def load_and_align():
    """count / flux 로드 후 15분 리샘플 & 시간 인덱스 정렬."""
    print("[ana1] Loading data...")
    df_count, _    = ksem_io.load(COUNT_PARQUET_DIR)
    sensor_data, _ = kma_ksem_flux_io.load(FLUX_PARQUET_DIR)
    df_flux        = sensor_data.get("proton", pd.DataFrame())

    for df in [df_count, df_flux]:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

    cnt_rs  = df_count.resample(RESAMPLE_FREQ).mean()
    flux_rs = df_flux.resample(RESAMPLE_FREQ).mean()

    common = cnt_rs.index.intersection(flux_rs.index)
    return cnt_rs.loc[common], flux_rs.loc[common]


def make_proxy(flux_rs: pd.DataFrame) -> pd.Series:
    """
    E8+E9+E10 ΔE-적분 합산 proxy.
    단위: cm-2 sr-1 s-1  (flux [cm-2 sr-1 s-1 keV-1] × ΔE [keV])
    단순 합산 대신 채널폭 가중 적분을 사용해 에너지 범위 차이를 보정.
    """
    proxy = pd.Series(0.0, index=flux_rs.index)
    for ch in SPE_PROXY_CHANNELS:
        if ch not in flux_rs.columns:
            print(f"  [ana1] warning: proxy 채널 {ch} 없음, 스킵")
            continue
        lo, hi = PROTON_CHANNELS[ch]
        proxy += flux_rs[ch].fillna(0) * (hi - lo)
    proxy[proxy == 0] = np.nan
    return proxy


def spe_mask_from(flux_s: pd.Series) -> pd.Series:
    """flux_s >= SPE_PROXY_THRESH 인 시점을 SPE로 분류."""
    return flux_s >= SPE_PROXY_THRESH


# ─────────────────────────────────────────────────────────────────
def power_fit(x: np.ndarray, y: np.ndarray) -> dict:
    """
    로그공간 OLS + Theil-Sen으로 y = a * x^b 멱함수 피팅.

    OLS  : 표준 최소제곱법. 이상치에 민감하나 전체 분포를 반영.
    Theil-Sen : 이상치 강건 추정. O(n²) 메모리로 TS_MAX개 서브샘플 사용.

    반환: a_ols, b_ols, a_ts, b_ts, r2(OLS), rmse(OLS, 원단위), n
    """
    valid = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    n = int(valid.sum())
    nan_result = dict(a_ols=np.nan, b_ols=np.nan,
                      a_ts=np.nan,  b_ts=np.nan,
                      r2=np.nan, rmse=np.nan, n=n)
    if n < 5:
        return nan_result

    lx = np.log10(x[valid])
    ly = np.log10(y[valid])

    slope, intercept, r, *_ = stats.linregress(lx, ly)
    a_ols, b_ols = 10 ** intercept, slope

    # 전체를 TS_MAX개 구간으로 나눠 각 구간에서 1개씩 균등 추출
    if len(lx) > TS_MAX:
        bins = np.array_split(np.arange(len(lx)), TS_MAX)
        rng  = np.random.default_rng(42)
        idx  = np.array([rng.choice(b) for b in bins if len(b)])
        lx_ts, ly_ts = lx[idx], ly[idx]
    else:
        lx_ts, ly_ts = lx, ly
    ts = stats.theilslopes(ly_ts, lx_ts)
    a_ts, b_ts = 10 ** ts.intercept, ts.slope

    y_pred = a_ols * x[valid] ** b_ols
    rmse   = float(np.sqrt(np.mean((y[valid] - y_pred) ** 2)))

    return dict(a_ols=a_ols, b_ols=b_ols,
                a_ts=a_ts,   b_ts=b_ts,
                r2=float(r**2), rmse=rmse, n=n)


# ─────────────────────────────────────────────────────────────────
def draw_scatter_grid(cnt_rs: pd.DataFrame,
                      flux_s: pd.Series,
                      is_spe: pd.Series,
                      flux_label: str,
                      flux_unit: str) -> tuple[plt.Figure, list[dict]]:
    """
    6행(PD1-A/B, PD2-A/B, PD3-A/B) × 3열(O, OU, OUT) 산점도 그리드.
    색: Quiet=회색, SPE=오렌지 / 피팅선: OLS=파랑실선, Theil-Sen=빨강점선.
    """
    nrows = len(COUNT_PD_KEYS) * len(COUNT_SIDES)
    ncols = len(COUNT_PROTON_LOGICS)

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.5 * ncols, 3.2 * nrows),
                              squeeze=False)
    fig.suptitle(
        f"KSEM Count vs Proton Flux  [{flux_label}]\n"
        f"15-min avg | Quiet=gray / SPE=orange | SPE: flux >= {SPE_PROXY_THRESH:.1e} {flux_unit}",
        fontsize=11, y=1.01)

    records = []
    row = 0
    for pd_key in COUNT_PD_KEYS:
        for side in COUNT_SIDES:
            for col, logic in enumerate(COUNT_PROTON_LOGICS):
                ax = axes[row][col]

                try:
                    cnt_s = cnt_rs[pd_key, side, logic]
                except KeyError:
                    ax.set_visible(False)
                    continue

                valid = cnt_s.notna() & flux_s.notna()
                x_all = flux_s[valid].values
                y_all = cnt_s[valid].values
                spe_v = is_spe[valid].values

                if len(x_all) == 0 or (x_all > 0).sum() == 0:
                    ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                            ha="center", va="center", fontsize=9, color="gray")
                    ax.set_title(f"{pd_key}-{side} / {logic}", fontsize=8)
                    continue

                ax.scatter(x_all[~spe_v], y_all[~spe_v],
                           s=3, alpha=0.25, color="#7f8c8d",
                           label="Quiet", rasterized=True)
                ax.scatter(x_all[spe_v],  y_all[spe_v],
                           s=7, alpha=0.65, color="#e67e22",
                           label="SPE",   rasterized=True)

                fit = power_fit(x_all, y_all)
                if np.isfinite(fit["a_ols"]):
                    x_lo  = max(x_all[x_all > 0].min(), 1e-6)
                    xline = np.logspace(np.log10(x_lo), np.log10(x_all.max()), 200)
                    ax.plot(xline, fit["a_ols"] * xline ** fit["b_ols"],
                            "b-", lw=1.2,
                            label=f'OLS a={fit["a_ols"]:.2e} b={fit["b_ols"]:.2f}')
                    ax.plot(xline, fit["a_ts"] * xline ** fit["b_ts"],
                            "r--", lw=1.0,
                            label=f'TS  a={fit["a_ts"]:.2e} b={fit["b_ts"]:.2f}')

                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_title(f"{pd_key}-{side} / {logic}", fontsize=8)
                ax.set_xlabel(f"Flux  ({flux_unit})", fontsize=7)
                ax.set_ylabel("Count [15-min avg]", fontsize=7)
                ax.tick_params(labelsize=6)

                r2_str = f'R2={fit["r2"]:.3f}' if np.isfinite(fit["r2"]) else "R2=N/A"
                ax.text(0.97, 0.04, r2_str, transform=ax.transAxes,
                        fontsize=7, ha="right", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.2", fc="w", alpha=0.7))
                ax.legend(fontsize=5, loc="upper left")

                records.append({
                    "pd_key": pd_key, "side": side, "logic": logic,
                    "flux_label": flux_label,
                    **{k: (round(v, 8) if isinstance(v, float) else v)
                       for k, v in fit.items()},
                })

            row += 1

    fig.tight_layout()
    return fig, records


# ─────────────────────────────────────────────────────────────────
def main():
    cnt_rs, flux_rs = load_and_align()
    all_records = []

    # 그림1: E10 단독 채널
    if SPE_SINGLE_CHANNEL in flux_rs.columns:
        flux_E10   = flux_rs[SPE_SINGLE_CHANNEL]
        is_spe_E10 = spe_mask_from(flux_E10)
        fig1, rec1 = draw_scatter_grid(
            cnt_rs, flux_E10, is_spe_E10,
            flux_label=f"{SPE_SINGLE_CHANNEL} (3.1-6.0 MeV)",
            flux_unit="cm-2 sr-1 s-1 keV-1")
        out1 = OUTPUT_DIR / "ana1_count_flux_scatter_E10.png"
        fig1.savefig(out1, dpi=150, bbox_inches="tight")
        plt.close(fig1)
        print(f"[ana1] E10 산점도 saved: {out1}")
        all_records.extend(rec1)
    else:
        print(f"[ana1] {SPE_SINGLE_CHANNEL} 채널 없음, 스킵")

    # 그림2: E8+E9+E10 proxy
    proxy      = make_proxy(flux_rs)
    is_spe_px  = spe_mask_from(proxy)
    fig2, rec2 = draw_scatter_grid(
        cnt_rs, proxy, is_spe_px,
        flux_label=SPE_PROXY_LABEL,
        flux_unit="cm-2 sr-1 s-1  (dE integral)")
    out2 = OUTPUT_DIR / "ana1_count_flux_scatter_proxy.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"[ana1] Proxy 산점도 saved: {out2}")
    all_records.extend(rec2)

    # CSV
    df_fit  = pd.DataFrame(all_records)
    out_csv = OUTPUT_DIR / "ana1_count_flux_fit.csv"
    df_fit.to_csv(out_csv, index=False, float_format="%.6g")
    print(f"[ana1] fit results saved: {out_csv}")
    print(df_fit.to_string(index=False))


if __name__ == "__main__":
    main()
