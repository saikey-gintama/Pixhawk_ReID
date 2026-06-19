"""
ana1_count_flux_scatter.py
==========================
Count-Flux 상관관계 분석

목적:
  KSEM count와 flux를 15분 리샘플 후 매칭하여
  멱함수 "count = a * flux^b" 관계를 도출한다.

  proton: count(O/OU/OUT) × proxy_p(E8+E9+E10 ΔE-적분)
  electron: count(F/FT/FTU/FTUO) × proxy_e(E4+E5+E6 ΔE-적분)

출력:
  ana_output/ana1_count_flux_scatter_proxy_p.png
  ana_output/ana1_count_flux_scatter_proxy_e.png
  ana_output/ana1_count_flux_fit.csv   ← proton/electron 행 모두 포함, flux_label로 구분

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
    OUTPUT_DIR,
    SPE_PROXY_THRESH, SPE_PROXY_LABEL,
    E_SPE_PROXY_THRESH, E_SPE_PROXY_LABEL,
    COUNT_PROTON_LOGICS, COUNT_ELECTRON_LOGICS, COUNT_PD_KEYS, COUNT_SIDES,
)
from ksem_common import (
    load_data,
    make_proton_proxy,
    make_electron_proxy,
)

# ── ana1 단독 튜닝값 ──────────────────────────────────────────────
RESAMPLE_FREQ = "15min"
TS_MAX        = 5000    # Theil-Sen 서브샘플 상한 (O(n²) 메모리 제약)


# ─────────────────────────────────────────────────────────────────
def load_and_align():
    """count / flux 로드 후 15분 리샘플 & 시간 인덱스 정렬."""
    print("[ana1] Loading data...")
    df_count, df_flux_p, df_flux_e = load_data()

    cnt_rs    = df_count.resample(RESAMPLE_FREQ).mean()
    flux_p_rs = df_flux_p.resample(RESAMPLE_FREQ).mean()
    flux_e_rs = df_flux_e.resample(RESAMPLE_FREQ).mean()

    common_p = cnt_rs.index.intersection(flux_p_rs.index)
    common_e = cnt_rs.index.intersection(flux_e_rs.index)

    return (cnt_rs.loc[common_p], flux_p_rs.loc[common_p],
            cnt_rs.loc[common_e], flux_e_rs.loc[common_e])


def spe_mask_from(flux_s: pd.Series, thresh: float) -> pd.Series:
    """flux_s >= thresh 인 시점을 SPE/E_SPE로 분류."""
    if thresh is None:
        return pd.Series(False, index=flux_s.index)
    return flux_s >= thresh


# ─────────────────────────────────────────────────────────────────
def power_fit(x: np.ndarray, y: np.ndarray) -> dict:
    """
    로그공간 OLS + Theil-Sen으로 y = a * x^b 멱함수 피팅.

    OLS       : 표준 최소제곱법. 이상치에 민감하나 전체 분포를 반영.
    Theil-Sen : 이상치 강건 추정. O(n²) 메모리로 TS_MAX개 구간 균등 샘플 사용.

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
                      logics: list[str],
                      flux_label: str,
                      flux_unit: str,
                      thresh: float | None) -> tuple[plt.Figure, list[dict]]:
    """
    6행(PD1-A/B, PD2-A/B, PD3-A/B) × len(logics)열 산점도 그리드.
    색: Quiet=회색, SPE/E_SPE=오렌지
    피팅선: OLS=파랑실선, Theil-Sen=빨강점선
    """
    nrows = len(COUNT_PD_KEYS) * len(COUNT_SIDES)
    ncols = len(logics)

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.5 * ncols, 3.2 * nrows),
                              squeeze=False)
    thresh_str = f"{thresh:.1e}" if thresh is not None else "N/A"
    fig.suptitle(
        f"KSEM Count vs Flux  [{flux_label}]\n"
        f"15-min avg | Quiet=gray / Event=orange | threshold: {thresh_str} {flux_unit}",
        fontsize=11, y=1.01)

    records = []
    row = 0
    for pd_key in COUNT_PD_KEYS:
        for side in COUNT_SIDES:
            for col, logic in enumerate(logics):
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
                           label="Event", rasterized=True)

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
    cnt_p_rs, flux_p_rs, cnt_e_rs, flux_e_rs = load_and_align()
    all_records = []

    # ── Proton: proxy_p × proton count ───────────────────────────
    proxy_p = make_proton_proxy(flux_p_rs)
    is_spe  = spe_mask_from(proxy_p, SPE_PROXY_THRESH)
    fig_p, rec_p = draw_scatter_grid(
        cnt_p_rs, proxy_p, is_spe,
        logics=COUNT_PROTON_LOGICS,
        flux_label=SPE_PROXY_LABEL,
        flux_unit="cm-2 sr-1 s-1  (dE integral)",
        thresh=SPE_PROXY_THRESH)
    out_p = OUTPUT_DIR / "ana1_count_flux_scatter_proxy_p.png"
    fig_p.savefig(out_p, dpi=150, bbox_inches="tight")
    plt.close(fig_p)
    print(f"[ana1] Proton proxy 산점도 saved: {out_p}")
    all_records.extend(rec_p)

    # ── Electron: proxy_e × electron count ───────────────────────
    proxy_e   = make_electron_proxy(flux_e_rs)
    is_e_spe  = spe_mask_from(proxy_e, E_SPE_PROXY_THRESH)
    fig_e, rec_e = draw_scatter_grid(
        cnt_e_rs, proxy_e, is_e_spe,
        logics=COUNT_ELECTRON_LOGICS,
        flux_label=E_SPE_PROXY_LABEL,
        flux_unit="cm-2 sr-1 s-1  (dE integral)",
        thresh=E_SPE_PROXY_THRESH)
    out_e = OUTPUT_DIR / "ana1_count_flux_scatter_proxy_e.png"
    fig_e.savefig(out_e, dpi=150, bbox_inches="tight")
    plt.close(fig_e)
    print(f"[ana1] Electron proxy 산점도 saved: {out_e}")
    all_records.extend(rec_e)

    # ── CSV (proton + electron 통합) ──────────────────────────────
    df_fit  = pd.DataFrame(all_records)
    out_csv = OUTPUT_DIR / "ana1_count_flux_fit.csv"
    df_fit.to_csv(out_csv, index=False, float_format="%.6g")
    print(f"[ana1] fit results saved: {out_csv}")
    print(df_fit.to_string(index=False))


if __name__ == "__main__":
    main()
