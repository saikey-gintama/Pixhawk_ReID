"""
ana4_count_response_profile.py
================================
이벤트별 Count 응답 프로파일 분석

SPE/E_SPE 기준:
  proton : proxy_p (E8+E9+E10 ΔE-적분) >= SPE_PROXY_THRESH
  electron: proxy_e (E4+E5+E6 ΔE-적분) >= E_SPE_PROXY_THRESH
  E_SPE_PROXY_THRESH=None 이면 electron 블록 스킵.

로직 포함 방침:
  proton  : O / OU / OUT 모두 포함.
  electron: F / FT / FTU / FTUO 모두 포함.
  각자 독립적으로 분석. proton-electron 직접 비교는 ana5 담당.

PD 방향각 비교:
  (PD1/PD2), (PD1/PD3), (PD2/PD3) 세 쌍 비율 히스토그램.

출력 (proton):
  ana4_superposed_epoch.png
  ana4_rise_time_cdf.png
  ana4_onset_comparison.png
  ana4_logic_order.png
  ana4_pd_direction.png
  ana4_onset_delay_table.csv
  ana4_onset_delay_raw.csv

출력 (electron, E_SPE_PROXY_THRESH 설정 시):
  ana4_electron_superposed_epoch.png
  ana4_electron_rise_time_cdf.png
  ana4_electron_onset_comparison.png
  ana4_electron_logic_order.png
  ana4_electron_pd_direction.png
  ana4_electron_onset_delay_table.csv
  ana4_electron_onset_delay_raw.csv

튜닝값 (이 파일 단독):
  PRE_H  : superposed epoch 앞 범위 [h].
  POST_H : superposed epoch 뒤 범위 [h].
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ksem_flux_config as cfg
from ksem_flux_config import (
    OUTPUT_DIR,
    SPE_PROXY_THRESH, SPE_PROXY_LABEL,
    E_SPE_PROXY_THRESH, E_SPE_PROXY_LABEL,
    COUNT_PROTON_LOGICS, COUNT_ELECTRON_LOGICS, COUNT_PD_KEYS, COUNT_SIDES,
    BG_WINDOW_DAYS, BG_QUIET_DAYS, COUNT_BG_K,
)
from ksem_common import (
    SPEEvent,
    load_data,
    make_proton_proxy,
    make_electron_proxy,
    detect_proton_events,
    detect_electron_events,
    get_bg_threshold,
)

# ── ana4 단독 튜닝값 ──────────────────────────────────────────────
PRE_H  = 24   # [h] superposed epoch 앞 범위
POST_H = 48   # [h] superposed epoch 뒤 범위


# ─────────────────────────────────────────────────────────────────
def event_bg_threshold(cnt: pd.Series, ev: SPEEvent) -> float:
    """
    이벤트 onset 기준, 채널별 배경 임계값 = bg_median + COUNT_BG_K·σ_quiet.
    onset 이전 가장 조용한 BG_QUIET_DAYS일(BG_WINDOW_DAYS 윈도우 내)로 추정.
    config 변경 시 ana2·ana4에 자동 반영.
    """
    return get_bg_threshold(cnt, ev.onset,
                            BG_WINDOW_DAYS, BG_QUIET_DAYS, COUNT_BG_K)


def count_onset(cnt: pd.Series,
                ev: SPEEvent) -> tuple[pd.Timestamp | None, bool]:
    """
    onset 전 6h ~ peak 후 6h 범위에서 count가 배경 임계값(bg_median + kσ_quiet)을
    첫 초과하는 시점. 임계값은 이 이벤트 onset 기준으로 채널별로 동적 계산한다.

    반환: (시점, detected)
      detected=True  → count가 실제로 임계값을 넘은 "진짜 검출".
      detected=False → 한 번도 못 넘음. 시점은 참고용 max 위치(fallback)이며
                       delay 통계에는 넣지 말 것.
      (None, False)  → 구간에 데이터 자체가 없거나 배경 임계값 추정 불가.

    [임계값 변경 이력 — 중요]
      이전 버전은 임계값으로 "quiet 구간 count의 p70"을 썼다. 이는 두 가지로
      잘못이었다:
        1) p70은 검출 임계가 아니라 quiet 분포의 위치값일 뿐이다. 정의상
           quiet의 30%가 이미 p70을 넘으므로 SPE와 무관한 배경 요동에서도
           onset이 잡힌다.
        2) OU/OUT처럼 quiet median<1인 채널은 p70=0~1이 되어, 단 1카운트만
           찍혀도 onset으로 검출되었다.
      당시 히스토그램 양 끝에 나타난 ±300~360분의 큰 lead time은 알고리즘이
      만든 "가짜 더미"가 아니었다. p70 임계값이 사실상 배경 수준이라 count가
      평상시에도 수시로 넘었기 때문에, 알고리즘은 정직하게 "임계값을 처음 넘은
      시점"을 보고했을 뿐이다. 원인은 onset 로직이 아니라 임계값 자체였다.
      → 배경 기반 임계값(bg_median + kσ_quiet, 실측 과분산 반영)으로 전환하여
        근본 해결한다. detected 플래그(검출/실패 구분)는 그대로 유효하다.
    """
    th = event_bg_threshold(cnt, ev)
    if not np.isfinite(th):
        return None, False

    t_start = ev.onset - pd.Timedelta(hours=6)
    t_end   = ev.peak  + pd.Timedelta(hours=6)
    seg     = cnt.loc[t_start:t_end]
    above   = seg[seg > th]
    if len(above):
        return above.index[0], True
    seg_ev = cnt.loc[ev.onset:t_end].dropna()
    if len(seg_ev):
        return seg_ev.idxmax(), False   # fallback: 참고용, 검출 아님
    return None, False


def print_proxy_diag(proxy: pd.Series, thresh: float | None, label: str):
    """proxy 분포 percentile 출력 — threshold 설정 전 참고용.
    (ksem_common에도 동일 함수가 있으나, import 순서/단독 실행 호환을 위해
     ana4 로컬에도 동일 구현을 둔다.)"""
    pv = proxy.dropna()
    print(f"  [diag] {label} proxy percentiles (cm-2 sr-1 s-1):")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"         p{p:2d} = {np.nanpercentile(pv, p):.3e}")
    thresh_str = f"{thresh:.1e}" if thresh is not None else "None (설정 필요)"
    print(f"  [diag] current threshold = {thresh_str}")
    if thresh is not None:
        print(f"  [diag] fraction above thresh = {(pv >= thresh).mean():.3f}")


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


def plot_superposed_epoch(df_count: pd.DataFrame,
                          events: list[SPEEvent],
                          logics: list[str],
                          suffix: str = ""):
    """
    3행(PD1/PD2/PD3) × len(logics)열.
    실선=median, 음영=IQR, A=실선, B=점선.
    suffix="" → proton / suffix="_electron" → electron
    """
    nrows, ncols = len(COUNT_PD_KEYS), len(logics)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5 * ncols, 3.8 * nrows), squeeze=False)
    label = "Electron" if suffix else "Proton"
    fig.suptitle(
        f"{label} Superposed Epoch  (peak-normalized, 30-min bin)\n"
        "solid=median, shade=IQR, solid line=A, dashed=B", fontsize=11)

    colors = {"PD1": "#2980b9", "PD2": "#27ae60", "PD3": "#8e44ad"}
    ls_map = {"A": "-", "B": "--"}

    for ri, pd_key in enumerate(COUNT_PD_KEYS):
        for ci, logic in enumerate(logics):
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
    out = OUTPUT_DIR / f"ana4{suffix}_superposed_epoch.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[ana4] Superposed epoch saved: {out}")


# ─────────────────────────────────────────────────────────────────
# (2) Rise Time CDF
# ─────────────────────────────────────────────────────────────────
def plot_rise_time_cdf(df_count: pd.DataFrame,
                       events: list[SPEEvent],
                       logics: list[str],
                       suffix: str = ""):
    """
    count > 배경 임계값(bg_median + kσ_quiet) 첫 시점 → peak 까지 시간 = rise time CDF.
    임계값은 count_onset 내부에서 이벤트 onset 기준으로 채널별 계산된다.
    suffix="" → proton / suffix="_electron" → electron
    """
    nrows = len(COUNT_PD_KEYS)
    ncols = len(logics)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.5 * ncols, 3.2 * nrows), squeeze=False)
    label = "Electron" if suffix else "Proton"
    fig.suptitle(
        f"{label} Rise Time CDF  "
        f"(count > bg_median + {COUNT_BG_K}\u03c3_quiet  first crossing \u2192 Peak)\n"
        "CDF: fraction of events reaching peak within x hours", fontsize=11)

    colors_side = {"A": "#e74c3c", "B": "#3498db"}

    for ri, pd_key in enumerate(COUNT_PD_KEYS):
        for ci, logic in enumerate(logics):
            ax = axes[ri][ci]
            for side in COUNT_SIDES:
                try:
                    cnt = df_count[pd_key, side, logic]
                except KeyError:
                    continue
                rt = []
                for ev in events:
                    t_on, detected = count_onset(cnt, ev)
                    if t_on is None or not detected:
                        continue   # 검출 실패는 rise time 정의 자체가 무의미
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
    out = OUTPUT_DIR / f"ana4{suffix}_rise_time_cdf.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[ana4] Rise time CDF saved: {out}")


# ─────────────────────────────────────────────────────────────────
# (3) Flux onset vs Count onset 비교
# ─────────────────────────────────────────────────────────────────
def compute_onset_delays(df_count: pd.DataFrame,
                         proxy: pd.Series,
                         events: list[SPEEvent],
                         logics: list[str],
                         thresh: float) -> pd.DataFrame:
    """
    delay_min = count_onset - flux_onset [분]
    negative  = count가 flux보다 먼저 상승
    count onset 임계값은 이벤트별·채널별 배경 임계값(bg_median + kσ_quiet).
    proxy/thresh/logics 인자화로 proton/electron 공용.
    """
    flux_onsets = {}
    for ev in events:
        seg   = proxy.loc[ev.onset - pd.Timedelta(hours=12):ev.peak]
        above = seg[seg >= thresh]
        flux_onsets[ev.onset] = above.index[0] if len(above) else ev.onset

    records = []
    for pd_key in COUNT_PD_KEYS:
        for side in COUNT_SIDES:
            for logic in logics:
                try:
                    cnt = df_count[pd_key, side, logic]
                except KeyError:
                    continue
                for ev in events:
                    c_on, detected = count_onset(cnt, ev)
                    f_on = flux_onsets.get(ev.onset)
                    if c_on is None or f_on is None:
                        continue
                    # 검출 성공분만 delay 통계에 사용. 실패는 detected=False로
                    # 남겨 두되 delay_min은 NaN.
                    if detected:
                        delay = (c_on - f_on).total_seconds() / 60
                        delay_val = round(delay, 1)
                    else:
                        delay_val = np.nan
                    records.append({
                        "pd_key":      pd_key,
                        "side":        side,
                        "logic":       logic,
                        "event_onset": ev.onset,
                        "flux_onset":  f_on,
                        "count_onset": c_on,
                        "delay_min":   delay_val,
                        "detected":    detected,
                        "peak_flux":   ev.peak_flux,
                    })
    return pd.DataFrame(records)


def plot_onset_comparison(df_delay: pd.DataFrame,
                          logics: list[str],
                          suffix: str = ""):
    """delay_min 히스토그램 (logic별 패널, PD별 색)."""
    if df_delay.empty:
        print(f"[ana4] onset delay{suffix}: No data")
        return
    ncols  = len(logics)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    if ncols == 1:
        axes = [axes]
    colors = {"PD1": "#2980b9", "PD2": "#27ae60", "PD3": "#8e44ad"}
    label  = "Electron" if suffix else "Proton"

    for ci, logic in enumerate(logics):
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

    fig.suptitle(f"{label} Count Onset vs Proxy Flux Onset delay distribution",
                 fontsize=11)
    fig.tight_layout()
    out = OUTPUT_DIR / f"ana4{suffix}_onset_comparison.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[ana4] Onset comparison saved: {out}")


# ─────────────────────────────────────────────────────────────────
# (4) Logic onset order
# ─────────────────────────────────────────────────────────────────
def plot_logic_order(df_delay: pd.DataFrame,
                     logics: list[str],
                     suffix: str = ""):
    """
    logic별 onset 지연 scatter. diamond = median.
    proton: O→OU→OUT 에너지 순서 확인
    electron: F→FT→FTU→FTUO 에너지 순서 확인
    """
    if df_delay.empty:
        return
    colors_l = {
        "O": "#e74c3c", "OU": "#f39c12", "OUT": "#27ae60",
        "F": "#e74c3c", "FT": "#f39c12", "FTU": "#27ae60", "FTUO": "#2980b9",
    }
    label = "Electron" if suffix else "Proton"
    fig, axes = plt.subplots(1, len(COUNT_PD_KEYS),
                              figsize=(5 * len(COUNT_PD_KEYS), 4))
    fig.suptitle(f"{label} Logic Onset Order\n"
                 "diamond=median | lower energy logic expected first",
                 fontsize=11)

    for ci, pd_key in enumerate(COUNT_PD_KEYS):
        ax  = axes[ci]
        sub = df_delay[df_delay["pd_key"] == pd_key]
        for logic in logics:
            vals = sub[sub["logic"] == logic]["delay_min"].dropna()
            if vals.empty:
                continue
            c = colors_l.get(logic, "#7f8c8d")
            ax.scatter([logic] * len(vals), vals,
                       color=c, alpha=0.5, s=20, zorder=3)
            ax.plot(logic, vals.median(), "D",
                    color=c, ms=10, zorder=5,
                    label=f"{logic} med={vals.median():.0f}min")
        ax.axhline(0, ls="--", color="gray", lw=0.8)
        ax.set_title(pd_key, fontsize=10)
        ax.set_ylabel("Delay vs flux onset (min)", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(labelsize=8)

    fig.tight_layout()
    out = OUTPUT_DIR / f"ana4{suffix}_logic_order.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[ana4] Logic onset order saved: {out}")


# ─────────────────────────────────────────────────────────────────
# (5) PD 방향각 의존성
# ─────────────────────────────────────────────────────────────────
def plot_pd_direction(df_count: pd.DataFrame,
                      events: list[SPEEvent],
                      logics: list[str],
                      suffix: str = ""):
    """
    (PD1/PD2), (PD1/PD3), (PD2/PD3) peak count 비율 히스토그램.
    비율 > 1 → 분자 PD 방향에서 더 많이 계수.
    suffix="" → proton / suffix="_electron" → electron
    """
    pairs  = [("PD1", "PD2"), ("PD1", "PD3"), ("PD2", "PD3")]
    ncols  = len(logics)
    nrows  = len(pairs)
    label  = "Electron" if suffix else "Proton"
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    fig.suptitle(f"{label} PD Peak Count Ratio (A vs B side)\n"
                 "ratio>1: numerator-direction dominant | "
                 "solid line=A median, dashed line=B median", fontsize=11)

    # side별 색 (A/B 구분). 비율 자체는 같은 PD쌍이므로 색으로만 분리.
    side_style = {
        "A": {"color": "#9b59b6", "ls": "--"},   # 보라 (기존 A 색 유지)
        "B": {"color": "#e67e22", "ls": ":"},     # 주황
    }

    for ri, (pd0, pd1) in enumerate(pairs):
        for ci, logic in enumerate(logics):
            ax = axes[ri][ci]

            # A·B 각각의 비율 분포를 같은 축에 겹쳐 그린다.
            for side in COUNT_SIDES:
                ratios = []
                for ev in events:
                    try:
                        p0 = df_count[pd0, side, logic].loc[ev.onset:ev.end].max()
                        p1 = df_count[pd1, side, logic].loc[ev.onset:ev.end].max()
                    except KeyError:
                        continue
                    if np.isfinite(p0) and np.isfinite(p1) and p1 > 0:
                        ratios.append(p0 / p1)
                if not ratios:
                    continue
                med = float(np.median(ratios))
                st  = side_style.get(side, {"color": "#7f8c8d", "ls": "-"})
                ax.hist(ratios, bins=10, color=st["color"],
                        alpha=0.45, edgecolor="white",
                        label=f"{side} (n={len(ratios)})")
                ax.axvline(med, ls=st["ls"], color=st["color"], lw=1.8,
                           label=f"{side} median={med:.2f}")

            ax.axvline(1.0, ls="-", color="gray", lw=0.8, alpha=0.6)
            ax.set_xlabel(f"{pd0}/{pd1} Ratio", fontsize=9)
            ax.set_ylabel("N events", fontsize=9)
            ax.set_title(f"{pd0}/{pd1} | {logic}", fontsize=9)
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = OUTPUT_DIR / f"ana4{suffix}_pd_direction.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[ana4] PD direction dependence saved: {out}")


def save_delay_tables(df_delay: pd.DataFrame, suffix: str = ""):
    """onset delay summary/raw CSV 저장.

    delay_median/mean/std/n_detected 는 '검출 성공분'만 집계된다
    (delay_min이 NaN인 검출 실패는 agg에서 자동 제외).
    여기에 n_events(전체 시도)와 detect_rate(검출 성공률)를 추가해,
    어떤 채널이 SPE를 실제로 얼마나 잡아내는지를 정량화한다.
    FSM 트리거 채널은 detect_rate가 높은 것을 골라야 한다.
    """
    if df_delay.empty:
        return
    grp = df_delay.groupby(["pd_key", "side", "logic"])
    summary = (grp["delay_min"]
               .agg(["median", "mean", "std", "count"])
               .reset_index()
               .rename(columns={"median": "delay_median_min",
                                "mean":   "delay_mean_min",
                                "std":    "delay_std_min",
                                "count":  "n_detected"}))
    # 전체 시도 수와 검출 성공률 부착
    n_total = grp["detected"].size().reset_index(name="n_events")
    summary = summary.merge(n_total, on=["pd_key", "side", "logic"])
    summary["detect_rate"] = (summary["n_detected"]
                              / summary["n_events"]).round(3)

    out_csv = OUTPUT_DIR / f"ana4{suffix}_onset_delay_table.csv"
    summary.to_csv(out_csv, index=False, float_format="%.2f")
    print(f"[ana4] summary table saved: {out_csv}")
    print(summary.to_string(index=False))

    df_delay.to_csv(OUTPUT_DIR / f"ana4{suffix}_onset_delay_raw.csv", index=False)
    print(f"[ana4] Raw delay saved: ana4{suffix}_onset_delay_raw.csv")


def run_profile(df_count: pd.DataFrame,
                proxy: pd.Series,
                events: list[SPEEvent],
                logics: list[str],
                thresh: float,
                suffix: str):
    """proton/electron 공용 분석 파이프라인."""
    plot_superposed_epoch(df_count, events, logics, suffix)
    plot_rise_time_cdf(df_count, events, logics, suffix)
    df_delay = compute_onset_delays(df_count, proxy, events, logics, thresh)
    plot_onset_comparison(df_delay, logics, suffix)
    plot_logic_order(df_delay, logics, suffix)
    plot_pd_direction(df_count, events, logics, suffix)
    save_delay_tables(df_delay, suffix)


# ─────────────────────────────────────────────────────────────────
def main():
    print("[ana4] Loading data...")
    df_count, df_flux_p, df_flux_e = load_data()

    # ── Proton proxy & 진단 ───────────────────────────────────────
    proxy_p = make_proton_proxy(df_flux_p)
    print_proxy_diag(proxy_p, SPE_PROXY_THRESH, "proton SPE")
    events_p = detect_proton_events(proxy_p)
    if not events_p:
        print("[ana4] No proton SPE events: exit")
        return

    # ── Proton 분석 ───────────────────────────────────────────────
    run_profile(df_count, proxy_p, events_p,
                logics=COUNT_PROTON_LOGICS,
                thresh=SPE_PROXY_THRESH,
                suffix="")

    # ── Electron proxy & 진단 ─────────────────────────────────────
    proxy_e = make_electron_proxy(df_flux_e)
    print_proxy_diag(proxy_e, E_SPE_PROXY_THRESH, "electron E_SPE")

    if E_SPE_PROXY_THRESH is None:
        print("[ana4] E_SPE_PROXY_THRESH=None → electron 분석 스킵. "
              "위 diag 결과의 p95를 참고해 config에 설정 후 재실행하세요.")
        return

    events_e = detect_electron_events(proxy_e)
    if not events_e:
        print("[ana4] No electron E_SPE events")
        return

    # ── Electron 분석 ─────────────────────────────────────────────
    run_profile(df_count, proxy_e, events_e,
                logics=COUNT_ELECTRON_LOGICS,
                thresh=E_SPE_PROXY_THRESH,
                suffix="_electron")


if __name__ == "__main__":
    main()