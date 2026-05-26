"""
ana2_spe_count_threshold.py
============================
SPE 구간 count 분포 및 임계값 도출

SPE 탐지 기준:
  proxy flux (E8+E9+E10 ΔE-적분, cm-2 sr-1 s-1) >= SPE_PROXY_THRESH 가
  MIN_SPE_DURATION_H 이상 연속되는 구간을 이벤트로 인정.
  단위: cm-2 sr-1 s-1 keV-1  (NOAA 10 pfu와 직접 비교 불가)

출력:
  ana2_event_XX_panel.png  (이벤트 수만큼)
    행0: proxy flux 시계열 (onset/peak/end 수직선)
    행1~3: O / OU / OUT count (PD1·PD2·PD3 A/B 겹쳐 그리기)
  ana2_spe_multiplier_hist.png
    SPE onset count / 14일 배경 median 배율 히스토그램 (logic별 패널)
  ana2_ab_symmetry.png
    (A-B)/(A+B) 시계열 (1시간 리샘플 median) — 0=완전 대칭
  ana2_spe_threshold_table.csv
    quiet_p50/75/90/95/99, spe_median_count,
    onset_bg_ratio_median, spe_exceed_q{SPE_COUNT_THRESH_PCTL}_rate

튜닝값 (이 파일 단독):
  BG_WINDOW_DAYS   : onset 직전 이 기간의 전체 median을 배경으로 사용.
                     이전 이벤트와 겹치면 배경이 오염될 수 있음.
  ONSET_WINDOW_H   : onset 시 count를 대표하는 윈도우. ana4 delay 결과(~49분)
                     참고하여 설정. 너무 짧으면 count 상승 전을 잡을 수 있음.
  PRE_ONSET_H      : 이벤트 패널 그림의 onset 앞 시각화 여백 (분석 범위 아님).
  POST_END_H       : 이벤트 패널 그림의 end 뒤 시각화 여백 (분석 범위 아님).
  PERCENTILES      : Quiet 분포에서 추출할 FSM threshold 후보 퍼센타일.
                     p{SPE_COUNT_THRESH_PCTL} 초과 비율(spe_exceed_qXX_rate)로 유효성 검증.
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

# ── ana2 단독 튜닝값 ──────────────────────────────────────────────
BG_WINDOW_DAYS  = 14   # [day] onset 이전 배경 기간 (이 기간 전체 median 사용)
ONSET_WINDOW_H  = 2    # [h]   onset 직후 count 대표값 윈도우
PRE_ONSET_H     = 12   # [h]   이벤트 패널 시각화 앞 여백
POST_END_H      = 12   # [h]   이벤트 패널 시각화 뒤 여백
PERCENTILES     = [50, 60, 70, 80, 90]   # Quiet 퍼센타일 (threshold 후보군)


# ─────────────────────────────────────────────────────────────────
class SPEEvent(NamedTuple):
    onset:     pd.Timestamp
    peak:      pd.Timestamp
    end:       pd.Timestamp
    peak_flux: float   # proxy 단위 [cm-2 sr-1 s-1]


# ─────────────────────────────────────────────────────────────────
def make_proxy(df_flux: pd.DataFrame) -> pd.Series:
    """
    E8+E9+E10 ΔE-적분 합산 proxy.
    단위: cm-2 sr-1 s-1
    """
    proxy = pd.Series(0.0, index=df_flux.index)
    for ch in SPE_PROXY_CHANNELS:
        if ch not in df_flux.columns:
            print(f"  [ana2] warning: {ch} 없음")
            continue
        lo, hi = PROTON_CHANNELS[ch]
        proxy += df_flux[ch].fillna(0) * (hi - lo)
    proxy[proxy == 0] = np.nan
    return proxy.rename("proxy")


def detect_spe_events(proxy: pd.Series) -> list[SPEEvent]:
    """
    proxy >= SPE_PROXY_THRESH 가 MIN_SPE_DURATION_H 이상 연속인 구간을 탐지.
    단발성 spike(< MIN_SPE_DURATION_H)는 이벤트로 미인정.
    """
    above  = proxy >= SPE_PROXY_THRESH
    events, in_ev, onset = [], False, None

    for t, v in above.items():
        if v and not in_ev:
            in_ev, onset = True, t
        elif not v and in_ev:
            seg   = proxy.loc[onset:t]
            dur_h = (t - onset).total_seconds() / 3600
            if dur_h >= MIN_SPE_DURATION_H:
                events.append(SPEEvent(onset, seg.idxmax(), t, float(seg.max())))
            in_ev = False

    # 데이터 끝까지 threshold를 초과한 경우
    if in_ev and onset is not None:
        seg = proxy.loc[onset:]
        if (seg.index[-1] - onset).total_seconds() / 3600 >= MIN_SPE_DURATION_H:
            events.append(SPEEvent(onset, seg.idxmax(),
                                   seg.index[-1], float(seg.max())))

    print(f"[ana2] SPE events detected: {len(events)}  "
          f"(proxy >= {SPE_PROXY_THRESH:.1e}, min {MIN_SPE_DURATION_H}h)")
    return events


def get_bg_median(cnt: pd.Series, onset: pd.Timestamp) -> float:
    """onset 직전 BG_WINDOW_DAYS 전체 데이터의 median을 배경값으로 반환."""
    seg = cnt.loc[onset - pd.Timedelta(days=BG_WINDOW_DAYS):onset].dropna()
    return float(seg.median()) if len(seg) > 10 else np.nan


# ─────────────────────────────────────────────────────────────────
# (1) 이벤트별 시계열 패널
# ─────────────────────────────────────────────────────────────────
def plot_event_panels(df_count: pd.DataFrame,
                      proxy: pd.Series,
                      events: list[SPEEvent]):
    """
    이벤트별 4행 패널:
      행0: proxy flux (로그 스케일, threshold 수평선)
      행1~3: O / OU / OUT count (PD1·PD2·PD3 A/B 겹쳐 그리기, symlog 스케일)
    onset·peak·end를 수직선으로 표시하여 PD간 편차를 한눈에 확인.
    """
    palette = {"PD1": "#2980b9", "PD2": "#27ae60", "PD3": "#8e44ad"}
    ls_map  = {"A": "-", "B": "--"}

    for ei, ev in enumerate(events):
        t0     = ev.onset - pd.Timedelta(hours=PRE_ONSET_H)
        t1     = ev.end   + pd.Timedelta(hours=POST_END_H)
        n_rows = len(COUNT_PROTON_LOGICS) + 1

        fig, axes = plt.subplots(n_rows, 1,
                                  figsize=(13, 2.8 * n_rows), sharex=True)
        fig.suptitle(
            f"SPE Event #{ei+1}  |  "
            f"Onset: {ev.onset.strftime('%Y-%m-%d %H:%M')}  "
            f"Peak flux: {ev.peak_flux:.2e} cm-2 sr-1 s-1",
            fontsize=10)

        # 행0: proxy flux
        ax0     = axes[0]
        seg_f   = proxy.loc[t0:t1].replace(0, np.nan)
        ax0.semilogy(seg_f.index, seg_f.values, "k-", lw=1.1)
        ax0.axhline(SPE_PROXY_THRESH, ls=":", color="red", lw=1,
                    label=f"threshold = {SPE_PROXY_THRESH:.1e}")
        for t, color, lbl in [(ev.onset, "orange", "onset"),
                               (ev.end,   "gray",   "end")]:
            ax0.axvline(t, ls="--", color=color, lw=1, alpha=0.8, label=lbl)
        ax0.set_ylabel(f"Proxy flux\n({SPE_PROXY_LABEL})\n[cm-2 sr-1 s-1]", fontsize=7)
        ax0.legend(fontsize=6, loc="upper right")
        ax0.grid(True, alpha=0.3)

        # 행1~3: logic별 count
        for li, logic in enumerate(COUNT_PROTON_LOGICS):
            ax = axes[li + 1]
            for pd_key in COUNT_PD_KEYS:
                for side in COUNT_SIDES:
                    try:
                        s = df_count[pd_key, side, logic].loc[t0:t1]
                    except KeyError:
                        continue
                    ax.plot(s.index, s.values,
                            color=palette[pd_key], ls=ls_map[side],
                            lw=0.9, alpha=0.8, label=f"{pd_key}-{side}")
            ax.axvline(ev.onset, ls="--", color="orange", lw=1, alpha=0.6)
            ax.axvline(ev.peak,  ls=":",  color="red",    lw=1.2, alpha=0.7,
                       label="peak")
            ax.axvline(ev.end,   ls="--", color="gray",   lw=1, alpha=0.6)
            ax.set_ylim(bottom=0)
            ax.set_ylabel(f"Count\n[{logic}]", fontsize=8)
            ax.legend(fontsize=5, ncol=3, loc="upper left")
            ax.grid(True, alpha=0.3)

        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
        fig.tight_layout()
        out = OUTPUT_DIR / f"ana2_event_{ei+1:02d}_panel.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)

    print(f"[ana2] event panels {len(events)} saved")


# ─────────────────────────────────────────────────────────────────
# (2) 퍼센타일 임계값 + 배율 계산
# ─────────────────────────────────────────────────────────────────
def compute_thresholds(df_count: pd.DataFrame,
                       proxy: pd.Series,
                       events: list[SPEEvent]):
    """
    각 (pd_key, side, logic) 채널에 대해 계산:
      quiet_pXX       : Quiet 구간 count 퍼센타일 → FSM threshold 후보
      spe_median_count: SPE 구간 count 중앙값
      onset_bg_ratio  : onset 직후 ONSET_WINDOW_H count / 14일 배경 median 배율
                        1.0 = 변화 없음, > 1.0 = SPE 때 count 상승
      spe_exceed_qxx  : SPE 구간 중 quiet_pxx를 초과한 시간 비율
                        높을수록 pxx를 threshold로 쓸 때 유효 경보율이 높음
    """
    is_spe   = proxy >= SPE_PROXY_THRESH
    is_quiet = ~is_spe

    records      = []
    mult_data    = {}   # (pd_key, side, logic) → list[float]
    ab_diff_data = {}   # (pd_key, logic)       → pd.Series

    for pd_key in COUNT_PD_KEYS:
        for side in COUNT_SIDES:
            for logic in COUNT_PROTON_LOGICS:
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
                for ev in events:
                    bg = get_bg_median(cnt, ev.onset)
                    if np.isnan(bg) or bg <= 0:
                        continue
                    on_cnt = cnt.loc[
                        ev.onset:ev.onset + pd.Timedelta(hours=ONSET_WINDOW_H)
                    ].median()
                    if np.isfinite(on_cnt):
                        ratios.append(on_cnt / bg)
                    thresh_key = f"quiet_p{SPE_COUNT_THRESH_PCTL}"
                    q_thresh = pct.get(thresh_key, np.nan)
                    if np.isfinite(q_thresh):
                        seg = cnt.loc[ev.onset:ev.end]
                        if len(seg) > 0:
                            exceed_rates.append((seg > q_thresh).mean())

                records.append({
                    "pd_key": pd_key, "side": side, "logic": logic,
                    **pct,
                    "spe_median_count":                            float(sp_vals.median()) if len(sp_vals) else np.nan,
                    "onset_bg_ratio_median":                       round(float(np.nanmedian(ratios)), 2) if ratios else np.nan,
                    f"spe_exceed_q{SPE_COUNT_THRESH_PCTL}_rate":   round(float(np.nanmean(exceed_rates)), 3) if exceed_rates else np.nan,
                    "quiet_n": len(q_vals),
                    "spe_n":   len(sp_vals),
                })
                mult_data[(pd_key, side, logic)] = ratios

        # A/B 비대칭 시계열: 0=완전 대칭, +1=A만 계수, -1=B만 계수
        for logic in COUNT_PROTON_LOGICS:
            try:
                cA = df_count[pd_key, "A", logic]
                cB = df_count[pd_key, "B", logic]
            except KeyError:
                continue
            denom = (cA + cB).replace(0, np.nan)
            ab_diff_data[(pd_key, logic)] = (cA - cB) / denom

    return pd.DataFrame(records), mult_data, ab_diff_data


def plot_multiplier_hist(mult_data: dict):
    """SPE onset 배율(onset count / 14일 배경 median) 히스토그램."""
    ncols  = len(COUNT_PROTON_LOGICS)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    colors = {"PD1": "#2980b9", "PD2": "#27ae60", "PD3": "#8e44ad"}

    for ci, logic in enumerate(COUNT_PROTON_LOGICS):
        ax = axes[ci]
        for pd_key in COUNT_PD_KEYS:
            for side in COUNT_SIDES:
                vals = mult_data.get((pd_key, side, logic), [])
                if vals:
                    ax.hist(vals, bins=15, alpha=0.4, color=colors[pd_key],
                            label=f"{pd_key}-{side}", density=True)
        ax.axvline(1.0, ls="--", color="black", lw=1, label="ratio=1 (no change)")
        ax.set_xlabel("Onset count / 14-day BG median", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_title(f"Logic: {logic}", fontsize=10)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"SPE Onset Multiplier  (onset {ONSET_WINDOW_H}h count / quiet {BG_WINDOW_DAYS}-day median)",
        fontsize=11)
    fig.tight_layout()
    out = OUTPUT_DIR / "ana2_spe_multiplier_hist.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ana2] multiplier histogram saved: {out}")


def plot_ab_symmetry(ab_diff_data: dict):
    """(A-B)/(A+B) 시계열. 1시간 리샘플 median."""
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
    fig.suptitle("A/B asymmetry  (A-B)/(A+B)  |  0=perfect symmetry  ±1=one-sided",
                 fontsize=11)
    fig.tight_layout()
    out = OUTPUT_DIR / "ana2_ab_symmetry.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[ana2] A/B symmetry saved: {out}")


# ─────────────────────────────────────────────────────────────────
def main():
    print("[ana2] Loading data...")
    df_count, _ = ksem_io.load(COUNT_PARQUET_DIR)
    sensor_data, _ = kma_ksem_flux_io.load(FLUX_PARQUET_DIR)
    df_flux = sensor_data.get("proton", pd.DataFrame())

    for df in [df_count, df_flux]:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

    proxy  = make_proxy(df_flux)
    events = detect_spe_events(proxy)

    plot_event_panels(df_count, proxy, events)

    df_thresh, mult_data, ab_data = compute_thresholds(df_count, proxy, events)
    plot_multiplier_hist(mult_data)
    plot_ab_symmetry(ab_data)

    out_csv = OUTPUT_DIR / "ana2_spe_threshold_table.csv"
    df_thresh.to_csv(out_csv, index=False, float_format="%.4g")
    print(f"[ana2] threshold table saved: {out_csv}")
    print(df_thresh.to_string(index=False))


if __name__ == "__main__":
    main()
