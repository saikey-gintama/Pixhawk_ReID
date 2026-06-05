"""
ana6_count_rolling_spe.py
=========================
고에너지(OU/OUT) count 단독 SPE 탐지 — flux/proxy ground truth 없음.

배경:
  ana1~ana5의 SPE 정의는 proton proxy(E8+E9+E10, 6 MeV 이하 적분)에 의존한다.
  하지만 KSEM count의 OU/OUT 로직은 채널 에너지 대역상 최대 ~18 MeV 관통 입자를
  보는 고에너지 coincidence 채널이고, KMA flux proxy가 6 MeV에서 잘려 있어
  proxy 이벤트 창에서는 OU/OUT이 (당연히) 반응하지 않는다.

  → 이 스크립트는 flux를 전혀 쓰지 않고, OU/OUT count 자체가 rolling 배경 대비
    상승하는 시점을 독립적으로 검출한다.

검출 모델 (이중 문턱):
  - 상승 기준: bg_median + k·σ_robust (MAD 기반, ksem_common). σ가 MAD라
    OU/OUT(quiet 과반이 0)에서는 σ≈0 → 임계가 ONSET_FLOOR에 수렴한다.
  - ONSET_FLOOR : onset/end 구간 경계(낮게). 상승 시작을 일찍 잡고 본체를 유지.
  - PEAK_FLOOR  : 이벤트 인정 사후 필터. 구간 peak<PEAK_FLOOR이면 노이즈로 버림.
  - onset 유효성: 임계 초과가 MIN_SPE_DURATION_H 이상 연속.
  - end/hysteresis/PRE_ALERT/재무장 등 FSM 로직 없음(FSM 브랜치에서).

파라미터 스윕 (data 브랜치: 특성 정량화가 목적):
  검출은 (k × onset_floor) 조합만 수행한다. peak_floor는 검출 후 사후 필터라
  같은 (k, onset_floor)에서 구간은 동일하고 peak로 거르기만 다르다
  → 무거운 검출은 len(K)×len(ONSET) 번만 돌고, peak_floor는 라벨링으로 처리.

출력 (그림 대신 표 중심):
  ana6_sweep_onset.csv
    onset 통과한 "모든 구간"(peak_floor 적용 전). (k × onset_floor) 조합.
    peak_floor 컬럼 없음 — 단, peak_count는 기록하므로 어떤 peak 기준이든 사후 적용 가능.
    리소스(onset 경보 빈도)·precursor(작은 peak 분포) 분석용 raw.
  ana6_sweep_event.csv
    위 구간 중 peak_floor를 통과한 이벤트만. (k × onset_floor × peak_floor) 조합.
    peak_floor 컬럼 있음.
  ana6_sweep_summary.csv
    (채널 × k × onset × peak)별 onset 구간 수 / event 수 집계.
  panels_final/...
    FINAL 파라미터(FINAL_K, FINAL_ONSET, FINAL_PEAK) 한 세트만 패널 저장.

각 이벤트(onset/event 공통) 기록 항목:
  k, onset_floor, [peak_floor], pd_key, side, logic, channel,
  onset_time, peak_time, end_time, onset_count, peak_count, end_count,
  duration_h, bg_median, bg_sigma, threshold   (모두 onset 시점 기준 bg)
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import ksem_flux_config as cfg
from ksem_flux_config import (
    OUTPUT_DIR,
    COUNT_PROTON_LOGICS, COUNT_PD_KEYS, COUNT_SIDES,
    BG_WINDOW_DAYS, BG_QUIET_DAYS,
)
from ksem_common import (
    SPEEvent,
    load_data,
    compute_rolling_bg,
)

# ══════════════════════════════════════════════════════════════════
# 파라미터 (이 블록만 조정)
# ══════════════════════════════════════════════════════════════════
RESAMPLE_FREQ        = "15min"          # 검출 적용 시계열 (ana1과 정합)
HE_LOGICS            = ["OU", "OUT"]    # 고에너지 채널만
MIN_SPE_DURATION_H   = cfg.MIN_SPE_DURATION_H   # onset 유효 최소 지속 (현재 1h)
BG_UPDATE_FREQ       = "1D"             # rolling 배경 갱신 주기

# ── 스윕 후보 ─────────────────────────────────────────────────────
K_SWEEP            = [3, 10, 20]        # bg_median + k·σ 의 k
ONSET_FLOOR_SWEEP  = [0.5, 1.0, 2.0]    # onset/end 구간 경계 floor
PEAK_FLOOR_SWEEP   = [2.0, 3.0, 5.0]    # 이벤트 인정 사후 필터 (검출 후 라벨링)

# ── 패널 저장용 FINAL 세트 (스윕 중 이 조합만 그림 저장) ──────────
FINAL_K             = 10
FINAL_ONSET_FLOOR   = 0.5
FINAL_PEAK_FLOOR    = 3.0
SAVE_PANELS         = True               # False면 그림 전부 생략(스윕 표만)
PANEL_WINDOW_H      = 24                 # 패널 onset 전후 ±시간
MAX_PANELS_PER_CHAN = 30                 # 채널당 패널 상한(폭주 가드)

# ── 출력 폴더 ─────────────────────────────────────────────────────
ANA6_OUTPUT_DIR = OUTPUT_DIR / "ana6_output"
ANA6_OUTPUT_DIR.mkdir(exist_ok=True)

# HE_LOGICS가 정말 COUNT_PROTON_LOGICS의 부분집합인지 안전 확인
assert set(HE_LOGICS) <= set(COUNT_PROTON_LOGICS), "HE_LOGICS must be proton logics"


# ══════════════════════════════════════════════════════════════════
# (1) 구간 검출 — peak_floor 적용 전 "모든 onset 구간"을 dict로 반환
# ══════════════════════════════════════════════════════════════════
def detect_segments(cnt: pd.Series,
                    thresh_series: pd.Series,
                    bg: pd.DataFrame,
                    min_duration_h: float) -> list[dict]:
    """
    count >= rolling 임계(thresh_series)가 min_duration_h 이상 연속인 구간을
    "모든" 반환한다(peak_floor 미적용). 각 구간마다 onset/peak/end의 시각·count와
    onset 시점 배경(bg_median, bg_sigma, threshold)을 기록한다.

    peak_floor는 호출부에서 peak_count로 사후 필터링한다(여기선 거르지 않음).
    임계가 NaN(배경 추정 불가)인 시점은 미검출(False)로 처리.
    """
    th  = thresh_series.reindex(cnt.index).ffill()
    med = bg["bg_median"].reindex(cnt.index).ffill()
    sig = bg["bg_std"].reindex(cnt.index).ffill()   # MAD 기반 robust σ (컬럼명은 bg_std)
    valid = cnt.notna() & th.notna()
    above = (cnt >= th) & valid

    segs: list[dict] = []
    in_ev = False
    onset = None

    def _close(seg, onset_t, end_t):
        if (end_t - onset_t).total_seconds() / 3600 < min_duration_h:
            return
        peak_t = seg.idxmax()
        segs.append({
            "onset_time": onset_t,
            "peak_time":  peak_t,
            "end_time":   end_t,
            "onset_count": round(float(cnt.loc[onset_t]), 3),
            "peak_count":  round(float(seg.max()), 3),
            "end_count":   round(float(cnt.loc[end_t]), 3),
            "duration_h":  round((end_t - onset_t).total_seconds() / 3600, 2),
            "bg_median":   round(float(med.loc[onset_t]), 4) if pd.notna(med.loc[onset_t]) else np.nan,
            "bg_sigma":    round(float(sig.loc[onset_t]), 4) if pd.notna(sig.loc[onset_t]) else np.nan,
            "threshold":   round(float(th.loc[onset_t]), 4),
        })

    for t, is_above in above.items():
        if is_above and not in_ev:
            in_ev, onset = True, t
        elif not is_above and in_ev:
            _close(cnt.loc[onset:t], onset, t)
            in_ev = False
    if in_ev and onset is not None:
        seg = cnt.loc[onset:]
        _close(seg, onset, seg.index[-1])

    return segs


def build_threshold(cnt: pd.Series, bg: pd.DataFrame,
                    k: float, onset_floor: float) -> pd.Series:
    """rolling 배경에서 (median + k·σ_robust)를 onset_floor로 클리핑한 임계 시계열."""
    th = bg["bg_median"] + k * bg["bg_std"]   # bg_std = MAD 기반 robust σ
    if onset_floor > 0:
        th = th.clip(lower=onset_floor)
    return th


# ══════════════════════════════════════════════════════════════════
# (2) 패널 — FINAL 조합만. segs(dict 리스트)를 그린다.
# ══════════════════════════════════════════════════════════════════
def plot_panels(cnt: pd.Series, thresh: pd.Series,
                segs: list[dict], panel_tag: str):
    if not SAVE_PANELS or len(segs) == 0:
        return
    if len(segs) > MAX_PANELS_PER_CHAN:
        print(f"  [ana6] {panel_tag}: {len(segs)} segs > {MAX_PANELS_PER_CHAN} "
              f"→ 패널 생략 (CSV만)")
        return

    panel_dir = ANA6_OUTPUT_DIR / "panels_final" / panel_tag
    panel_dir.mkdir(parents=True, exist_ok=True)
    th_ff = thresh.reindex(cnt.index).ffill()

    for ei, ev in enumerate(segs):
        t0 = ev["onset_time"] - pd.Timedelta(hours=PANEL_WINDOW_H)
        t1 = ev["end_time"]   + pd.Timedelta(hours=PANEL_WINDOW_H)
        fig, ax = plt.subplots(1, 1, figsize=(13, 3.6))
        fig.suptitle(
            f"HE_SPE [{panel_tag}] #{ei+1}  |  "
            f"Onset {ev['onset_time'].strftime('%Y-%m-%d %H:%M')}  "
            f"Peak {ev['peak_count']:.1f}", fontsize=10)
        c_seg = cnt.loc[t0:t1]
        t_seg = th_ff.loc[t0:t1]
        ax.plot(c_seg.index, c_seg.values, color="#2c3e50", lw=1.0, label="count")
        ax.plot(t_seg.index, t_seg.values, color="red", ls=":", lw=1.2, label="threshold")
        ax.axvline(ev["onset_time"], ls="--", color="orange", lw=1, alpha=0.8, label="onset")
        ax.axvline(ev["peak_time"],  ls=":",  color="red",    lw=1.2, alpha=0.6, label="peak")
        ax.axvline(ev["end_time"],   ls="--", color="gray",   lw=1, alpha=0.8, label="end")
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Count [15-min]", fontsize=8)
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
        fig.tight_layout()
        fig.savefig(panel_dir / f"ana6_event_{ei+1:02d}_panel.png",
                    dpi=120, bbox_inches="tight")
        plt.close(fig)
    print(f"  [ana6] {panel_tag} panels {len(segs)} saved → {panel_dir}")


# ══════════════════════════════════════════════════════════════════
def main():
    print("[ana6] Loading data...")
    df_count, _, _ = load_data()
    cnt_rs = df_count.resample(RESAMPLE_FREQ).mean()

    onset_rows = []   # sweep_onset: peak_floor 적용 전 모든 구간 (k × onset_floor)
    event_rows = []   # sweep_event: peak_floor 통과 (k × onset_floor × peak_floor)
    summary_rows = []

    for pd_key in COUNT_PD_KEYS:
        for side in COUNT_SIDES:
            for logic in HE_LOGICS:
                try:
                    cnt = cnt_rs[pd_key, side, logic].dropna()
                except KeyError:
                    continue
                if len(cnt) < 100:
                    continue
                chan = f"{pd_key}{side}-{logic}"
                print(f"\n[ana6] {chan}  (n={len(cnt)} pts)")

                # 배경은 채널당 1회만 (k·onset과 무관)
                bg = compute_rolling_bg(cnt, BG_WINDOW_DAYS, BG_QUIET_DAYS, BG_UPDATE_FREQ)

                # 검출: (k × onset_floor)만. peak_floor는 사후 라벨링.
                for k in K_SWEEP:
                    for onf in ONSET_FLOOR_SWEEP:
                        thr = build_threshold(cnt, bg, k, onf)
                        segs = detect_segments(cnt, thr, bg, MIN_SPE_DURATION_H)

                        base = {"k": k, "onset_floor": onf,
                                "pd_key": pd_key, "side": side,
                                "logic": logic, "channel": chan}

                        # sweep_onset: 모든 구간 기록 (peak_floor 무관)
                        for s in segs:
                            onset_rows.append({**base, **s})

                        # sweep_event: peak_floor별 통과분 기록
                        for pkf in PEAK_FLOOR_SWEEP:
                            passed = [s for s in segs if s["peak_count"] >= pkf]
                            for s in passed:
                                event_rows.append({**base, "peak_floor": pkf, **s})
                            summary_rows.append({
                                **base, "peak_floor": pkf,
                                "n_onset": len(segs),          # peak_floor 전 구간 수
                                "n_event": len(passed),        # peak_floor 후 이벤트 수
                            })

                        # 패널: FINAL 조합만
                        if (k == FINAL_K and onf == FINAL_ONSET_FLOOR):
                            final_segs = [s for s in segs
                                          if s["peak_count"] >= FINAL_PEAK_FLOOR]
                            tag = f"k{k}_on{onf}_pk{FINAL_PEAK_FLOOR}_{chan}"
                            plot_panels(cnt, thr, final_segs, tag)

                        print(f"    k={k:>2} onset={onf}: {len(segs)} segs "
                              f"(peak>=2/3/5: "
                              f"{sum(s['peak_count']>=2 for s in segs)}/"
                              f"{sum(s['peak_count']>=3 for s in segs)}/"
                              f"{sum(s['peak_count']>=5 for s in segs)})")

    # ── 저장 ──────────────────────────────────────────────────────
    ONSET_COLS = ["k", "onset_floor", "pd_key", "side", "logic", "channel",
                  "onset_time", "peak_time", "end_time",
                  "onset_count", "peak_count", "end_count", "duration_h",
                  "bg_median", "bg_sigma", "threshold"]
    EVENT_COLS = ["k", "onset_floor", "peak_floor", "pd_key", "side", "logic", "channel",
                  "onset_time", "peak_time", "end_time",
                  "onset_count", "peak_count", "end_count", "duration_h",
                  "bg_median", "bg_sigma", "threshold"]

    df_on = pd.DataFrame(onset_rows, columns=ONSET_COLS)
    df_ev = pd.DataFrame(event_rows, columns=EVENT_COLS)
    df_sum = pd.DataFrame(summary_rows)

    out_on  = ANA6_OUTPUT_DIR / "ana6_sweep_onset.csv"
    out_ev  = ANA6_OUTPUT_DIR / "ana6_sweep_event.csv"
    out_sum = ANA6_OUTPUT_DIR / "ana6_sweep_summary.csv"
    df_on.to_csv(out_on, index=False)
    df_ev.to_csv(out_ev, index=False)
    df_sum.to_csv(out_sum, index=False)

    print(f"\n[ana6] sweep_onset  saved: {out_on}  ({len(df_on)} rows)")
    print(f"[ana6] sweep_event  saved: {out_ev}  ({len(df_ev)} rows)")
    print(f"[ana6] sweep_summary saved: {out_sum}  ({len(df_sum)} rows)")

    # 요약 피벗: 채널 × (k,onset,peak) → n_event
    if not df_sum.empty:
        piv = df_sum.pivot_table(
            index=["pd_key", "side", "logic"],
            columns=["k", "onset_floor", "peak_floor"],
            values="n_event", fill_value=0)
        print("\n[ana6] n_event pivot (channel × k/onset/peak):")
        print(piv.to_string())


if __name__ == "__main__":
    main()