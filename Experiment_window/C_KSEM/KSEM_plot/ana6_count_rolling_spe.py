"""
ana6_count_rolling_spe.py
=========================
고에너지(OU/OUT) count 단독 SPE 탐지 — flux/proxy ground truth 없음.

배경:
  ana1~ana5의 SPE 정의는 proton proxy(E8+E9+E10, 6 MeV 이하 적분)에 의존한다.
  하지만 KSEM count의 OU/OUT 로직은 채널 에너지 대역상 최대 ~18 MeV 관통 입자를
  보는 고에너지 coincidence 채널이고, KMA flux proxy가 6 MeV에서 잘려 있어
  proxy 이벤트 창에서는 OU/OUT이 (당연히) 반응하지 않는다. 둘은 서로 다른 입자
  집단을 본다.

  → 이 스크립트는 flux를 전혀 쓰지 않고, OU/OUT count 자체가 rolling 배경 대비
    상승하는 시점을 독립적으로 검출한다. 6 MeV 이상 고에너지 대역에서 KSEM의
    유일한 검출 경로다.

ana2와의 관계 (의도적으로 최대한 동일하게 유지):
  - 이벤트 구조(SPEEvent), 패널 레이아웃, main 흐름 모두 ana2를 복제.
  - 단 하나의 차이: 이벤트 탐지 기준이
      ana2  : flux >= 고정 thresh           (ksem_common.detect_events)
      ana6  : count >= rolling 배경 × n      (이 파일 detect_events_rolling)
    rolling 배경 = compute_rolling_bg (직전 30일 중 조용한 7일, 일별 갱신).

이번 단계 범위:
  - OU/OUT 채널만, 각 PD·각 side 독립.
  - 상승 기준: bg_median + k·σ_robust (MAD 기반, ksem_common). σ가 MAD라
    OU/OUT(quiet 과반이 0)에서는 σ≈0 → 임계가 ONSET_FLOOR에 수렴한다.
    k는 탐색을 위해 {10, 15, 20} 동시 산출(단 σ≈0이라 k 민감도는 낮음).
  - 이중 문턱:
      ONSET_FLOOR : onset/end 구간 경계(낮게). 상승 시작을 일찍 잡고
                    본체를 끝까지 유지 → onset 지연·본체 절단 방지.
      PEAK_FLOOR  : 이벤트 인정 사후 필터. 구간 peak<PEAK_FLOOR이면 노이즈로 버림.
  - onset 유효성: 임계 초과가 MIN_SPE_DURATION_H 이상 연속(현재 1h).
  - end 조건/hysteresis/PRE_ALERT/재무장 등 FSM 로직 없음(FSM 브랜치에서).
  - 이벤트별 ±PANEL_WINDOW_H 패널. 폭주 시 패널 스킵 가드.

출력:
  ana_output/panels_he_<crit>_<PD><side>_<logic>/ana6_event_XX_panel.png
  ana_output/ana6_rolling_spe_events.csv   (전 기준·전 채널 검출 이벤트)
  ana_output/ana6_event_count_summary.csv  (기준×채널별 이벤트 수)

튜닝값 (이 파일 단독):
  RESAMPLE_FREQ        : 검출 적용 전 count 리샘플 (기존 15min과 정합).
  KSIG_K_SWEEP         : bg_median + k·σ 의 k 후보 (ana4 COUNT_BG_K=5 포함).
  PANEL_WINDOW_H       : 패널 onset 전후 여백.
  MAX_PANELS_PER_CHAN  : 채널당 패널 수 상한(폭주 가드).
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

# ── ana6 단독 튜닝값 ──────────────────────────────────────────────
RESAMPLE_FREQ        = "15min"          # 검출 적용 시계열 (ana1과 정합)
HE_LOGICS            = ["OU", "OUT"]    # 고에너지 채널만 (COUNT_PROTON_LOGICS의 부분집합)
KSIG_K_SWEEP         = [10, 15, 20]     # bg_median + k·σ 의 k 후보
MIN_SPE_DURATION_H   = cfg.MIN_SPE_DURATION_H   # 6h, flux 탐지와 정합
BG_UPDATE_FREQ       = "1D"             # rolling 배경 갱신 주기
PANEL_WINDOW_H       = 24               # 패널 onset 전후 ±시간
MAX_PANELS_PER_CHAN  = 150              # 채널당 패널 상한(폭주 가드)

# ── 이중 문턱: onset/end 경계 floor + 이벤트 인정 peak floor ──────
# 기존엔 floor 하나(MIN_ABS_COUNT_FLOOR)가 onset/end 판정에 직접 걸려,
# (1) onset이 floor를 넘어야 시작 → 상승 초반을 놓쳐 onset 지연
# (2) floor 아래로 한 번만 떨어져도 end → 본체 절단
# 문제가 있었다. MAD 도입 후 OU/OUT은 median+kσ≈0이라 floor가 검출을
# 100% 지배하므로 이 문제가 더 두드러진다.
#
# → 두 문턱으로 분리한다:
#   ONSET_FLOOR : onset/end 구간 경계. 낮게 둬서 상승 시작을 일찍 잡고
#                 본체를 끝까지 유지한다. threshold_onset = max(med+kσ, ONSET_FLOOR).
#   PEAK_FLOOR  : 이벤트 인정 문턱(사후 필터). 구간 peak_count가 이 값
#                 미만이면 노이즈로 보고 버린다. onset/end엔 관여하지 않는다.
# (GOES 방식과 동일: 낮은 경계로 구간을 잡고, 높은 문턱으로 이벤트를 판정.)
ONSET_FLOOR  = 0.5   # onset/end 구간 경계 floor
PEAK_FLOOR   = 3.0   # 이벤트 인정 최소 peak_count (사후 필터)

# ── 출력 부모 폴더 (패널 폴더들을 한 곳에 모음) ───────────────────
ANA6_OUTPUT_DIR = OUTPUT_DIR / "ana6_output"
ANA6_OUTPUT_DIR.mkdir(exist_ok=True)

# HE_LOGICS가 정말 COUNT_PROTON_LOGICS의 부분집합인지 안전 확인
assert set(HE_LOGICS) <= set(COUNT_PROTON_LOGICS), "HE_LOGICS must be proton logics"


# ─────────────────────────────────────────────────────────────────
# (1) rolling 임계 기반 이벤트 탐지 — ana6 단독 (공유함수 무손상)
# ─────────────────────────────────────────────────────────────────
def detect_events_rolling(cnt: pd.Series,
                          thresh_series: pd.Series,
                          min_duration_h: float,
                          peak_floor: float = 0.0,
                          label: str = "HE_SPE") -> list[SPEEvent]:
    """
    count >= 시점별 rolling 임계(thresh_series) 가 min_duration_h 이상 연속인
    구간을 SPEEvent로 반환. 단, 구간 peak_count < peak_floor 인 이벤트는
    노이즈로 보고 제외한다(이중 문턱의 사후 필터).

    thresh_series는 onset/end 경계용(ONSET_FLOOR 반영)이고, peak_floor는
    이벤트 인정용으로 별개다. 둘을 분리해 onset 지연/본체 절단 없이
    노이즈만 거른다.

    ksem_common.detect_events와 연속구간 묶기 로직은 동일하나, thresh가 스칼라가
    아닌 시계열이라는 점만 다르다(공유함수를 건드리지 않기 위해 별도 구현).
    SPEEvent.peak_flux 자리에는 구간 peak count를 넣는다(필드 재활용).

    임계값이 NaN(배경 추정 불가)인 시점은 미검출(False)로 처리한다.
    """
    th = thresh_series.reindex(cnt.index).ffill()
    valid = cnt.notna() & th.notna()
    above = (cnt >= th) & valid

    events: list[SPEEvent] = []
    in_ev = False
    onset = None

    def _maybe_append(seg, onset_t, end_t):
        """구간을 peak_floor로 필터링 후 이벤트 추가."""
        if (end_t - onset_t).total_seconds() / 3600 < min_duration_h:
            return
        pk = float(seg.max())
        if pk < peak_floor:          # 이중 문턱: peak가 낮으면 노이즈로 버림
            return
        events.append(SPEEvent(onset_t, seg.idxmax(), end_t, pk))

    for t, is_above in above.items():
        if is_above and not in_ev:
            in_ev, onset = True, t
        elif not is_above and in_ev:
            _maybe_append(cnt.loc[onset:t], onset, t)
            in_ev = False

    if in_ev and onset is not None:
        seg = cnt.loc[onset:]
        _maybe_append(seg, onset, seg.index[-1])

    print(f"  [ana6] {label} events: {len(events)}  "
          f"(min {min_duration_h}h, peak>={peak_floor})")
    return events


def build_threshold_series(cnt: pd.Series) -> dict[str, pd.Series]:
    """
    한 채널(cnt)의 rolling 배경에서 ksig(median + k·σ) 임계 시계열 생성.
    ana1~ana5에서 쓰던 임계 형태는 ana4의 median + k·σ(ksig) 하나뿐이며,
    ana6도 동일 형태를 rolling으로 적용한다(배수/percentile 방식은 쓰지 않음).
    k는 탐색을 위해 KSIG_K_SWEEP를 동시 산출.
    반환 key: "ksig10", "ksig15", "ksig20"
    """
    bg = compute_rolling_bg(cnt, BG_WINDOW_DAYS, BG_QUIET_DAYS, BG_UPDATE_FREQ)
    med, std = bg["bg_median"], bg["bg_std"]

    out: dict[str, pd.Series] = {}
    for k in KSIG_K_SWEEP:
        th = med + k * std
        if ONSET_FLOOR > 0:
            th = th.clip(lower=ONSET_FLOOR)   # onset/end 경계 floor (낮게)
        out[f"ksig{k}"] = th.rename(f"ksig{k}")
    return out


# ─────────────────────────────────────────────────────────────────
# (2) 이벤트 패널 — ana2 plot_event_panels 복제, 행0를 OU/OUT count로 교체
# ─────────────────────────────────────────────────────────────────
def plot_event_panels(cnt: pd.Series,
                      thresh: pd.Series,
                      events: list[SPEEvent],
                      panel_tag: str):
    """
    검출 채널 count와 적용된 rolling 임계를 ±PANEL_WINDOW_H 창으로 그린다.
    ana2 패널과 동일한 레이아웃 철학(onset/peak 수직선, 시간축 포맷)이되,
    flux 행이 없으므로 단일 행(count + threshold)만 그린다.

    MAX_PANELS_PER_CHAN 초과 시 패널 생략(이벤트 폭주 가드).
    """
    if len(events) == 0:
        return
    if len(events) > MAX_PANELS_PER_CHAN:
        print(f"  [ana6] {panel_tag}: {len(events)} events > "
              f"{MAX_PANELS_PER_CHAN} → 패널 생략 (CSV만 저장)")
        return

    panel_dir = ANA6_OUTPUT_DIR / f"panels_he_{panel_tag}"
    panel_dir.mkdir(exist_ok=True)

    for ei, ev in enumerate(events):
        t0 = ev.onset - pd.Timedelta(hours=PANEL_WINDOW_H)
        t1 = ev.end   + pd.Timedelta(hours=PANEL_WINDOW_H)

        fig, ax = plt.subplots(1, 1, figsize=(13, 3.6))
        fig.suptitle(
            f"HE_SPE [{panel_tag}] Event #{ei+1}  |  "
            f"Onset: {ev.onset.strftime('%Y-%m-%d %H:%M')}  "
            f"Peak count: {ev.peak_flux:.1f}",
            fontsize=10)

        c_seg = cnt.loc[t0:t1]
        t_seg = thresh.reindex(cnt.index).ffill().loc[t0:t1]
        ax.plot(c_seg.index, c_seg.values, color="#2c3e50", lw=1.0, label="count")
        ax.plot(t_seg.index, t_seg.values, color="red", ls=":", lw=1.2,
                label="rolling threshold")
        ax.axvline(ev.onset, ls="--", color="orange", lw=1, alpha=0.8, label="onset")
        ax.axvline(ev.peak,  ls=":",  color="red",    lw=1.2, alpha=0.6, label="peak")
        ax.axvline(ev.end,   ls="--", color="gray",   lw=1, alpha=0.8, label="end")
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Count [15-min]", fontsize=8)
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))

        fig.tight_layout()
        out = panel_dir / f"ana6_event_{ei+1:02d}_panel.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)

    print(f"  [ana6] {panel_tag} panels {len(events)} saved → {panel_dir}")


# ─────────────────────────────────────────────────────────────────
def main():
    print("[ana6] Loading data...")
    df_count, _, _ = load_data()

    cnt_rs = df_count.resample(RESAMPLE_FREQ).mean()

    event_records = []   # 전 이벤트 행
    summary_rows  = []   # 기준×채널별 이벤트 수

    for pd_key in COUNT_PD_KEYS:
        for side in COUNT_SIDES:
            for logic in HE_LOGICS:
                try:
                    cnt = cnt_rs[pd_key, side, logic].dropna()
                except KeyError:
                    continue
                if len(cnt) < 100:
                    continue

                print(f"\n[ana6] {pd_key}-{side} / {logic}  "
                      f"(n={len(cnt)} resampled pts)")
                thr_dict = build_threshold_series(cnt)

                for crit, thr in thr_dict.items():
                    evs = detect_events_rolling(
                        cnt, thr, MIN_SPE_DURATION_H,
                        peak_floor=PEAK_FLOOR,
                        label=f"{pd_key}{side}_{logic}_{crit}")

                    summary_rows.append({
                        "pd_key": pd_key, "side": side, "logic": logic,
                        "criterion": crit, "n_events": len(evs),
                    })

                    for ei, ev in enumerate(evs):
                        event_records.append({
                            "pd_key": pd_key, "side": side, "logic": logic,
                            "criterion": crit, "event_idx": ei + 1,
                            "onset": ev.onset, "peak": ev.peak, "end": ev.end,
                            "duration_h": round(
                                (ev.end - ev.onset).total_seconds() / 3600, 2),
                            "peak_count": round(ev.peak_flux, 2),
                        })

                    panel_tag = f"{crit}_{pd_key}{side}_{logic}"
                    plot_event_panels(cnt, thr, evs, panel_tag)

    # ── CSV 저장 ──────────────────────────────────────────────────
    df_ev = pd.DataFrame(event_records)
    out_ev = ANA6_OUTPUT_DIR / "ana6_rolling_spe_events.csv"
    df_ev.to_csv(out_ev, index=False)
    print(f"\n[ana6] events saved: {out_ev}  ({len(df_ev)} rows)")

    df_sum = pd.DataFrame(summary_rows)
    out_sum = ANA6_OUTPUT_DIR / "ana6_event_count_summary.csv"
    df_sum.to_csv(out_sum, index=False)
    print(f"[ana6] summary saved: {out_sum}")
    if not df_sum.empty:
        pivot = df_sum.pivot_table(
            index=["pd_key", "side", "logic"],
            columns="criterion", values="n_events", fill_value=0)
        print(pivot.to_string())


if __name__ == "__main__":
    main()