"""
fsm_count_spe_blc1_fixed.py
===========================
BL-C1: 고정임계(fixed-threshold) SPE onset 검출기 — rolling 배경 추정을
완전히 제거한 baseline. 본 연구 제안 방식(rolling MAD-robust FSM)의 ablation
대조군으로, "매일 배경을 다시 추정하는 rolling이 정말 기여하는가"를 답한다.

이 파일은 quietoff_mad / blc1_lowe 와 **검출 엔진 뒷단을 완전히 공유**한다.
FSM끼리는 배경 추정부만 달라야 한다는 원칙에 맞춰, 아래 항목은 4종과 동일하다:
  - build_threshold 의 onset_floor 하한 클립        (모든 FSM 공통)
  - detect_segments (peak 필터 없음, 전체 seg 반환)  (byte 단위 동일)
  - peak_floor 는 main 에서만 거는 사후필터          (onset 목록 미오염)
  - 출력 스키마 ONSET_COLS / EVENT_COLS              (quietoff_mad 와 동일)

배경 추정부만 교체했다:
  compute_const_bg : rolling 없이 bg_median = 상수 C, bg_std = NaN
  build_threshold  : T(t) = clip(C, lower=onset_floor)

  ┌─ 왜 onset_floor 를 거는가 (이전판과의 차이) ──────────────────────┐
  │ 실제 ROS2 노드(wp_ksem_node)는 threshold=max(bg+K·σ, ONSET_FLOOR) │
  │ 로 onset_floor 를 항상 건다. 비교 공정성을 위해 BL-C1 도 동일한    │
  │ floor 를 공유해야 한다. floor 는 BL-C1 고유가 아니라 모든 방식이   │
  │ 똑같이 쓰는 뒷단 요소이므로, "rolling vs 상수 C" 대조는 floor 위쪽 │
  │ 에서 그대로 보존된다. 단, C < onset_floor 인 저카운트 채널에서는   │
  │ C 가 floor 로 들려 올라가 그 구간이 비-식별 영역이 된다(아래 주의).│
  └──────────────────────────────────────────────────────────────────┘

C 결정 방식 두 가지 (--mode):
  const : 채널별 절대 상수 C. 우선순위:
            1순위 — --const-csv <path> CSV의 channel→C_fpr0.05 (per-channel 실측).
            2순위 — CONST_C_BY_CHANNEL 오버라이드(하드코딩, NaN이면 skip).
            3순위 — CONST_C_BY_LOGIC fallback(하드코딩 근사값; 경고 출력됨).
          CSV 미지정 시 2·3순위만 작동.
          ana_event_count_profile.py 의 C_fpr0.05 (FPR≤0.05 제약 최대 TPR).
  pctl  : C = percentile(채널 전 기간 count, PCTL). 기본 PCTL=95.

⚠ 주의 (const 모드 × onset_floor 상호작용):
  onset_floor=0.5 가 모든 FSM에 균일 적용되므로, C < 0.5 인 양성자 트리거
  채널(예: OU=0.30, OUT=0.45)은 실효 임계가 0.5로 클립된다. 이 구간에서는
  BL-C1 과 rolling 방식이 동일하게 floor 에 묶여 식별력이 없다.

출력 스키마는 quietoff_mad / blc1_lowe 와 동일(noaa_goes_spe_match 호환).
k 는 BL-C1에 개념이 없어 placeholder(0). 적용한 C 는 bg_median / threshold
컬럼에 남고, onset_floor 컬럼은 다른 FSM과 동일하게 floor 값(0.5)을 기록한다.

사용:
  # 분위수 고정임계 (전 채널, C=count p95)
  python fsm_count_spe_blc1_fixed.py --mode pctl --pctl 95

  # CSV per-channel C_fpr0.05 고정임계
  python fsm_count_spe_blc1_fixed.py --mode const \
      --const-csv ../KSEM_count/ana_output/noaa_spe_event_count_stats.csv

자립성: ksem_flux_config / ksem_common 비의존. count 로드만 ksem_io 사용.
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ── ksem_io 경로 (count 로드 전용) ────────────────────────────────
_THIS_DIR         = Path(__file__).parent.resolve()
KSEM_COUNT_DIR    = _THIS_DIR.parent / "KSEM_count"
COUNT_PARQUET_DIR = KSEM_COUNT_DIR / "ksem_cache_parquet"
if str(KSEM_COUNT_DIR) not in sys.path:
    sys.path.insert(0, str(KSEM_COUNT_DIR))
import ksem_io   # noqa: E402

# ══════════════════════════════════════════════════════════════════
# 파라미터 (이 블록만 조정 — 검출 로직 본문엔 숫자를 박지 않는다)
# ══════════════════════════════════════════════════════════════════
RESAMPLE_FREQ      = "15min"
MIN_SPE_DURATION_H = 1

# ── 이 파일의 방식 고정 (BL-C1 고정임계) ─────────────────────────
DEFAULT_MODE = "pctl"
DEFAULT_PCTL = 95.0

PROTON_LOGICS   = ["O", "OU", "OUT", "CR"]
ELECTRON_LOGICS = ["F", "FT", "FTU", "FTUO"]
TARGET_LOGICS   = PROTON_LOGICS + ELECTRON_LOGICS
PD_KEYS         = ["PD1", "PD2", "PD3"]
SIDES           = ["A", "B"]

ONSET_FLOOR = 0.5    # 모든 FSM 공통 floor (wp_ksem_node 와 동일)
PEAK_FLOOR  = 2.0
MIN_PTS_PER_CHANNEL = 100

TAG = "blc1_fixed"

# ── const 모드 fallback C (logic별 근사값) ────────────────────────
# CSV(--const-csv)가 없거나 채널이 CSV에 없을 때만 사용. 발동 시 경고 출력.
# CSV가 있을 때는 CSV per-channel 값이 우선한다.
CONST_C_BY_LOGIC = {
    "OU":   0.30,    "OUT":  0.45,
    "FTU":  35.0,    "FTUO": 16.0,
    "FT":   1000.0,  "F":    20000.0,   # FT/F는 배경 큼 — 참고용
    "O":    50.0,    "CR":   90.0,
}
# 채널 단위 예외 오버라이드 (NaN = 해당 채널 const 모드 skip)
CONST_C_BY_CHANNEL: dict[str, float] = {
    "PD1A-OU": np.nan,   # 노이즈 채널 — const 모드에서도 의미 약함(참고)
}


# ══════════════════════════════════════════════════════════════════
# CSV 로드
# ══════════════════════════════════════════════════════════════════
def load_const_csv(path: str | None) -> dict[str, float]:
    """stats CSV(channel, C_fpr0.05) → {channel: float} 매핑 로드.
    파일 없거나 실패하면 {} 반환(fallback 전용 모드로 동작)."""
    if not path:
        return {}
    try:
        df = pd.read_csv(path, usecols=lambda c: c in ("channel", "C_fpr0.05"))
        result = {}
        for _, row in df.iterrows():
            try:
                c = float(row["C_fpr0.05"])
                if np.isfinite(c):
                    result[str(row["channel"])] = c
            except (ValueError, KeyError):
                pass
        print(f"[{TAG}] const-csv 로드: {Path(path).name}  ({len(result)}채널)")
        return result
    except Exception as e:
        print(f"[{TAG}] WARN const-csv 로드 실패({e}) → fallback 전용 모드")
        return {}


def parse_args():
    p = argparse.ArgumentParser(
        description=f"FSM count SPE detector [{TAG}] — fixed-threshold C "
                    f"(rolling 제거 ablation). params override defaults, "
                    f"output names auto-built from TAG+params.")
    p.add_argument("--mode", choices=["const", "pctl"], default=DEFAULT_MODE)
    p.add_argument("--pctl", type=float, default=DEFAULT_PCTL,
                   help=f"pctl 모드의 분위수 (default {DEFAULT_PCTL})")
    p.add_argument("--const-csv", default=None,
                   help="const 모드 채널별 C_fpr0.05 CSV 경로 "
                        "(없으면 CONST_C_BY_LOGIC fallback만 사용)")
    p.add_argument("--onset", type=float, default=ONSET_FLOOR)
    p.add_argument("--peak",  type=float, default=PEAK_FLOOR)
    return p.parse_args()


def build_runtag(tag, mode, pctl, onset, peak):
    cstr = f"pctl{_numstr(pctl)}" if mode == "pctl" else "const"
    return f"{tag}_{cstr}_on{_numstr(onset)}_pk{_numstr(peak)}"


def _numstr(v):
    f = float(v)
    return str(int(f)) if f.is_integer() else str(f)


FSM_OUTPUT_DIR = _THIS_DIR / "fsm2_output"
FSM_OUTPUT_DIR.mkdir(exist_ok=True)


def _group(logic: str) -> str:
    if logic in ("OU", "OUT"):
        return "A"
    if logic in ("FTU", "FTUO"):
        return "B"
    return "C"


# ══════════════════════════════════════════════════════════════════
# C 결정 (배경 추정부 — 이 함수만 다른 FSM과 다르다)
# ══════════════════════════════════════════════════════════════════
def resolve_C(channel: str, logic: str, cnt: pd.Series,
              mode: str, pctl: float,
              const_csv_map: dict | None = None) -> tuple[float, str]:
    """채널의 고정임계 C 결정.
    반환: (C, source)
      source = 'csv'             — CSV per-channel C_fpr0.05
             | 'fallback_channel'— CONST_C_BY_CHANNEL 오버라이드
             | 'fallback_logic'  — CONST_C_BY_LOGIC 근사값(경고 필요)
             | 'pctl'            — percentile
    """
    if mode == "const":
        # 1순위: CSV per-channel
        if const_csv_map and channel in const_csv_map:
            return float(const_csv_map[channel]), "csv"
        # 2순위: CONST_C_BY_CHANNEL 명시 오버라이드
        if channel in CONST_C_BY_CHANNEL:
            v = CONST_C_BY_CHANNEL[channel]
            return float(v), "fallback_channel"   # NaN이면 호출부에서 skip
        # 3순위: CONST_C_BY_LOGIC fallback
        return float(CONST_C_BY_LOGIC.get(logic, np.nan)), "fallback_logic"
    # pctl 모드
    return float(np.nanpercentile(cnt.values, pctl)), "pctl"


# ══════════════════════════════════════════════════════════════════
# 배경 추정: 고정임계 C (rolling 없음) ─ 다른 FSM과 byte 동일한 인터페이스
# ══════════════════════════════════════════════════════════════════
def compute_const_bg(cnt: pd.Series, C: float) -> pd.DataFrame:
    """고정임계용 '배경' 시계열. rolling 갱신이 없으므로 bg_median=C 상수,
    bg_std=NaN. detect_segments / build_threshold 가 다른 FSM과 동일한
    bg[bg_median, bg_std] 인터페이스를 그대로 받도록 프레임만 맞춘다.
    반환: DataFrame[bg_median, bg_std] (cnt.index 정렬)."""
    if cnt.empty:
        return pd.DataFrame(columns=["bg_median", "bg_std"])
    return pd.DataFrame(
        {"bg_median": np.full(len(cnt), float(C)),
         "bg_std":    np.full(len(cnt), np.nan)},
        index=cnt.index,
    )


def build_threshold(bg: pd.DataFrame, onset_floor: float) -> pd.Series:
    """임계 = C(=bg_median)를 onset_floor로 하한 클립.
    다른 FSM의 build_threshold와 동일하게 floor 클립을 공유한다
    (BL-C1은 산포항 k·σ 가 없어 bg_median 자체가 곧 임계)."""
    th = bg["bg_median"].copy()
    if onset_floor > 0:
        th = th.clip(lower=onset_floor)
    return th


# ══════════════════════════════════════════════════════════════════
# 검출 엔진 (quietoff_mad / blc1_lowe 와 byte 단위 동일 — peak 필터 없음)
# ══════════════════════════════════════════════════════════════════
def detect_segments(cnt: pd.Series, thresh_series: pd.Series,
                    bg: pd.DataFrame, min_duration_h: float) -> list:
    """count >= 임계가 min_duration_h 이상 연속인 구간. peak_floor는 호출부에서."""
    th  = thresh_series.reindex(cnt.index).ffill()
    med = bg["bg_median"].reindex(cnt.index).ffill()
    sig = bg["bg_std"].reindex(cnt.index).ffill()
    above = (cnt >= th) & cnt.notna() & th.notna()
    segs = []; in_ev = False; onset = None
    def _close(seg, onset_t, end_t):
        if (end_t - onset_t).total_seconds() / 3600 < min_duration_h:
            return
        segs.append({
            "onset_time": onset_t, "peak_time": seg.idxmax(), "end_time": end_t,
            "onset_count": round(float(cnt.loc[onset_t]), 3),
            "peak_count":  round(float(seg.max()), 3),
            "end_count":   round(float(cnt.loc[end_t]), 3),
            "duration_h":  round((end_t - onset_t).total_seconds() / 3600, 2),
            "bg_median": round(float(med.loc[onset_t]), 4) if pd.notna(med.loc[onset_t]) else np.nan,
            "bg_sigma":  round(float(sig.loc[onset_t]), 4) if pd.notna(sig.loc[onset_t]) else np.nan,
            "threshold": round(float(th.loc[onset_t]), 4),
        })
    for t, a in above.items():
        if a and not in_ev:
            in_ev, onset = True, t
        elif not a and in_ev:
            _close(cnt.loc[onset:t], onset, t); in_ev = False
    if in_ev and onset is not None:
        seg = cnt.loc[onset:]; _close(seg, onset, seg.index[-1])
    return segs


# ══════════════════════════════════════════════════════════════════
def load_count() -> pd.DataFrame:
    df_count, _ = ksem_io.load(COUNT_PARQUET_DIR)
    if not df_count.empty and df_count.index.tz is None:
        df_count.index = df_count.index.tz_localize("UTC")
    return df_count


def main():
    args = parse_args()
    mode     = args.mode
    pctl     = args.pctl
    onset_fl = args.onset
    peak_fl  = args.peak
    runtag   = build_runtag(TAG, mode, pctl, onset_fl, peak_fl)
    out_dir  = FSM_OUTPUT_DIR / runtag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fsm:{TAG}] params: mode={mode} pctl={pctl} onset={onset_fl} "
          f"peak={peak_fl}")
    print(f"[fsm:{TAG}] runtag: {runtag}")

    # ── const-csv 로드 및 검증 출력 ──────────────────────────────
    const_csv_map = load_const_csv(args.const_csv) if mode == "const" else {}

    if mode == "const" and const_csv_map:
        VERIFY_CHANNELS = ["PD1A-OU", "PD1B-OU", "PD2B-OU", "PD3A-OUT", "PD1A-FTU"]
        print(f"\n[{TAG}] const-csv 로드 검증 (샘플 채널):")
        for ch in VERIFY_CHANNELS:
            if ch in const_csv_map:
                print(f"  {ch:12s} → C_fpr0.05 = {const_csv_map[ch]:.4f}  (csv)")
            else:
                fb = CONST_C_BY_LOGIC.get(ch.split("-")[-1], np.nan)
                print(f"  {ch:12s} → (CSV 없음) fallback = {fb}")
    elif mode == "const":
        print(f"\n[{TAG}] --const-csv 미지정 → CONST_C_BY_LOGIC fallback 전용 모드")

    print(f"\n[fsm:{TAG}] Loading count data...")
    df_count = load_count()
    cnt_rs = df_count.resample(RESAMPLE_FREQ).mean()

    onset_rows, event_rows = [], []
    fallback_channels: list[tuple[str, str, float]] = []  # (chan, logic, C)

    for pd_key in PD_KEYS:
        for side in SIDES:
            for logic in TARGET_LOGICS:
                try:
                    cnt = cnt_rs[pd_key, side, logic].dropna()
                except KeyError:
                    continue
                if len(cnt) < MIN_PTS_PER_CHANNEL:
                    continue
                chan = f"{pd_key}{side}-{logic}"
                C, c_src = resolve_C(chan, logic, cnt, mode, pctl, const_csv_map)

                # fallback 발동 시 즉시 경고
                if c_src == "fallback_logic":
                    fallback_channels.append((chan, logic, C))
                    print(f"[fallback] {chan}: CSV에 C_fpr0.05 없음 → "
                          f"logic 근사값 {C} 사용")
                elif c_src == "fallback_channel":
                    fallback_channels.append((chan, logic, C))
                    if np.isfinite(C):
                        print(f"[fallback] {chan}: CSV에 C_fpr0.05 없음 → "
                              f"channel 오버라이드값 {C} 사용")

                if not np.isfinite(C):
                    print(f"\n[fsm:{TAG}] {chan}  C 미정의(src={c_src}) — skip")
                    continue

                th_eff = max(C, onset_fl) if onset_fl > 0 else C
                print(f"\n[fsm:{TAG}] {chan}  (n={len(cnt)} pts)  "
                      f"C={C:.4f} [{c_src}] → th={th_eff:.4f}  group={_group(logic)}")
                bg   = compute_const_bg(cnt, C)
                thr  = build_threshold(bg, onset_fl)
                segs = detect_segments(cnt, thr, bg, MIN_SPE_DURATION_H)
                base = {"k": 0, "onset_floor": onset_fl, "pd_key": pd_key,
                        "side": side, "logic": logic, "channel": chan}
                for s in segs:
                    onset_rows.append({**base, **s})
                passed = [s for s in segs if s["peak_count"] >= peak_fl]
                for s in passed:
                    event_rows.append({**base, "peak_floor": peak_fl, **s})
                print(f"    segs={len(segs)}  peak>={peak_fl}: {len(passed)}")

    # ── fallback 요약 ─────────────────────────────────────────────
    if mode == "const":
        if fallback_channels:
            print(f"\n[{TAG}] ⚠ fallback 발동: {len(fallback_channels)}채널 "
                  f"(CSV에 C_fpr0.05 없거나 미지정)")
            for ch, lg, cv in fallback_channels:
                print(f"    {ch} (logic={lg}): C={cv}")
        else:
            print(f"\n[{TAG}] fallback 발동: 0채널 (전 채널 CSV 로드 성공)")

    ONSET_COLS = ["k","onset_floor","pd_key","side","logic","channel",
                  "onset_time","peak_time","end_time","onset_count","peak_count",
                  "end_count","duration_h","bg_median","bg_sigma","threshold"]
    EVENT_COLS = ["k","onset_floor","peak_floor","pd_key","side","logic","channel",
                  "onset_time","peak_time","end_time","onset_count","peak_count",
                  "end_count","duration_h","bg_median","bg_sigma","threshold"]
    df_on = pd.DataFrame(onset_rows, columns=ONSET_COLS)
    df_ev = pd.DataFrame(event_rows, columns=EVENT_COLS)
    out_on = out_dir / f"fsm_onset_{runtag}.csv"
    out_ev = out_dir / f"fsm_event_{runtag}.csv"
    df_on.to_csv(out_on, index=False)
    df_ev.to_csv(out_ev, index=False)
    print(f"\n[fsm:{TAG}] onset saved: {out_on}  ({len(df_on)} rows)")
    print(f"[fsm:{TAG}] event saved: {out_ev}  ({len(df_ev)} rows)")
    print(f"[fsm:{TAG}] METHOD=fixed-C  mode={mode} pctl={pctl} "
          f"floor={onset_fl} peak={peak_fl}")
    if not df_on.empty:
        n_on = df_on.groupby("channel").size().rename("n_onset")
        n_ev = (df_ev.groupby("channel").size().rename("n_event")
                if not df_ev.empty else pd.Series(dtype=int, name="n_event"))
        tbl = (pd.concat([n_on, n_ev], axis=1).fillna(0)
                 .astype(int).sort_values("n_onset", ascending=False))
        print(f"\n[fsm:{TAG}] per-channel counts:")
        print(tbl.to_string())


if __name__ == "__main__":
    main()
