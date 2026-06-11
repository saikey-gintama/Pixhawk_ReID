"""
fsm_count_spe_blc1_fixed.py
===========================
BL-C1: 고정임계(fixed-threshold) SPE onset 검출기 — rolling 배경 추정을
완전히 제거한 baseline. 본 연구 제안 방식(rolling MAD-robust FSM)의 ablation
대조군으로, "매일 배경을 다시 추정하는 rolling이 정말 기여하는가"를 답한다.

다른 4종 FSM(quietoff_mad/quiet7_mad/quietoff_std/quiet7_std)과 검출 엔진
뒷단(1시간 지속 규칙, peak_floor 사후필터, 출력 스키마)은 동일하되, 배경
추정부(compute_rolling_bg, quiet 선택, robust σ)를 통째로 들어내고 임계를
시간 불변 상수 C 하나로 둔다:

    threshold(t) = C            (max/floor/rolling 없음 — 순수 고정임계)

C 결정 방식 두 가지 (--mode):
  const : 채널별 절대 상수 C를 직접 지정.
          기본값은 ana_event_count_profile.py가 산출한 C_fpr0.05
          (FPR≤0.05 제약 하 최대 TPR 임계)를 logic별 대표값으로 박음.
          Papaioannou(2014)류 "운영자가 채널별 경보 수준을 직접 지정"에 대응.
  pctl  : C = percentile(채널 전 기간 count, PCTL).
          기본 PCTL=95. ana 통계에서 quiet_p95 ≈ C_fpr0.05 로 수렴함을
          확인했기에 p95 채택. Papaioannou류 정온기 배경 분위수 방식.

⚠ max(C, onset_floor) 같은 혼합은 쓰지 않는다. floor와 C를 한 임계에 섞으면
  "C가 이긴 건지 floor가 이긴 건지" 분리가 안 돼 ablation 목적이 깨진다.
  BL-C1은 오직 C 하나로만 임계를 정의한다.

출력 스키마는 ana6_sweep_event.csv 규격(noaa_goes_spe_match / swpc_alert_espe_match
호환). k는 BL-C1에 개념이 없으므로 placeholder(0)로 채우고, 적용한 C는
onset_floor 컬럼에 기록한다(매칭 groupby 키 일관성 유지).

사용:
  # 분위수 고정임계 (전 채널, C=count p95)
  python fsm_count_spe_blc1_fixed.py --mode pctl --pctl 95

  # 절대상수 고정임계 (logic별 기본 C, 전 채널)
  python fsm_count_spe_blc1_fixed.py --mode const

  # peak_floor 조정
  python fsm_count_spe_blc1_fixed.py --mode pctl --pctl 95 --peak 2.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── ksem_io 경로 (count 로드 전용) ────────────────────────────────
_THIS_DIR         = Path(__file__).parent.resolve()
KSEM_COUNT_DIR    = _THIS_DIR.parent / "KSEM_count"
COUNT_PARQUET_DIR = KSEM_COUNT_DIR / "ksem_cache_parquet"

FSM_OUTPUT_DIR = _THIS_DIR / "fsm_output"
FSM_OUTPUT_DIR.mkdir(exist_ok=True)

if str(KSEM_COUNT_DIR) not in sys.path:
    sys.path.insert(0, str(KSEM_COUNT_DIR))
import ksem_io   # noqa: E402


# ── 파라미터 (다른 FSM과 동일) ────────────────────────────────────
RESAMPLE_FREQ        = "15min"   # 원본 1분 → 15분 평균 리샘플
MIN_SPE_DURATION_H   = 1.0       # count>=C가 이 시간 이상 연속이면 검출
MIN_PTS_PER_CHANNEL  = 100       # 포인트 부족 채널 스킵
DEFAULT_PEAK_FLOOR   = 2.0       # peak_count >= 이 값인 구간만 이벤트 인정

PD_KEYS  = ["PD1", "PD2", "PD3"]
SIDES    = ["A", "B"]
LOGICS   = ["O", "OU", "OUT", "F", "FT", "FTU", "FTUO", "CR"]

# logic→group (match 코드 _group과 동일)
def _group(logic: str) -> str:
    if logic in ("OU", "OUT"):
        return "A"
    if logic in ("FTU", "FTUO"):
        return "B"
    return "C"

# ── const 모드 기본 C (logic별 대표값) ────────────────────────────
# ana_event_count_profile.py 의 C_fpr0.05 (FPR≤0.05 제약 최적 임계) 를
# logic별 대표값으로 정리. 채널 개별값이 필요하면 CONST_C_BY_CHANNEL로 덮어쓴다.
#   양성자 트리거: OU/OUT ~0.2~0.6 (quiet bg 극저)
#   전자 트리거 : FTU ~35, FTUO ~16, FT ~높음(배경 큼), F 포화(부적합)
CONST_C_BY_LOGIC = {
    "OU":   0.30,    "OUT":  0.45,
    "FTU":  35.0,    "FTUO": 16.0,
    "FT":   1000.0,  "F":    20000.0,   # FT/F는 배경 큼 — 참고용(트리거 부적합)
    "O":    50.0,    "CR":   90.0,
}
# 채널 단위로 더 정밀하게 박고 싶을 때만 사용 (비우면 logic 기본값 사용)
CONST_C_BY_CHANNEL: dict[str, float] = {
    # 예) "PD1B-OU": 0.24, "PD2B-OU": 0.22, "PD1A-FTU": 35.0,
    "PD1A-OU": np.nan,   # 노이즈 채널 — const 모드에서도 의미 약함(참고)
}


# ─────────────────────────────────────────────────────────────────
def load_count() -> pd.DataFrame:
    print(f"[data] count 로드: {COUNT_PARQUET_DIR}", flush=True)
    df, _ = ksem_io.load(COUNT_PARQUET_DIR)
    if not df.empty and df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df.resample(RESAMPLE_FREQ).mean()
    print(f"[data] count shape: {df.shape}, "
          f"기간 {df.index[0].date()} ~ {df.index[-1].date()}")
    return df


def resolve_C(channel: str, logic: str, cnt: pd.Series,
              mode: str, pctl: float) -> float:
    """채널의 고정임계 C 결정.
    const: 채널별 지정값 → 없으면 logic 기본값.
    pctl : 채널 전 기간 count의 pctl 분위수."""
    if mode == "const":
        if channel in CONST_C_BY_CHANNEL and np.isfinite(CONST_C_BY_CHANNEL[channel]):
            return float(CONST_C_BY_CHANNEL[channel])
        return float(CONST_C_BY_LOGIC.get(logic, np.nan))
    # pctl
    return float(np.nanpercentile(cnt.values, pctl))


def detect_segments(cnt: pd.Series, C: float,
                    peak_floor: float) -> list[dict]:
    """count >= C 가 MIN_SPE_DURATION_H 이상 연속인 구간을 검출.
    다른 FSM의 detect_segments와 동일 로직 (임계가 상수 C라는 점만 다름).
    각 구간: onset/peak/end 시각 + peak_count + 배경통계."""
    if not np.isfinite(C):
        return []
    dt_h = RESAMPLE_FREQ
    min_pts = int(pd.Timedelta(hours=MIN_SPE_DURATION_H) / pd.Timedelta(dt_h))
    above = (cnt.values >= C)
    segs = []
    i, n = 0, len(cnt)
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j < n and above[j]:
            j += 1
        # [i, j) 가 연속 초과 구간
        if (j - i) >= max(min_pts, 1):
            seg = cnt.iloc[i:j]
            peak_count = float(seg.max())
            if peak_count >= peak_floor:        # peak_floor 사후필터
                peak_idx = seg.idxmax()
                segs.append({
                    "onset_time": seg.index[0],
                    "peak_time":  peak_idx,
                    "end_time":   seg.index[-1],
                    "peak_count": peak_count,
                    "duration_h": (seg.index[-1] - seg.index[0])
                                  / pd.Timedelta(hours=1),
                    "C": C,
                })
        i = j
    return segs


def run_blc1(df_count: pd.DataFrame, mode: str, pctl: float,
             peak_floor: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """전 채널 BL-C1 검출. 반환 (event_df, onset_df) — match 호환 스키마."""
    ev_rows, on_rows = [], []
    for pd_key in PD_KEYS:
        for side in SIDES:
            for logic in LOGICS:
                try:
                    cnt = df_count[pd_key, side, logic].dropna()
                except KeyError:
                    continue
                if len(cnt) < MIN_PTS_PER_CHANNEL:
                    continue
                channel = f"{pd_key}{side}-{logic}"
                C = resolve_C(channel, logic, cnt, mode, pctl)
                if not np.isfinite(C):
                    print(f"  [skip] {channel}: C 미정의(mode={mode})")
                    continue
                segs = detect_segments(cnt, C, peak_floor)
                print(f"  {channel:11s} C={C:9.3f}  group={_group(logic)}  "
                      f"검출 {len(segs)}건")
                for s in segs:
                    # match 호환: channel/k/onset_floor/peak_floor + 검출시각
                    # k는 BL-C1에 없으므로 0(placeholder), 적용 C는 onset_floor에 기록
                    row = {
                        "channel": channel,
                        "k": 0,                       # placeholder (rolling 없음)
                        "onset_floor": round(C, 4),   # 적용한 고정임계 C
                        "peak_floor": peak_floor,
                        "group": _group(logic),
                        "onset_time": s["onset_time"],
                        "peak_time":  s["peak_time"],
                        "end_time":   s["end_time"],
                        "max_pfu":    s["peak_count"],  # 검출 peak count (match가 읽는 세기)
                        "duration_h": round(s["duration_h"], 2),
                    }
                    ev_rows.append(row)
                    on_rows.append({k: row[k] for k in
                                    ("channel", "k", "onset_floor", "peak_floor",
                                     "group", "onset_time")})
    ev_df = pd.DataFrame(ev_rows)
    on_df = pd.DataFrame(on_rows)
    return ev_df, on_df


def main():
    ap = argparse.ArgumentParser(
        description="BL-C1 고정임계 SPE 검출기 (rolling 제거 baseline)")
    ap.add_argument("--mode", choices=["const", "pctl"], default="pctl",
                    help="const: 채널/logic별 절대 C, pctl: count 분위수 C")
    ap.add_argument("--pctl", type=float, default=95.0,
                    help="pctl 모드의 분위수 (기본 95 — quiet_p95≈C_fpr0.05 수렴)")
    ap.add_argument("--peak", type=float, default=DEFAULT_PEAK_FLOOR,
                    help="peak_floor 사후필터 (기본 2.0)")
    args = ap.parse_args()

    df_count = load_count()

    if args.mode == "pctl":
        tag = f"blc1_pctl{args.pctl:g}_pk{args.peak:g}"
    else:
        tag = f"blc1_const_pk{args.peak:g}"

    print(f"\n=== BL-C1 고정임계 검출  (mode={args.mode}, tag={tag}) ===")
    ev_df, on_df = run_blc1(df_count, args.mode, args.pctl, args.peak)

    outdir = FSM_OUTPUT_DIR / tag
    outdir.mkdir(parents=True, exist_ok=True)
    ev_path = outdir / f"fsm_event_{tag}.csv"
    on_path = outdir / f"fsm_onset_{tag}.csv"
    ev_df.to_csv(ev_path, index=False)
    on_df.to_csv(on_path, index=False)

    n_ev = len(ev_df)
    n_ch = ev_df["channel"].nunique() if n_ev else 0
    print(f"\n[완료] 검출 이벤트 {n_ev}건 / {n_ch}채널")
    print(f"  event → {ev_path}")
    print(f"  onset → {on_path}")
    print(f"\n매칭 예:")
    print(f"  python ..\\NOAA_GOES\\noaa_goes_spe_match.py "
          f"--events {ev_path} --catalog ..\\NOAA_GOES\\noaa_goes_spe_cache_parquet "
          f"--count-dir .\\ksem_cache_parquet --k 0 --onset {{C}} --peak {args.peak:g}")
    print(f"  (BL-C1은 채널마다 C(=onset_floor)가 다르므로, 매칭 시 단일 "
          f"--onset 으로 거르지 말고 sweep 표 전체를 보거나 channel별로 확인)")


if __name__ == "__main__":
    main()