"""
ksem_bl_c2.py  —  BL-C2: Löwe et al. (2025) Reactive Nowcasting
=================================================================
Löwe, J. L. et al. (2025). Nowcasting Solar Energetic Particle Events
for Mars Missions. Space Weather, 23, e2025SW004372.

[ 논문 Section 3 방법론 완전 재현 ]

Background 추정 (논문 Section 3 / Fig. 3):
  - 이벤트 직전 5일치 데이터에 "constant linear fit" → 수평 상수(mean)
    논문 원문: "We apply a constant linear fit to the dose rate E
    measured in 15-min cadences over the five days preceding the SEP event."
    Figure 3 좌측 패널에서 Background가 수평 점선으로 표시됨.
  - background(t) = mean(직전 5일) = 상수
  - "updated daily": 하루 1회 갱신
  - ALERT 진행 중 fitting 중단(freeze), 종료 후 재개
  - history(직전 5일) 데이터가 충분하지 않은 날은 FSM에서 제외(skip)

트리거 조건 (논문 Fig.4 case 2):
  - threshold = background x 1.25
  - 단일 측정 초과  → PRE_ALERT  (1차 경고, "begin preparing")
  - 연속 2회 초과   → ALERT      (확정, "seek shelter ASAP")

이벤트 종료:
  - value < threshold → NOMINAL 복귀, fitting 재개

FAR (논문 Eq. 1):
  FAR = false_trigger / (false_trigger + correct_trigger)

Lead Time (논문 Figure 3 right):
  트리거(PRE_ALERT onset) 시각 → 총 SEP dose의 90% 누적 지점까지 남은 시간(분)

폴더 구조:
  D:\\Raw Count\\
    201905\\   ← YYYYMM
      20190501_PD1_Raw Count.csv
      20190501_PD2_Raw Count.csv
      20190501_PD3_Raw Count.csv
      ...
    201906\\
    ...
    202412\\

사용법:
  # 단일 날짜 (테스트)
  python ksem_bl_c2.py --root "D:\\Raw Count" --date 20240510

  # 날짜 범위 전체 분석
  python ksem_bl_c2.py --root "D:\\Raw Count" --start 201905 --end 202412

수정 이력:
  v2  2025-05  논문 compliance 4개 항목 수정
      [Fix 1] constant linear fit → np.mean() 상수 (slope 제거)
      [Fix 2] 배경 갱신 주기 → 하루 1회 (매분 갱신 제거)
      [Fix 3] run_range 배경 동결 — FSM과 연동하여 이벤트 중 freeze
      [Fix 4] run_range / run_single_date 배경 추정 통일
               (동일한 fit_background() 사용)

  v3  2025-05  인계 문서 기반 추가 수정
      [Fix A] A+B 합산: (s_A + s_B) / 2 → s_A + s_B
              (검출기 A, B는 반대 방향 독립 검출 → 합산이 올바름)
      [Fix B] 15분 리샘플링: 논문 cadence(15-min) 재현
              로더에서 1분 원본을 유지, run_range/run_single_date 진입 직전
              resample('15min').mean() 적용
      [Fix C] MIN_PD_VOTE_ALERT: 2 → 1
              (3축 PD는 입자 방향에 따라 1개 PD에만 강하게 잡힐 수 있음;
               FAR 억제는 "연속 2포인트(=30분)" 조건이 담당)
      [Fix D] 플롯 Panel 2 레전드: Patch → Line2D (수직선과 일치)
      [Fix E] history 없는 첫 5일치 skip:
              run_fsm_with_daily_bg에서 직전 5일 window가 비어있는 날은
              FSM 처리 자체를 건너뜀 (fallback bg로 처리하지 않음)
              run_single_date도 history=None이면 분석 skip
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

# ============================================================
# 0.  CONFIG  — 논문 고정값
# ============================================================

# 논문 Section 3
TRIGGER_PCT            = 20.00   # threshold = background x 1.25
CONSECUTIVE_FOR_ALERT  = 2      # 연속 N회 초과 → ALERT (논문 case 2, 15분 x 2=30분)
BACKGROUND_WINDOW_DAYS = 5      # 직전 5일 preceding window

# KSEM 데이터
CADENCE_MIN = 1                 # 원본 1분 cadence (로더 단계)
RESAMPLE_MIN = 15               # 논문 cadence: 15분 리샘플링 후 FSM 투입
MIN_HISTORY_PTS = int(BACKGROUND_WINDOW_DAYS * 24 * 60 / RESAMPLE_MIN)  # 480포인트, 5일치 history가 있어야 계산 시작 

# Proton bin 정의 (KSEM: O / OU / OUT event logic)
# O   event logic: bin index 1~23  (50~6000 keV)
# OU  event logic: bin index 25~41
# OUT event logic: bin index 46~56
PROTON_BINS = (
    list(range(1, 24)) +    # O
    list(range(25, 42)) +   # OU
    list(range(46, 57))     # OUT
)
EXCLUDE_BINS = {0, 24, 45, 57, 81, 105, 117, 127}  # underflow / trash

# 투표 임계
# [Fix C] ALERT: 3축 중 1개라도 ALERT → 경보 (OR 조건)
#          FAR 억제는 "연속 2포인트(=30분)" 조건이 담당
MIN_PD_VOTE_PRE   = 1   # PRE_ALERT: PD 1개 이상
MIN_PD_VOTE_ALERT = 1   # ALERT:     PD 1개 이상 (v2: 2 → v3: 1)

# 출력
PLOT_DPI = 150


# ============================================================
# 1.  STATE
# ============================================================

class State(Enum):
    NOMINAL   = 0   # 정상
    PRE_ALERT = 1   # 단일 측정 초과 → 준비 시작
    ALERT     = 2   # 연속 2회 초과 → 즉시 대피

STATE_COLOR = {
    State.NOMINAL:   "#4a9ede",
    State.PRE_ALERT: "#f5a623",
    State.ALERT:     "#c0392b",
}


# ============================================================
# 2.  LOADER
# ============================================================

def _parse_bin_idx(col: str, prefix: str) -> Optional[int]:
    try:
        return int(col[len(prefix):].split("(")[0])
    except Exception:
        return None


def load_one_csv(filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """CSV 1개 → (df_A, df_B): index=Timestamp, columns=bin_index(int)."""
    raw = pd.read_csv(filepath)
    raw["Time"] = pd.to_datetime(raw["Time"])
    raw = raw.set_index("Time").sort_index()

    def extract(prefix: str) -> pd.DataFrame:
        m = {c: _parse_bin_idx(c, prefix)
             for c in raw.columns if c.startswith(prefix)}
        m = {c: i for c, i in m.items() if i is not None}
        df = raw[list(m)].copy()
        df.columns = [m[c] for c in df.columns]
        return df[sorted(df.columns)]

    return extract("A"), extract("B")


def ym_folders(root: Path, start_ym: str, end_ym: str) -> List[Path]:
    """root 아래 YYYYMM 폴더를 start_ym ~ end_ym 범위로 정렬 반환."""
    folders = sorted([f for f in root.iterdir()
                      if f.is_dir() and f.name.isdigit()
                      and len(f.name) == 6])
    return [f for f in folders if start_ym <= f.name <= end_ym]


def find_pd_files_by_date(folder: Path, pd_key: str) -> List[Path]:
    """월 폴더 안에서 pd_key에 해당하는 CSV를 날짜순으로 모두 반환."""
    matches = sorted([f for f in folder.glob("*.csv")
                      if pd_key.lower() in f.name.lower()])
    return matches


def proton_total_from_csv(filepath: Path) -> pd.Series:
    """
    CSV 1개 → proton total (O+OU+OUT, A+B 합산) Series.

    [Fix A] 검출기 A와 B는 서로 반대 방향을 향하는 독립 검출기이므로
    전방향 총 계수 = A + B (평균이 아닌 합산).
    기존 / 2.0 제거.
    """
    df_A, df_B = load_one_csv(str(filepath))
    bins_A = [b for b in PROTON_BINS if b in df_A.columns
              and b not in EXCLUDE_BINS]
    bins_B = [b for b in PROTON_BINS if b in df_B.columns
              and b not in EXCLUDE_BINS]
    s_A = df_A[bins_A].sum(axis=1)
    s_B = df_B[bins_B].sum(axis=1)
    # [Fix A] 합산 (÷2 제거)
    return (s_A + s_B).rename("proton_total")


def load_series_range(root: Path,
                      start_ym: str,
                      end_ym: str,
                      pd_key: str) -> pd.Series:
    """YYYYMM 폴더들에서 pd_key CSV를 날짜순으로 모두 로드해 연속 Series 반환.
    반환값은 원본 1분 cadence."""
    folders = ym_folders(root, start_ym, end_ym)
    parts = []
    for folder in folders:
        files = find_pd_files_by_date(folder, pd_key)
        if not files:
            print(f"  [WARN] {folder.name}: {pd_key} 파일 없음")
            continue
        for fp in files:
            try:
                s = proton_total_from_csv(fp)
                parts.append(s)
            except Exception as e:
                print(f"  [WARN] {fp.name} 로드 실패: {e}")
    if not parts:
        raise FileNotFoundError(
            f"{pd_key} CSV를 찾을 수 없습니다: {root}/{start_ym}~{end_ym}")
    full = pd.concat(parts).sort_index()
    full = full[~full.index.duplicated(keep="first")]
    return full


def load_single_date(root: Path,
                     date_str: str,
                     pd_key: str) -> Tuple[pd.Series, Optional[pd.Series]]:
    """
    단일 날짜(YYYYMMDD)의 당일 Series와 직전 5일 history Series 반환.
    같은 월 폴더 + 이전 월 폴더에서 history 탐색.
    반환값은 원본 1분 cadence.
    """
    target_date   = pd.Timestamp(date_str)
    ym_target     = target_date.strftime("%Y%m")
    folder        = root / ym_target

    if not folder.exists():
        raise FileNotFoundError(f"폴더 없음: {folder}")

    date_prefix = target_date.strftime("%Y%m%d")
    day_files   = [f for f in folder.glob(f"{date_prefix}*{pd_key}*.csv")]
    if not day_files:
        day_files = find_pd_files_by_date(folder, pd_key)
    if not day_files:
        raise FileNotFoundError(f"{pd_key} not found for {date_str} in {folder}")

    parts_cur = []
    for fp in day_files:
        s        = proton_total_from_csv(fp)
        day_mask = s.index.date == target_date.date()
        if day_mask.sum() > 0:
            parts_cur.append(s[day_mask])
    if not parts_cur:
        raise FileNotFoundError(f"{pd_key}: {date_str} 데이터 없음")
    current = pd.concat(parts_cur).sort_index()
    current = current[~current.index.duplicated(keep="first")]

    # history: 직전 5일치
    hist_end      = target_date - pd.Timedelta(minutes=1)
    hist_start    = target_date - pd.Timedelta(days=BACKGROUND_WINDOW_DAYS)
    hist_ym_start = hist_start.strftime("%Y%m")

    try:
        hist_full = load_series_range(root, hist_ym_start, ym_target, pd_key)
        history   = hist_full.loc[hist_start:hist_end]
        if len(history) == 0:
            history = None
    except Exception:
        history = None

    return current, history


# ============================================================
# 3.  BACKGROUND ESTIMATOR  (논문 Section 3)
#
# [Fix 1] "constant linear fit" = 상수(mean).
#          논문 Figure 3 좌측: background가 수평 점선(horizontal dashed line).
#          slope 있는 linregress → np.mean() 으로 교체.
#
# [Fix 2] "updated daily": 호출 단위를 하루 1회로 관리하는 것은
#          run_fsm_with_daily_bg() 에서 담당.
#          이 함수는 주어진 history로부터 단순 상수 background를 반환.
# ============================================================

def fit_background(history: Optional[pd.Series],
                   current_len: int,
                   current_index: pd.DatetimeIndex
                   ) -> Tuple[Optional[pd.Series], str]:
    """
    논문 방법 (constant linear fit = 수평 상수):
      background = mean(직전 5일 데이터)  ← 상수, 기울기 없음

    [Fix E] history가 None이면 None을 반환 → 호출부에서 skip 처리.

    Returns
    -------
    bg      : pd.Series 또는 None (history 없으면 None)
    method  : str
    """
    if history is not None and len(history) > 0:
        y    = history.values.astype(float)
        mask = np.isfinite(y) & (y > 0)
        if mask.sum() >= MIN_HISTORY_PTS:
            bg_val = float(np.mean(y[mask]))
            method = "constant_mean_5day"
        else:
            return None, "no_valid_history"
        bg_val = max(bg_val, 1.0)
        return (pd.Series(np.full(current_len, bg_val), index=current_index),
                method)

    # [Fix E] history 없음 → skip 신호
    return None, "no_history_skip"


# ============================================================
# 4.  FSM  (논문 Figure 4 case 2)
# ============================================================

class FSMContext:
    """FSM 내부 상태."""
    def __init__(self):
        self.state         = State.NOMINAL
        self.consec_above  = 0        # 연속 threshold 초과 횟수
        self.in_event      = False    # 이벤트 진행 중 여부
        self.event_start   = None     # 이벤트 시작 타임스탬프
        self.bg_frozen     = None     # ALERT 중 freeze된 background 값
        self.event_cumsum  = 0.0      # 누적 SEP dose (bg 제거)
        self.event_peak    = 0.0      # 이벤트 내 최대값

    def start_event(self, t: pd.Timestamp):
        self.in_event     = True
        self.event_start  = t
        self.event_cumsum = 0.0
        self.event_peak   = 0.0

    def end_event(self):
        self.in_event     = False
        self.event_start  = None
        self.bg_frozen    = None
        self.consec_above = 0


def fsm_step(ctx: FSMContext,
             t: pd.Timestamp,
             value: float,
             bg: float,
             threshold: float
             ) -> Tuple[State, dict]:
    """
    15분 타임스텝 (리샘플링 후 투입).

    논문 case 2:
      1회 초과 → PRE_ALERT ("begin preparing")
      2회 연속 → ALERT     ("seek shelter ASAP")  ← 15분 x 2 = 30분
      threshold 아래 → NOMINAL ("safe to leave shelter")

    ALERT 중 background freeze:
      논문: "Background fitting is paused during the SEP event
             and resumes only after the SEP event has ended
             to avoid overestimating the background."
    """
    # ALERT 중에는 진입 시점의 bg를 그대로 사용 (freeze)
    if ctx.state == State.ALERT and ctx.bg_frozen is not None:
        bg        = ctx.bg_frozen
        threshold = bg * (1 + TRIGGER_PCT)

    above = value > threshold

    if above:
        ctx.consec_above += 1
    else:
        ctx.consec_above = 0

    prev        = ctx.state
    event_ended = False

    # ── 전이 로직 ──
    if ctx.state == State.NOMINAL:
        if above:
            ctx.state = State.PRE_ALERT
            ctx.start_event(t)

    elif ctx.state == State.PRE_ALERT:
        if ctx.consec_above >= CONSECUTIVE_FOR_ALERT:
            ctx.state     = State.ALERT
            ctx.bg_frozen = bg          # bg freeze 시작
        elif not above:
            # 1회 초과 후 내려옴 → false trigger → NOMINAL
            ctx.state = State.NOMINAL
            ctx.end_event()

    elif ctx.state == State.ALERT:
        if not above:
            ctx.state = State.NOMINAL
            ctx.end_event()
            event_ended = True

    # 누적 dose (bg 제거, 음수 clamp)
    if ctx.in_event:
        ctx.event_cumsum += max(value - bg, 0.0)
        ctx.event_peak    = max(ctx.event_peak, value)

    effective_threshold = (ctx.bg_frozen * (1 + TRIGGER_PCT)
                           if ctx.state == State.ALERT
                              and ctx.bg_frozen is not None
                           else threshold)

    return ctx.state, {
        "prev_state":     prev,
        "state":          ctx.state,
        "above":          above,
        "consec_above":   ctx.consec_above,
        "in_event":       ctx.in_event,
        "event_cumsum":   ctx.event_cumsum,
        "event_ended":    event_ended,
        "bg_used":        bg,
        "threshold_used": effective_threshold,
    }


def run_fsm(series: pd.Series,
            bg_series: pd.Series,
            thr_series: pd.Series) -> pd.DataFrame:
    """전체 시계열에 FSM 적용 (단일 날짜 모드용)."""
    ctx  = FSMContext()
    rows = []
    for t in series.index:
        v   = float(series[t])
        bg  = float(bg_series[t])
        thr = float(thr_series[t])
        state, info = fsm_step(ctx, t, v, bg, thr)
        rows.append({
            "Time":         t,
            "value":        v,
            "background":   info["bg_used"],
            "threshold":    info["threshold_used"],
            "above":        info["above"],
            "consec_above": info["consec_above"],
            "in_event":     info["in_event"],
            "event_cumsum": info["event_cumsum"],
            "state":        state,
            "state_val":    state.value,
        })
    return pd.DataFrame(rows).set_index("Time")


# ============================================================
# 4b.  RANGE 모드 통합 루프 (논문 compliance)
#
# [Fix 2] "updated daily": background를 하루 1회 갱신
# [Fix 3] 이벤트 중 freeze: background 계산과 FSM을 단일 루프로 통합
#          → 간이 FSM의 상태 불연속 문제 해결 (자정을 넘기는 이벤트 처리)
# [Fix 4] fit_background()와 동일한 constant mean 방식 사용
# [Fix E] 직전 5일 window가 비어있는 날(history 없음) → FSM skip
# ============================================================

def run_fsm_with_daily_bg(full: pd.Series,
                          pre_history: Optional[pd.Series] = None
                          ) -> pd.DataFrame:
    """
    background 계산과 FSM을 단일 루프로 통합한 range 모드 전용 함수.

    날짜 D의 처리 순서:
      1. D일 시작 시점에 bg 갱신 판단
         - FSMContext가 ALERT 상태 → bg freeze (이전 값 유지)
         - ALERT 아님 → mean(직전 5일) 으로 bg 갱신
           * 직전 5일은 full + pre_history 를 합쳐서 계산
         - [Fix E] 직전 5일 window가 비어있으면 해당 날 skip
      2. D일의 각 15분 포인트를 실제 FSMContext로 처리

    Parameters
    ----------
    full        : 분석 대상 전체 시계열 (15분 리샘플링 완료본)
    pre_history : full 시작 이전의 최대 5일치 데이터 (bg 초기화용, 15분 리샘플링 완료본)
                  None이면 bg window가 채워질 때까지 해당 날 skip

    Returns
    -------
    pd.DataFrame  (run_fsm과 동일한 컬럼 구조, history 충족 날짜만 포함)
    """
    ctx   = FSMContext()
    dates = sorted(set(full.index.date))
    rows  = []
    day_bg: Optional[float] = None   # None = 아직 유효한 bg 없음

    # pre_history와 full을 합쳐 bg 계산용 참조 시계열 구성
    if pre_history is not None and len(pre_history) > 0:
        ref = pd.concat([pre_history, full]).sort_index()
        ref = ref[~ref.index.duplicated(keep="first")]
    else:
        ref = full

    skipped_days = 0

    for d in dates:
        # ── 날짜 시작: bg 갱신 판단 ──
        if ctx.state != State.ALERT:
            win_end   = pd.Timestamp(d)
            win_start = win_end - pd.Timedelta(days=BACKGROUND_WINDOW_DAYS)
            hist      = ref[(ref.index >= win_start) & (ref.index < win_end)]

            if len(hist) > 0:
                y     = hist.values.astype(float)
                valid = np.isfinite(y) & (y > 0)
                if valid.sum() >= MIN_HISTORY_PTS:
                    # 5일치 history가 있어야 통과
                    day_bg = max(float(np.mean(y[valid])), 1.0)
                else:
                    # window 내 데이터가 있으나 유효값 없음 → skip
                    day_bg = None
            else:
                # [Fix E] window 내 데이터 없음 → bg 계산 불가 → skip
                day_bg = None

        # [Fix E] 유효한 bg가 없으면 이 날 전체를 skip (행 추가 안 함)
        if day_bg is None:
            skipped_days += 1
            continue

        # ── 15분 단위 FSM 처리 ──
        day_mask = full.index.date == d
        day_idx  = full.index[day_mask]
        for t in day_idx:
            val     = float(full[t])
            thr     = day_bg * (1 + TRIGGER_PCT)
            state, info = fsm_step(ctx, t, val, day_bg, thr)
            rows.append({
                "Time":         t,
                "value":        val,
                "background":   info["bg_used"],
                "threshold":    info["threshold_used"],
                "above":        info["above"],
                "consec_above": info["consec_above"],
                "in_event":     info["in_event"],
                "event_cumsum": info["event_cumsum"],
                "state":        state,
                "state_val":    state.value,
            })

    if skipped_days > 0:
        print(f"  [INFO] history 부족으로 skip된 날: {skipped_days}일")

    if not rows:
        raise RuntimeError("모든 날짜가 history 부족으로 skip됨. "
                           "pre_history 로드 범위를 확인하세요.")

    return pd.DataFrame(rows).set_index("Time")


# ============================================================
# 5.  VOTING
# ============================================================

def vote(pd_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    PRE_ALERT: 1 PD 이상 → 채택 (보수적 경고)
    ALERT:     1 PD 이상 → 채택 (OR 조건; FAR 억제는 연속 30분 조건이 담당)

    [Fix C] MIN_PD_VOTE_ALERT: 2 → 1
    """
    keys  = sorted(pd_results.keys())
    index = pd_results[keys[0]].index
    sv    = pd.DataFrame(
        {k: pd_results[k]["state_val"].reindex(index).values for k in keys},
        index=index)

    voted = []
    for _, row in sv.iterrows():
        vals        = list(row[keys])
        n_alert     = sum(v == State.ALERT.value     for v in vals)
        n_pre_alert = sum(v == State.PRE_ALERT.value for v in vals)

        if n_alert >= MIN_PD_VOTE_ALERT:
            voted.append(State.ALERT.value)
        elif (n_alert + n_pre_alert) >= MIN_PD_VOTE_PRE:
            voted.append(State.PRE_ALERT.value)
        else:
            voted.append(State.NOMINAL.value)

    sv["voted"]       = voted
    sv["voted_state"] = sv["voted"].map(lambda v: State(v))
    return sv


# ============================================================
# 6.  METRICS  (논문 4.1절)
# ============================================================

def extract_trigger_events(voted: pd.DataFrame
                           ) -> Tuple[List[pd.Timestamp],
                                      List[pd.Timestamp],
                                      List[dict]]:
    """
    트리거 시각과 이벤트 구간 추출.

    Returns
    -------
    pre_alert_times : PRE_ALERT onset 시각 리스트
    alert_times     : ALERT onset 시각 리스트
    detected_events : [{"pre_alert": t, "alert": t, "end": t}, ...]
    """
    pre_alert_times = []
    alert_times     = []
    detected_events = []

    prev      = State.NOMINAL
    cur_event: Optional[dict] = None

    for t, row in voted.iterrows():
        s = row["voted_state"]

        if prev == State.NOMINAL and s == State.PRE_ALERT:
            pre_alert_times.append(t)
            cur_event = {"pre_alert": t, "alert": None, "end": None}

        if prev != State.ALERT and s == State.ALERT:
            alert_times.append(t)
            if cur_event is not None:
                cur_event["alert"] = t

        if prev != State.NOMINAL and s == State.NOMINAL:
            if cur_event is not None:
                cur_event["end"] = t
                detected_events.append(cur_event)
                cur_event = None

        prev = s

    # 마지막 이벤트가 끝나지 않은 경우
    if cur_event is not None:
        cur_event["end"] = voted.index[-1]
        detected_events.append(cur_event)

    return pre_alert_times, alert_times, detected_events


def compute_metrics(voted: pd.DataFrame,
                    pd_results: Dict[str, pd.DataFrame]
                    ) -> dict:
    """
    알고리즘 출력 지표만 계산 (OBC 탑재용).

    포함:
      - duration_*   : 상태별 체류 시간 (알고리즘 자체 출력)
      - n_*_triggers : 트리거 횟수
      - PD1 기본 통계 (bg / threshold)

    제외 (→ evaluate.py):
      - FAR / Detection Rate  (ground truth 필요)
      - Lead Time             (ground truth 필요)

    제외 (→ benchmark.py):
      - 추론 시간 / RAM / Power
    """
    metrics = {}
    total   = len(voted)

    # 상태 체류 시간 (단위: 15분 포인트 수)
    for s in State:
        cnt = (voted["voted_state"] == s).sum()
        metrics[f"duration_{s.name}_pts"] = int(cnt)
        metrics[f"duration_{s.name}_min"] = int(cnt * RESAMPLE_MIN)
        metrics[f"duration_{s.name}_pct"] = round(cnt / total * 100, 2)

    pre_times, alert_times, _ = extract_trigger_events(voted)
    metrics["n_pre_alert_triggers"] = len(pre_times)
    metrics["n_alert_triggers"]     = len(alert_times)

    # PD 기본 통계
    for key in ["PD1", "PD2", "PD3"]:
        if key in pd_results:
            d = pd_results[key]
            metrics[f"{key}_value_mean"]     = round(float(d["value"].mean()),      2)
            metrics[f"{key}_value_max"]      = round(float(d["value"].max()),       2)
            metrics[f"{key}_bg_mean"]        = round(float(d["background"].mean()), 2)
            metrics[f"{key}_threshold_mean"] = round(float(d["threshold"].mean()),  2)

    return metrics


# ============================================================
# 7.  VISUALIZER  (3-panel)
# ============================================================

def plot_results(pd_results:   Dict[str, pd.DataFrame],
                 voted:        pd.DataFrame,
                 metrics:      dict,
                 pre_times:    List[pd.Timestamp],
                 alert_times:  List[pd.Timestamp],
                 bg_method:    str,
                 output_path:  str):

    pd_colors = {"PD1": "#3266ad", "PD2": "#d4653b", "PD3": "#2a9d6a"}
    fig = plt.figure(figsize=(16, 11))
    gs  = GridSpec(3, 1, figure=fig, hspace=0.38,
                   left=0.07, right=0.78)
    fig.suptitle(
        "KSEM PD  —  BL-C2  (Löwe et al. 2025 Reactive Nowcasting)\n"
        f"Background: {bg_method}  |  "
        f"Threshold: x{1 + TRIGGER_PCT:.2f}  |  "
        f"Cadence: {RESAMPLE_MIN} min  |  "
        f"Consecutive for ALERT: {CONSECUTIVE_FOR_ALERT} pts ({CONSECUTIVE_FOR_ALERT * RESAMPLE_MIN} min)",
        fontsize=11, y=0.98)

    # ── Panel 1: value + background + threshold ──
    ax1 = fig.add_subplot(gs[0])
    for key, df in pd_results.items():
        ax1.plot(df.index, df["value"],
                 color=pd_colors[key], alpha=0.2, linewidth=0.6)
        ax1.plot(df.index,
                 df["value"].rolling(5, min_periods=1).mean(),
                 color=pd_colors[key], linewidth=1.5,
                 label=f"{key} 5-pt avg")

    if "PD1" in pd_results:
        d = pd_results["PD1"]
        ax1.plot(d.index, d["background"],
                 color="gray", linestyle="--", linewidth=1.3,
                 label="Background (PD1)")
        ax1.plot(d.index, d["threshold"],
                 color="red",  linestyle="--", linewidth=1.3,
                 label=f"Threshold x{1+TRIGGER_PCT:.2f} (PD1)")

    ax1.set_yscale("log")
    ax1.set_ylabel("Proton count rate\n(A+B sum, O+OU+OUT logic)", fontsize=9)
    ax1.legend(fontsize=7, ncol=4, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: FSM 상태 ──
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    times = voted.index
    svals = voted["voted"].values

    for i in range(len(times) - 1):
        ax2.axvspan(times[i], times[i + 1],
                    alpha=0.28,
                    color=STATE_COLOR[State(svals[i])])

    offsets = {"PD1": 0.0, "PD2": 0.07, "PD3": -0.07}
    for key, df in pd_results.items():
        ax2.plot(df.index,
                 df["state_val"].values + offsets.get(key, 0),
                 color=pd_colors[key], linewidth=0.9,
                 alpha=0.8, label=key)

    ax2.step(voted.index, voted["voted"], color="black",
             linewidth=2.0, linestyle="--", where="post",
             label="Voted", zorder=5)

    for tt in pre_times:
        ax2.axvline(tt, color=STATE_COLOR[State.PRE_ALERT],
                    linewidth=1.5, alpha=0.9, zorder=4)
    for at in alert_times:
        ax2.axvline(at, color=STATE_COLOR[State.ALERT],
                    linewidth=1.5, alpha=0.9, linestyle=":", zorder=4)

    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(["NOMINAL", "PRE-ALERT", "ALERT"], fontsize=8)
    ax2.set_ylabel("FSM State", fontsize=9)
    ax2.grid(True, alpha=0.3, axis="x")

    # [Fix D] 레전드: state는 Patch, onset 마커는 Line2D로 구분
    legend_h = (
        [mpatches.Patch(color=STATE_COLOR[s], alpha=0.5, label=s.name)
         for s in State]
        + [Line2D([0], [0], color=STATE_COLOR[State.PRE_ALERT],
                  linewidth=1.5, label="PRE-ALERT onset"),
           Line2D([0], [0], color=STATE_COLOR[State.ALERT],
                  linewidth=1.5, linestyle=":", label="ALERT onset")]
    )
    ax2.legend(handles=legend_h, fontsize=7,
               loc="upper right", ncol=5)

    # ── Panel 3: value / background ratio ──
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    for key, df in pd_results.items():
        ratio = df["value"] / df["background"].replace(0, np.nan)
        ax3.plot(df.index, ratio,
                 color=pd_colors[key], linewidth=1.0,
                 alpha=0.85, label=key)

    ax3.axhline(1 + TRIGGER_PCT, color="red", linestyle="--",
                linewidth=1.2,
                label=f"Threshold ({1+TRIGGER_PCT:.2f}x)")
    ax3.axhline(1.0, color="gray", linewidth=0.8)
    ax3.set_ylabel("Value / Background", fontsize=9)
    ax3.set_xlabel("Time (UTC)", fontsize=9)
    upper = ax3.get_ylim()[1]
    ax3.set_ylim(0, min(upper, 20))
    ax3.legend(fontsize=7, loc="upper left", ncol=4)
    ax3.grid(True, alpha=0.3)

    # ── 오른쪽 metrics 박스 ──
    # FAR / Detection Rate / Lead Time → evaluate.py 참조
    lines = [
        "[ BL-C2 Metrics ]",
        "─" * 28,
        f"PRE-ALERT triggers : {metrics.get('n_pre_alert_triggers','—')}",
        f"ALERT triggers      : {metrics.get('n_alert_triggers','—')}",
        "─" * 28,
        "(FAR / Lead Time → evaluate.py)",
        "─" * 28,
    ]

    for s in State:
        lines.append(
            f"{s.name:10s}: "
            f"{metrics[f'duration_{s.name}_min']:6d} min "
            f"({metrics[f'duration_{s.name}_pct']:.1f}%)")

    pd_stat_lines = []
    for key in ["PD1", "PD2", "PD3"]:
        bg_k  = f"{key}_bg_mean"
        thr_k = f"{key}_threshold_mean"
        if bg_k in metrics:
            pd_stat_lines += [
                "─" * 28,
                f"{key} BG  avg : {metrics[bg_k]:.1f}",
                f"{key} Thr avg : {metrics[thr_k]:.1f}",
            ]
    lines += pd_stat_lines

    fig.text(0.80, 0.92, "\n".join(lines),
             fontsize=7.5, va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5",
                       facecolor="white",
                       edgecolor="gray", alpha=0.9))

    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    print(f"[Plot] 저장: {output_path}")


# ============================================================
# 8.  MAIN RUNNERS
# ============================================================

def _resample_15min(s: pd.Series) -> pd.Series:
    """1분 cadence Series를 15분 평균으로 리샘플링."""
    return s.resample(f"{RESAMPLE_MIN}min").mean().dropna()


def _process_pd_single(pd_key: str,
                       current: pd.Series,
                       history: Optional[pd.Series],
                       bg_method_out: list) -> Optional[pd.DataFrame]:
    """
    단일 날짜 모드: 하나의 PD 처리.

    [Fix E] history가 None이면 None 반환 → 호출부에서 skip.
    [Fix B] current / history 모두 15분 리샘플링 후 FSM 투입.
    """
    # [Fix B] 15분 리샘플링
    current_15 = _resample_15min(current)
    history_15 = _resample_15min(history) if history is not None else None

    bg_series, method = fit_background(
        history_15, len(current_15), current_15.index)
    bg_method_out.append(method)

    # [Fix E] history 없으면 skip
    if bg_series is None:
        print(f"  {pd_key}: history 없음 → skip (method={method})")
        return None

    thr_series = bg_series * (1 + TRIGGER_PCT)
    result     = run_fsm(current_15, bg_series, thr_series)
    print(f"  {pd_key}: method={method}  "
          f"bg={bg_series.mean():.1f}  "
          f"thr={thr_series.mean():.1f}  "
          f"ALERT+PRE={(result['state_val'] > 0).sum()}포인트"
          f"({(result['state_val'] > 0).sum() * RESAMPLE_MIN}분)")
    return result


def run_single_date(root: Path, date_str: str, output_dir: Path):
    """단일 날짜 분석 (테스트 / 검증용)."""
    print(f"\n[Mode] 단일 날짜: {date_str}")
    pd_results: Dict[str, pd.DataFrame] = {}
    bg_methods: list = []

    for key in ["PD1", "PD2", "PD3"]:
        try:
            current, history = load_single_date(root, date_str, key)
            print(f"  {key}: current={len(current)}분  "
                  f"history={len(history) if history is not None else 0}분")
            result = _process_pd_single(key, current, history, bg_methods)
            if result is not None:
                pd_results[key] = result
        except FileNotFoundError as e:
            print(f"  [SKIP] {key}: {e}")

    if not pd_results:
        raise RuntimeError(
            "분석할 PD 데이터가 없습니다. "
            "history(직전 5일)가 없는 날짜는 분석이 skip됩니다.")

    _finalize(pd_results, bg_methods, output_dir, tag=date_str)


def run_range(root: Path, start_ym: str, end_ym: str,
              output_dir: Path):
    """
    날짜 범위 전체 분석.

    분석 시작 전 5일치(pre_history)를 별도 로드하여
    첫날 background 초기화에 사용한다.

    [Fix B] full / pre_history 모두 15분 리샘플링 후 FSM 투입.
    [Fix E] pre_history 없는 초기 5일은 run_fsm_with_daily_bg 내에서 skip.
    """
    print(f"\n[Mode] 범위: {start_ym} ~ {end_ym}")
    pd_results: Dict[str, pd.DataFrame] = {}
    bg_methods: list = []

    # 분석 시작일 계산 (pre_history 로드 범위용)
    start_date     = pd.Timestamp(f"{start_ym[:4]}-{start_ym[4:]}-01")
    pre_hist_start = start_date - pd.Timedelta(days=BACKGROUND_WINDOW_DAYS)
    pre_hist_ym    = pre_hist_start.strftime("%Y%m")

    for key in ["PD1", "PD2", "PD3"]:
        try:
            # 분석 대상 전체 시계열 (1분 원본 로드)
            full_1min = load_series_range(root, start_ym, end_ym, key)
            print(f"  {key}: 총 {len(full_1min)}분 ({len(full_1min)/1440:.1f}일)")

            # [Fix B] 15분 리샘플링
            full = _resample_15min(full_1min)
            print(f"  {key}: 15분 리샘플 후 {len(full)}포인트 ({len(full)*15/1440:.1f}일)")

            # 분석 시작 이전 최대 5일치 pre_history 로드
            pre_history = None
            if pre_hist_ym < start_ym:
                try:
                    pre_full_1min = load_series_range(root, pre_hist_ym,
                                                      pre_hist_ym, key)
                    # full 시작 이전 데이터만 + 최대 5일치 (1분 단위로 슬라이싱 후 리샘플)
                    mask = pre_full_1min.index < full_1min.index[0]
                    if mask.sum() > 0:
                        pre_raw = pre_full_1min[mask].iloc[
                            -BACKGROUND_WINDOW_DAYS * 24 * 60:]
                        pre_history = _resample_15min(pre_raw)
                        print(f"  {key}: pre_history={len(pre_history)}포인트 (15분)")
                except FileNotFoundError:
                    print(f"  {key}: pre_history 없음 (이전 달 데이터 없음)")
            else:
                print(f"  {key}: pre_history 없음 (분석 시작월 = 최초 데이터월)")

            result = run_fsm_with_daily_bg(full, pre_history)

            pd_results[key] = result
            bg_methods.append("constant_mean_5day_daily_update_15min")
            print(f"  {key}: ALERT+PRE={(result['state_val']>0).sum()}포인트"
                  f"({(result['state_val']>0).sum() * RESAMPLE_MIN}분)")
        except FileNotFoundError as e:
            print(f"  [SKIP] {key}: {e}")

    if not pd_results:
        raise RuntimeError("분석할 PD 데이터가 없습니다.")

    _finalize(pd_results, bg_methods, output_dir,
              tag=f"{start_ym}_{end_ym}")


def _finalize(pd_results: Dict[str, pd.DataFrame],
              bg_methods: list,
              output_dir: Path,
              tag: str):
    """투표 → 지표 → 저장 공통 처리.

    known_events 기반 FAR / Lead Time 계산은 evaluate.py 로 분리됨.
    """

    voted = vote(pd_results)

    metrics = compute_metrics(voted, pd_results)

    pre_times, alert_times, _ = extract_trigger_events(voted)

    # 콘솔 출력
    print("\n[Metrics]")
    for k, v in metrics.items():
        print(f"  {k:40s}: {v}")

    print("\n[Trigger log]")
    for tt in pre_times:
        print(f"  PRE-ALERT @ {tt.strftime('%Y-%m-%d %H:%M')}")
    for at in alert_times:
        print(f"  ALERT     @ {at.strftime('%Y-%m-%d %H:%M')}")

    # CSV
    out_csv  = output_dir / f"bl_c2_states_{tag}.csv"
    save     = voted[["voted", "voted_state"]].copy()
    save.columns = ["state_val", "state_name"]
    save["state_name"] = save["state_name"].map(lambda s: s.name)
    for k, df in pd_results.items():
        save[f"{k}_value"]      = df["value"].reindex(save.index).values
        save[f"{k}_background"] = df["background"].reindex(save.index).round(3).values
        save[f"{k}_threshold"]  = df["threshold"].reindex(save.index).round(3).values
        save[f"{k}_state"]      = df["state"].reindex(save.index).map(lambda s: s.name if isinstance(s, State) else None).values
    save.to_csv(out_csv)
    print(f"\n[Output] CSV: {out_csv}")

    # Metrics JSON
    out_json = output_dir / f"bl_c2_metrics_{tag}.json"
    clean    = {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                for k, v in metrics.items()}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
    print(f"[Output] Metrics: {out_json}")

    # Plot
    bg_method_str = bg_methods[0] if bg_methods else "unknown"
    out_plot      = output_dir / f"bl_c2_plot_{tag}.png"
    plot_results(pd_results, voted, metrics,
                 pre_times, alert_times,
                 bg_method_str, str(out_plot))

    print("\n완료.")


# ============================================================
# 9.  ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="KSEM BL-C2 — Löwe et al. 2025 Reactive Nowcasting")
    parser.add_argument("--root",       default="D:\\Raw Count",
                        help="Raw Count 루트 폴더 (기본: D:\\Raw Count)")
    parser.add_argument("--date",       default=None,
                        help="단일 날짜 분석: YYYYMMDD (예: 20240510)")
    parser.add_argument("--start",      default=None,
                        help="범위 시작 월: YYYYMM (예: 201905)")
    parser.add_argument("--end",        default=None,
                        help="범위 종료 월: YYYYMM (예: 202412)")
    parser.add_argument("--output_dir", default="./output_bl_c2",
                        help="결과 저장 폴더")
    args = parser.parse_args()

    root       = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.date:
        run_single_date(root, args.date, output_dir)
    elif args.start and args.end:
        run_range(root, args.start, args.end, output_dir)
    else:
        parser.error("--date 또는 --start/--end 중 하나를 지정하세요.")


if __name__ == "__main__":
    main()