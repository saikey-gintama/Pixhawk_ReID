"""
ksem_common.py
==============
ana1~ana5 공유 로직 모듈.

설계 원칙:
  - config 상수(경로·채널·threshold): ksem_flux_config.py
  - 분석별 튜닝값(window 크기 등): 각 ana 파일 상단
  - 데이터 로드·proxy 생성·이벤트 탐지·배경 추정: 이 파일

중복 제거 현황:
  SPEEvent        ana2/ana4/ana5 각자 정의 → 여기 하나로
  make_proxy_series  ana2(proton/electron 분리) / ana3/ana4/ana5 각자 정의 → 통합
  detect_events   ana2/ana4 각자 정의, ana5는 detect_proton_spe로 분리 → 통합
  get_bg_median   ana2 단독 → 여기로 이동 (ana3 quiet 샘플 추출에도 사용)
  print_proxy_diag  ana2/ana4 각자 정의 → 통합
  load_data       ana2~ana5 main() 공통 패턴 → 함수로 추출
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from ksem_flux_config import (
    COUNT_PARQUET_DIR, FLUX_PARQUET_DIR,
    PROTON_CHANNELS, ELECTRON_CHANNELS,
    SPE_PROXY_CHANNELS, SPE_PROXY_THRESH,
    E_SPE_PROXY_CHANNELS, E_SPE_PROXY_THRESH,
    MIN_SPE_DURATION_H, MIN_E_SPE_DURATION_H,
)
import ksem_io
import kma_ksem_flux_io


# ── 이벤트 데이터클래스 ───────────────────────────────────────────
class SPEEvent(NamedTuple):
    onset:     pd.Timestamp
    peak:      pd.Timestamp
    end:       pd.Timestamp
    peak_flux: float   # [cm-2 sr-1 s-1]


# ── 데이터 로드 ───────────────────────────────────────────────────
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    count / proton flux / electron flux 로드 후 UTC 통일.
    반환: (df_count, df_flux_p, df_flux_e)
    """
    df_count, _    = ksem_io.load(COUNT_PARQUET_DIR)
    sensor_data, _ = kma_ksem_flux_io.load(FLUX_PARQUET_DIR)
    df_flux_p      = sensor_data.get("proton",   pd.DataFrame())
    df_flux_e      = sensor_data.get("electron", pd.DataFrame())

    for df in [df_count, df_flux_p, df_flux_e]:
        if not df.empty and df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

    return df_count, df_flux_p, df_flux_e


# ── Proxy 생성 ────────────────────────────────────────────────────
def make_proxy_series(df_flux: pd.DataFrame,
                      channels: list[str],
                      ch_ranges: dict[str, tuple[float, float]]) -> pd.Series:
    """
    ΔE-적분 합산 proxy.
    channels / ch_ranges로 proton·electron 공용.
    반환 단위: cm-2 sr-1 s-1
    """
    proxy = pd.Series(0.0, index=df_flux.index)
    for ch in channels:
        if ch not in df_flux.columns:
            continue
        lo, hi = ch_ranges[ch]
        proxy += df_flux[ch].fillna(0) * (hi - lo)
    proxy[proxy == 0] = np.nan
    return proxy


def make_proton_proxy(df_flux_p: pd.DataFrame) -> pd.Series:
    """E8+E9+E10 ΔE-적분 합산. 단위: cm-2 sr-1 s-1"""
    return make_proxy_series(df_flux_p, SPE_PROXY_CHANNELS,
                             PROTON_CHANNELS).rename("proxy_p")


def make_electron_proxy(df_flux_e: pd.DataFrame) -> pd.Series:
    """E4+E5+E6 ΔE-적분 합산. 단위: cm-2 sr-1 s-1"""
    return make_proxy_series(df_flux_e, E_SPE_PROXY_CHANNELS,
                             ELECTRON_CHANNELS).rename("proxy_e")


# ── 이벤트 탐지 ───────────────────────────────────────────────────
def detect_events(flux: pd.Series,
                  thresh: float,
                  min_duration_h: float,
                  label: str = "SPE") -> list[SPEEvent]:
    """
    flux >= thresh 가 min_duration_h 이상 연속인 구간을 SPEEvent로 반환.
    proton(label="SPE") / electron(label="E_SPE") 공용.
    """
    above  = flux >= thresh
    events: list[SPEEvent] = []
    in_ev  = False
    onset  = None

    for t, v in above.items():
        if v and not in_ev:
            in_ev, onset = True, t
        elif not v and in_ev:
            seg   = flux.loc[onset:t]
            dur_h = (t - onset).total_seconds() / 3600
            if dur_h >= min_duration_h:
                events.append(SPEEvent(onset, seg.idxmax(), t, float(seg.max())))
            in_ev = False

    if in_ev and onset is not None:
        seg = flux.loc[onset:]
        if (seg.index[-1] - onset).total_seconds() / 3600 >= min_duration_h:
            events.append(SPEEvent(onset, seg.idxmax(),
                                   seg.index[-1], float(seg.max())))

    print(f"[ksem_common] {label} events detected: {len(events)}  "
          f"(flux >= {thresh:.1e}, min {min_duration_h}h)")
    return events


def detect_proton_events(proxy_p: pd.Series) -> list[SPEEvent]:
    """설정값 기반 proton SPE 탐지. config 변경 시 전 파일에 자동 반영."""
    return detect_events(proxy_p, SPE_PROXY_THRESH, MIN_SPE_DURATION_H, "SPE")


def detect_electron_events(proxy_e: pd.Series) -> list[SPEEvent]:
    """설정값 기반 electron E_SPE 탐지."""
    if E_SPE_PROXY_THRESH is None:
        print("[ksem_common] E_SPE_PROXY_THRESH=None → electron 이벤트 탐지 스킵")
        return []
    return detect_events(proxy_e, E_SPE_PROXY_THRESH,
                         MIN_E_SPE_DURATION_H, "E_SPE")


# ── 배경 추정 (Löwe et al. 2025) ──────────────────────────────────
def _select_quiet_samples(cnt: pd.Series,
                          onset: pd.Timestamp,
                          bg_window_days: int,
                          bg_quiet_days: int) -> pd.Series:
    """
    onset 이전 bg_window_days 중 "가장 조용한 bg_quiet_days일"의 원본 샘플 반환.
    get_bg_median / get_bg_threshold / get_quiet_bg_samples가 공유하는 단일 진입점.
    → 세 함수가 항상 동일한 날을 보게 보장한다.

    "가장 조용한 N일" 정의:
      1. onset 이전 bg_window_days 슬라이딩 윈도우 내 원본 데이터 추출
      2. 하루 단위 median 계산 → 일별 대표값
      3. 그 중 값이 가장 낮은 bg_quiet_days일 선택
      4. 선택된 날의 원본 데이터 반환

    이 방식을 쓰는 이유:
      - 분 단위 하위 N%는 데이터 갭·0값 노이즈가 섞임
      - 하루 단위 집계 후 선택하면 단발 스파이크 영향 최소화
      - 이벤트가 연속으로 와도 조용한 날이 bg_quiet_days개 이상이면
        배경 추정이 오염되지 않음

    데이터 부족 시 빈 Series 반환(len==0).
    """
    seg = cnt.loc[onset - pd.Timedelta(days=bg_window_days):onset].dropna()
    if len(seg) < 10:
        return pd.Series(dtype=float)

    daily = seg.resample("1D").median().dropna()
    if len(daily) < bg_quiet_days:
        return pd.Series(dtype=float)

    quiet_days = daily.nsmallest(bg_quiet_days)
    quiet_mask = seg.index.normalize().isin(quiet_days.index.normalize())
    return seg[quiet_mask]


def get_bg_median(cnt: pd.Series,
                  onset: pd.Timestamp,
                  bg_window_days: int,
                  bg_quiet_days: int) -> float:
    """가장 조용한 bg_quiet_days일의 count median. 진단·배율 계산용."""
    q = _select_quiet_samples(cnt, onset, bg_window_days, bg_quiet_days)
    return float(q.median()) if len(q) else np.nan


def get_bg_threshold(cnt: pd.Series,
                     onset: pd.Timestamp,
                     bg_window_days: int,
                     bg_quiet_days: int,
                     k: float) -> float:
    """
    Count onset 임계값 = bg_median + k · σ_quiet.

    onset 이전 가장 조용한 bg_quiet_days일의 원본 count로 배경 통계를 추정하고,
    median에 실측 표준편차(ddof=1)의 k배를 더한 값을 임계값으로 반환한다.

    산포항으로 √median(Poisson)이 아니라 실측 σ를 쓰는 이유:
      KSEM count는 강한 과분산(ana3 Index of Dispersion ≫ 1, OU/OUT은 7만~21만).
      Poisson 가정은 분산을 median으로 과소평가해 임계값이 배경 수준까지 내려가고,
      이전 p70 방식이 OU/OUT(quiet median<1)에서 단일 카운트를 onset으로
      오검출한 것과 같은 문제를 일으킨다. 실측 σ는 채널 고유의 노이즈 폭을
      그대로 반영한다.

    주의:
      배경 자체가 0에 가까운 저카운트 채널(OU/OUT 등)은 bg+kσ도 매우 작은
      정수가 될 수 있다. floor(절대 최소 임계값) 또는 트리거 채널 제외 여부는
      이 함수 결과를 본 뒤 별도로 결정한다(현재는 bg+kσ만 적용).

    데이터 부족 시 np.nan.
    """
    q = _select_quiet_samples(cnt, onset, bg_window_days, bg_quiet_days)
    if len(q) < 2:
        return np.nan
    return float(q.median() + k * q.std(ddof=1))


def compute_rolling_bg(cnt: pd.Series,
                       bg_window_days: int,
                       bg_quiet_days: int,
                       update_freq: str = "1D") -> pd.DataFrame:
    """
    전 기간 rolling 배경 시계열. flux/이벤트 ground truth 없이,
    "매 시점 직전 bg_window_days 중 조용한 bg_quiet_days일" 배경을 추정한다.
    onboard FSM이 실제로 도는 방식(미래를 모르고 과거 N일만 봄)과 동일.

    비용 관리:
      배경은 천천히 변하므로 매 샘플이 아니라 update_freq(기본 1일)마다 한 번만
      갱신하고, 그 사이는 직전 갱신값을 forward-fill한다. 1분 cadence 290만
      포인트를 일별 ~수천 회 계산으로 줄인다. get_bg_median/get_bg_threshold와
      동일한 _select_quiet_samples를 재사용 → 단일 시점 함수와 정의가 일치.

    반환: DataFrame(index=cnt.index, columns=["bg_median", "bg_std"])
      갱신 불가(데이터 부족) 구간은 NaN → 호출부에서 검출 제외.
    """
    if cnt.empty:
        return pd.DataFrame(columns=["bg_median", "bg_std"])

    update_points = pd.date_range(cnt.index[0], cnt.index[-1], freq=update_freq)
    rows = []
    for t in update_points:
        q = _select_quiet_samples(cnt, t, bg_window_days, bg_quiet_days)
        if len(q) >= 2:
            rows.append((t, float(q.median()), float(q.std(ddof=1))))
        elif len(q) == 1:
            rows.append((t, float(q.median()), np.nan))
        else:
            rows.append((t, np.nan, np.nan))

    bg = pd.DataFrame(rows, columns=["t", "bg_median", "bg_std"]).set_index("t")
    # cnt 전 시점으로 확장 후 직전 갱신값 유지
    bg = bg.reindex(cnt.index.union(bg.index)).ffill().reindex(cnt.index)
    return bg


def get_quiet_bg_samples(cnt: pd.Series,
                         events: list[SPEEvent],
                         bg_window_days: int,
                         bg_quiet_days: int) -> pd.Series:
    """
    ana3 Poisson QQ 전용.
    모든 이벤트에서 배경 추정이 실제로 선택한 날짜의 데이터를 모아 반환.
    get_bg_median / get_bg_threshold와 동일한 _select_quiet_samples를 쓰므로
    "배경 추정에 쓰인 데이터가 실제로 조용한가"를 정확히 같은 표본으로 검증한다.
    """
    all_quiet: list[pd.Series] = []
    for ev in events:
        q = _select_quiet_samples(cnt, ev.onset, bg_window_days, bg_quiet_days)
        if len(q):
            all_quiet.append(q)

    if not all_quiet:
        return pd.Series(dtype=float)
    return pd.concat(all_quiet).drop_duplicates()


# ── 진단 출력 ─────────────────────────────────────────────────────
def print_proxy_diag(proxy: pd.Series,
                     thresh: float | None,
                     label: str) -> None:
    """proxy 분포 percentile 출력. threshold 설정 전 참고용."""
    pv = proxy.dropna()
    print(f"  [diag] {label} flux percentiles (cm-2 sr-1 s-1):")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"         p{p:2d} = {np.nanpercentile(pv, p):.3e}")
    thresh_str = f"{thresh:.1e}" if thresh is not None else "None (설정 필요)"
    print(f"  [diag] current threshold = {thresh_str}")
    if thresh is not None:
        print(f"  [diag] fraction above thresh = {(pv >= thresh).mean():.3f}")