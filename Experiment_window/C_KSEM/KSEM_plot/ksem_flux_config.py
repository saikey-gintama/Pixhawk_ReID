"""
ksem_flux_config.py
====================
모든 ana 스크립트가 공유하는 설정값 전용 모듈.
스크립트 단독으로 사용하는 튜닝값은 각 ana 파일 상단에 정의한다.

KSEM 채널 컬럼명 (실측 확인):
  sensor_data['proton']   → E1~E10
  sensor_data['electron'] → E1~E10

메타(_kma_ksem_flux_meta.json) 기준 에너지 범위:
  proton   E1=77-119 keV   E2=119-148   E3=148-229   E4=229-354
           E5=354-548      E6=548-681   E7=681-1052
           E8=1052-2021    E9=2021-3123 E10=3123-6000
  electron E1=100-150 keV  E2=150-225   E3=225-325   E4=325-450
           E5=450-700      E6=700-1350  E7=1350-1800
           E8=1800-2600    E9=2600-3800 E10=2000-3800

proton SPE proxy:
  NOAA 10 MeV 기준 채널 없음 → E8+E9+E10 ΔE-적분 합산으로 대체.
  SPE_PROXY_THRESH는 데이터 분포 p95(~1.2e+04) 기준.
  변경 시 ana2·ana3·ana4·ana5 전체 재실행 필요.

electron E_SPE proxy:
  Torres et al. (2025) >0.25~0.67 MeV 대응 → E4+E5+E6 ΔE-적분 합산.
  E_SPE_PROXY_THRESH는 첫 실행 후 ana2 proxy_e 분포 p95 보고 설정.
  변경 시 ana2·ana3·ana5 재실행 필요.
"""

import sys
import matplotlib
from pathlib import Path

# ── 경로 ──────────────────────────────────────────────────────────
_PLOT_DIR = Path(__file__).parent.resolve()
_ROOT_DIR = _PLOT_DIR.parent

for _d in [_ROOT_DIR / "KSEM_count", _ROOT_DIR / "KMA_KSEM_flux"]:
    _s = str(_d)
    if _s not in sys.path:
        sys.path.insert(0, _s)

COUNT_PARQUET_DIR = _ROOT_DIR / "KSEM_count"    / "ksem_cache_parquet"
FLUX_PARQUET_DIR  = _ROOT_DIR / "KMA_KSEM_flux" / "kma_ksem_flux_cache_parquet"
OUTPUT_DIR        = _PLOT_DIR / "ana_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Proton flux 채널 에너지 범위 [keV] ───────────────────────────
PROTON_CHANNELS = {
    "E1":  (77,   119),
    "E2":  (119,  148),
    "E3":  (148,  229),
    "E4":  (229,  354),
    "E5":  (354,  548),
    "E6":  (548,  681),
    "E7":  (681,  1052),
    "E8":  (1052, 2021),
    "E9":  (2021, 3123),
    "E10": (3123, 6000),
}

# ── Electron flux 채널 에너지 범위 [keV] ─────────────────────────
ELECTRON_CHANNELS = {
    "E1":  (100,  150),
    "E2":  (150,  225),
    "E3":  (225,  325),
    "E4":  (325,  450),
    "E5":  (450,  700),
    "E6":  (700,  1350),
    "E7":  (1350, 1800),
    "E8":  (1800, 2600),
    "E9":  (2600, 3800),
    "E10": (2000, 3800),
}

# ── Proton SPE proxy 설정 ─────────────────────────────────────────
SPE_PROXY_CHANNELS = ["E8", "E9", "E10"]
SPE_PROXY_LABEL    = "E8+E9+E10 (1-6 MeV dE integral, proton)"
SPE_PROXY_THRESH   = 1.2e+04   # [cm-2 sr-1 s-1] 첫 실행 후 ana2 proxy_p 분포 p95 보고 설정

# ── Electron E_SPE proxy 설정 ─────────────────────────────────────
# Torres et al. (2025) >0.25~0.67 MeV 채널 대응: E4(325-450)+E5(450-700)+E6(700-1350)
E_SPE_PROXY_CHANNELS = ["E4", "E5", "E6"]
E_SPE_PROXY_LABEL    = "E4+E5+E6 (325-1350 keV dE integral, electron)"
E_SPE_PROXY_THRESH   = 5.2e+05   # [cm-2 sr-1 s-1] 첫 실행 후 ana2 proxy_e 분포 p95 보고 설정

# ── Count 분석 대상 차원 ──────────────────────────────────────────
COUNT_PROTON_LOGICS   = ["O", "OU", "OUT"]
COUNT_ELECTRON_LOGICS = ["F", "FT", "FTU", "FTUO"]
COUNT_PD_KEYS         = ["PD1", "PD2", "PD3"]
COUNT_SIDES           = ["A", "B"]

# ── 이벤트 탐지 공통 조건 ─────────────────────────────────────────
# proxy >= THRESH 가 이 시간 이상 연속되어야 이벤트로 인정
MIN_SPE_DURATION_H    = 1    # [h] proton SPE (ana2·ana4 공유)
MIN_E_SPE_DURATION_H  = 1    # [h] electron E_SPE (ana2·ana4·ana5 공유), 초기값 동일

# ── 배경(quiet) 추정 공통 파라미터 (Löwe et al. 2025) ─────────────
# onset 이전 BG_WINDOW_DAYS 슬라이딩 윈도우 중 가장 조용한 BG_QUIET_DAYS일을
# 골라 그 구간 count로 배경 통계를 추정한다. ana2/ana3/ana4 전부 공유.
# (이전에는 ana2·ana3에 각각 하드코딩되어 있었음 → 여기로 승격)
BG_WINDOW_DAYS = 30   # [day] onset 이전 배경용 슬라이딩 윈도우
BG_QUIET_DAYS  = 7    # [day] 윈도우 중 가장 조용한 N일만 사용

# ── Count onset 임계값: 배경 기반 (bg_median + k·σ_quiet) ─────────
# 산포항은 실측 quiet σ를 쓴다. KSEM count는 강한 과분산(ana3 ID≫1,
# OU/OUT은 7만~21만)이라 √median(Poisson) 가정은 임계값을 과소평가한다.
# 채널별·이벤트별로 get_bg_threshold()가 동적으로 계산한다.
COUNT_BG_K = 5   # 배경 σ 배수 (ana3 SPIKE_MULTIPLIER=5와 정합)

# ── [DEPRECATED] Count onset 판정 퍼센타일 ────────────────────────
# 더 이상 onset 임계값으로 쓰지 않는다. quiet 분포의 위치 퍼센타일은
# 과분산 채널마다 의미가 달라지고, OU/OUT처럼 quiet median<1인 채널은
# p70=0~1이 되어 단일 카운트가 onset으로 오검출되었다.
# → 배경 기반 임계값(COUNT_BG_K)으로 대체. ana2 진단 출력에만 참고로 남김.
SPE_COUNT_THRESH_PCTL   = 70   # [%] (deprecated, 진단용)
E_SPE_COUNT_THRESH_PCTL = 70   # [%] (deprecated, 진단용)


matplotlib.rcParams["axes.unicode_minus"] = False