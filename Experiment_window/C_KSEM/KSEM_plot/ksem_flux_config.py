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

NOAA SPE 기준(>10 MeV proton) 채널 없음.
→ E8+E9+E10 ΔE-적분 합산을 SPE proxy로 사용.
  proxy 단위: cm-2 sr-1 s-1 (flux × ΔE keV 적산)
  SPE_PROXY_THRESH는 데이터 분포의 p95(~1.2e+04)를 기준으로 설정.
  변경 시 ana2·ana3·ana4 전체 재실행 필요.
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

# ── Proton 채널 에너지 범위 [keV] ────────────────────────────────
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

# ── Electron 채널 에너지 범위 [keV] ──────────────────────────────
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

# ── SPE proxy 설정 ────────────────────────────────────────────────
# NOAA 10 MeV 기준 채널 없음 → E8+E9+E10 ΔE-적분 합산으로 대체
# 변경 시 ana2·ana3·ana4 전체 재실행 필요
SPE_PROXY_CHANNELS = ["E8", "E9", "E10"]
SPE_PROXY_LABEL    = "E8+E9+E10 (1-6 MeV dE integral, proton)"
SPE_PROXY_THRESH   = 1.2e+04   # [cm-2 sr-1 s-1] proxy 분포 p95 기준
SPE_SINGLE_CHANNEL = "E10"     # ana1 단독 채널 플롯용

# ── 배경 추세 분석용 채널 그룹 (ana3 공유) ───────────────────────
ELECTRON_BG_CHANNELS = ["E1", "E2", "E3", "E4", "E5", "E6"]
PROTON_BG_CHANNELS   = ["E8", "E9", "E10"]

# ── Count 분석 대상 차원 ──────────────────────────────────────────
COUNT_PROTON_LOGICS = ["O", "OU", "OUT"]   # 검출기 로직
COUNT_PD_KEYS       = ["PD1", "PD2", "PD3"]
COUNT_SIDES         = ["A", "B"]

# ── SPE 이벤트 탐지 공통 조건 (ana2·ana4 공유) ───────────────────
# proxy >= SPE_PROXY_THRESH 가 이 시간 이상 연속되어야 이벤트로 인정
MIN_SPE_DURATION_H    = 6    # [h]
SPE_COUNT_THRESH_PCTL = 75   # [%] 현재값, 데이터 보고 조정


matplotlib.rcParams["axes.unicode_minus"] = False
