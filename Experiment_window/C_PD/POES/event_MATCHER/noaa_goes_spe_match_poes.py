"""
noaa_goes_spe_match_poes.py
===========================
POES FSM 검출 이벤트 ↔ NOAA SPE 카탈로그 매칭/평가 (POES 판).

공통 로직은 _match_core_poes.py 에서 import.
이 파일은 CATALOG_LABEL / DEFAULT_IO / OUT_PREFIX 만 지정하고
run_matcher() 를 호출한다.

출력 (outdir = <--out>/<runtag>/):
  noaa_match_summary_all_<runtag>.csv   전 조합 POD/FAR
  noaa_match_<runtag>.csv               FINAL 조합 채널별, POD 내림차순
  fig_noaa_scatter_<runtag>.png         POD-FAR 산점도 (species 색상)

사용 (POES/event_MATCHER/ 에서):
  python noaa_goes_spe_match_poes.py \\
    --events ../MetOp03_count/fsm2_output/<tag>/fsm_onset_<tag>.csv

  # catalog 는 ../../NOAA_GOES/noaa_goes_spe_cache_parquet 으로 자동 설정됨
"""
from pathlib import Path
from _match_core_poes import run_matcher

_HERE = Path(__file__).resolve().parent          # POES/event_MATCHER/

if __name__ == "__main__":
    run_matcher(
        catalog_label   = "NOAA SPE",
        default_io      = "noaa_goes_spe_io",
        out_prefix      = "noaa",
        default_catalog = str(_HERE.parent.parent / "NOAA_GOES" / "noaa_goes_spe_cache_parquet"),
    )
