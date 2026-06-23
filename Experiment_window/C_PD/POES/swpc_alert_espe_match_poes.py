"""
swpc_alert_espe_match_poes.py
=============================
POES FSM 검출 이벤트 ↔ SWPC ESPE 카탈로그 매칭/평가 (POES 판).

공통 로직은 _match_core_poes.py 에서 import.
이 파일은 CATALOG_LABEL / DEFAULT_IO / OUT_PREFIX 만 지정하고
run_matcher() 를 호출한다.

출력 (outdir = <--out>/<runtag>/):
  swpc_match_summary_all_<runtag>.csv   전 조합 POD/FAR
  swpc_match_<runtag>.csv               FINAL 조합 채널별, POD 내림차순
  fig_swpc_scatter_<runtag>.png         POD-FAR 산점도 (species 색상)

사용 (POES 루트에서):
  python swpc_alert_espe_match_poes.py \\
    --events MetOp03_count/fsm2_output/<tag>/fsm_onset_<tag>.csv \\
    --catalog ../SWPC_Alert/espe_cache_parquet \\
    --spe-io ../SWPC_Alert/swpc_alert_espe_io \\
    --out MetOp03_count/swpc_match_output
"""
from _match_core_poes import run_matcher

if __name__ == "__main__":
    run_matcher(
        catalog_label = "SWPC ESPE",
        default_io    = "swpc_alert_espe_io",
        out_prefix    = "swpc",
    )
