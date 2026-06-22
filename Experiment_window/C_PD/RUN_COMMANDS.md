# blc1_fixed 실행 명령 (GK2A · POES × MetOp03 · NOAA19 × const · pctl)

## 공통 사항
- GK2A 스크립트는 `count_FSM/` 에서 실행.
- POES 스크립트는 `POES/` 에서 실행.
- `--mode const` 는 `--const-csv` 경로 필수.
- 출력 위치: `fsm2_output/<runtag>/fsm_onset_*.csv`, `fsm_event_*.csv`
- 전체 실행 전 1채널 테스트 권장 (`--mode const` 는 프리뷰 출력으로 검증 먼저).

---

## GK2A (KSEM count)

```powershell
# ──────────────────────────────────────────
# 실행 위치: Experiment_window/C_PD/GK2A/count_FSM/
# ──────────────────────────────────────────

# [GK2A pctl95] 분위수 고정임계, 전 채널
python fsm_count_spe_blc1_fixed.py `
    --mode pctl --pctl 95 `
    --onset 0.5 --peak 2.0

# [GK2A const · NOAA SPE 기준] per-channel C_fpr0.05, proton/electron 혼합
python fsm_count_spe_blc1_fixed.py `
    --mode const `
    --const-csv ../KSEM_count/ana_output/noaa_spe_event_count_stats.csv `
    --onset 0.5 --peak 2.0

# [GK2A const · SWPC ESPE 기준] 전자 채널 C_fpr0.05 우선시
python fsm_count_spe_blc1_fixed.py `
    --mode const `
    --const-csv ../KSEM_count/ana_output/swpc_espe_event_count_stats.csv `
    --onset 0.5 --peak 2.0
```

---

## POES MetOp03

```powershell
# ──────────────────────────────────────────
# 실행 위치: Experiment_window/C_PD/POES/
# ──────────────────────────────────────────

# [MetOp03 pctl95]
python fsm_count_spe_blc1_fixed_poes.py `
    --io poes_metop03_io `
    --cache MetOp03_count/poes_metop03_cache_parquet `
    --mode pctl --pctl 95 `
    --onset 0.5 --peak 2.0

# [MetOp03 const · NOAA SPE 기준] proton-facing C_fpr0.05
python fsm_count_spe_blc1_fixed_poes.py `
    --io poes_metop03_io `
    --cache MetOp03_count/poes_metop03_cache_parquet `
    --mode const `
    --const-csv MetOp03_count/ana_event_output/noaa_spe_event_count_stats.csv `
    --onset 0.5 --peak 2.0

# [MetOp03 const · SWPC ESPE 기준] electron-facing C_fpr0.05
python fsm_count_spe_blc1_fixed_poes.py `
    --io poes_metop03_io `
    --cache MetOp03_count/poes_metop03_cache_parquet `
    --mode const `
    --const-csv MetOp03_count/ana_event_output/swpc_espe_event_count_stats.csv `
    --onset 0.5 --peak 2.0

# [MetOp03 geo 태깅 생략] geo 캐시 없을 때
python fsm_count_spe_blc1_fixed_poes.py `
    --io poes_metop03_io `
    --cache MetOp03_count/poes_metop03_cache_parquet `
    --mode pctl --pctl 95 --no-geo
```

---

## POES NOAA19

```powershell
# ──────────────────────────────────────────
# 실행 위치: Experiment_window/C_PD/POES/
# ──────────────────────────────────────────

# [NOAA19 pctl95]
python fsm_count_spe_blc1_fixed_poes.py `
    --io poes_noaa19_io `
    --cache NOAA19_count/poes_noaa19_cache_parquet `
    --mode pctl --pctl 95 `
    --onset 0.5 --peak 2.0

# [NOAA19 const · NOAA SPE 기준]
python fsm_count_spe_blc1_fixed_poes.py `
    --io poes_noaa19_io `
    --cache NOAA19_count/poes_noaa19_cache_parquet `
    --mode const `
    --const-csv NOAA19_count/ana_event_output/noaa_spe_event_count_stats.csv `
    --onset 0.5 --peak 2.0

# [NOAA19 const · SWPC ESPE 기준]
python fsm_count_spe_blc1_fixed_poes.py `
    --io poes_noaa19_io `
    --cache NOAA19_count/poes_noaa19_cache_parquet `
    --mode const `
    --const-csv NOAA19_count/ana_event_output/swpc_espe_event_count_stats.csv `
    --onset 0.5 --peak 2.0
```

---

## Löwe baseline (blc1_lowe · POES) — 참고

```powershell
# 실행 위치: Experiment_window/C_PD/POES/

# [MetOp03 Löwe 기본 파라미터]
python fsm_count_spe_blc1_lowe_poes.py `
    --io poes_metop03_io `
    --cache MetOp03_count/poes_metop03_cache_parquet

# [NOAA19 Löwe]
python fsm_count_spe_blc1_lowe_poes.py `
    --io poes_noaa19_io `
    --cache NOAA19_count/poes_noaa19_cache_parquet
```
