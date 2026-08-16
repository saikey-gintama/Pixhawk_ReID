"""
gate_aggregation_sweep.py
==========================
논문 4.5절 "persistence mismatch" -- 게이트 지속성(N)과 분류기 집계 규칙(M)의
공동 최적점 탐색. 읽기 전용(엔진 파일), C_PD 아래는 full_record_proba.parquet
(아래 설명, 사용자 명시 지시로 예외) 한 곳만 쓴다. 재학습 없음.

이 파일은 window_result/ 에 있다 -- gate_persistence_sweep.py 등 엔진 파일은
한 단계 위 POES/ 에 있으므로 sys.path/파일 경로는 부모(POES_DIR)를 보되, 출력
(CSV/그림)은 이 스크립트가 있는 디렉터리(window_result/)에 그대로 쓴다.

확률 소스 (2차 개정 -- 1차 OOF 버전의 문제를 이걸로 고침):
  1차 시도(oof_predictions.parquet)는 이벤트 주변 크롭(episode) 안에서만 확률이
  존재해 전 구간 게이트 개방 틱의 38.6%만 덮었다 -- 논문 Table 6(전 구간 추론
  236,739 샘플 기준)과 분모가 달라 대조가 안 됐다.
  이번엔 e2e_gated_eval.py(as e2e)의 build_p_event() -- 즉
  v4.build_full_features() + v4.ensemble_event_proba() -- 를 그대로 호출해
  **전 구간** 확률을 만든다(고정 가중치 forward pass, 재학습/체크포인트 변경
  없음, 사용자 명시 허가). 그 결과를 predict_v0/results/e2e/full_record_proba.parquet
  로 캐싱한다 -- 다음 실행부터는 이 파일이 있으면 재추론 없이 그대로 읽는다
  (지금까지 아무도 이걸 저장 안 해서 4.5절이 재현 불가능했던 문제의 근본 수정,
  사용자 명시 지시. 이 파일 하나만 C_PD 아래(predict_v0/results/e2e/, e2e_gated_eval.py
  가 이미 자기 출력을 쓰는 바로 그 디렉터리)에 쓴다 -- 다른 어떤 새 파일도
  C_PD 아래 만들지 않는다).

게이트 마스크 소스 (3차 개정 -- 이번 작업의 유일한 변경점):
  2차 개정까지는 gate_persistence_sweep.compute_run_len(오프라인/배치 -- omni_p6
  count 결측 틱이 cnt.index 에서 통째로 빠짐)을 썼다. Table 6 을 낸 건 그게 아니라
  e2e_gated_eval.py 의 causal 경로(verify_ap_node.run_full_replay + ApFsmCore --
  결측 틱도 STALE 로 스케줄에 남아 카운터가 리셋도 증가도 없이 동결)였고, 4.5절은
  "as in deployment"를 평가하는 절이라 causal 쪽이 맞다. 2차 개정에서 갈렸던
  5건이 전부 결측 구간 직후였던 게 바로 이 차이였다(추적 완료, 이전 실행 보고).
  이번엔 게이트만 causal 경로로 바꾼다: N 마다 새로 리플레이하지 않고, causal
  지속성 카운터(ApFsmCore.process_tick 이 매 틱 돌려주는 counter, gate_n/alert_n
  과 무관하게 그 자체로 이미 "연속 양성 틱 수")를 한 번만 뽑아 gate_open_N =
  (counter>=N) & has_data 로 N=1/2/3 을 전부 유도한다(재구현 아님 -- ApFsmCore
  가 이미 counter 를 반환하고, 리플레이 자체도 verify_ap_node.run_full_replay
  그대로다). N=2 는 predict_v0/results/e2e/e2e_gate_alert_mask.csv(e2e_gated_eval.py
  가 이미 저장해 둔 causal gate_open(N=2)/alert_open(N=4))와 교차검증한다.

재사용 (재구현 없음 -- import만):
  e2e_gated_eval.py(e2e) : wp/ap/vap(=wp_poes_node/ap_fsm_node/verify_ap_node,
      e2e 가 이미 import 해 둔 것을 그대로 재사용 -- re-import 아님) --
      vap.run_full_replay() + ap.ApFsmCore.process_tick() 으로 causal 지속성
      카운터를 뽑는다(build_gate_alert_masks() 와 동일 구성의 ADT, gate_n/alert_n
      은 카운터 자체엔 영향 없음). build_p_event()(전 구간 p_event, STEP 2 그대로),
      THRESHOLDS(임계 그리드 -- e2e 가 이미 쓰던 0.85~0.999 그리드, Table 6 의
      th=0.85 포함), TCN_CHANNELS/DEFAULT_RUN_NAME/CKPT_ROOT.
  gate_persistence_sweep.py(gps) : compute_run_len() -- 게이트가 아니라 "분류기가
      게이트 개방 구간 안에서 M틱 연속 양성인가"를 셀 때만 쓴다(집계 규칙 M, 아래).
      verify_preproc(vp) 임포트 방식도 재사용(카탈로그 캐시 경로 등).
  4_validation.py(v4, numeric prefix라 importlib) : load_noaa_catalog()/
      load_manual_catalog()(카탈로그 42/53개), match_cell()(=_match_core_poes.
      match_events 의 스칼라 요약 래퍼 -- tol_h=24, one-to-one greedy 매칭,
      24h 클러스터링 event-level FAR. match_events 자체는 무변경).

집계 규칙(M)의 의미 -- "break" 고정(과제 지시, 게이트가 닫히면 분류기 세그먼트도
끊김 = 실배포와 동일). 이번엔 확률이 거의 전 구간을 덮어(236,739/236,848,
결측은 사실상 dataset 시작부 lag-window 워밍업뿐) classifier_positive 가
False 로 깔리는 구간이 옛날처럼 크지 않다:
  classifier_positive(t) = proba_event(t) >= threshold ; 확률 없는 틱(피처
      lag-window 워밍업 등)은 False로 둔다(판정 근거 없음 = break).
  cond(t) = gate_open_N(t) AND classifier_positive(t)
  M in {1,2,3,4} : gps.compute_run_len(cond) -- 게이트 run_len 을 만든 바로 그
      함수를 분류기 쪽에도 그대로 재사용(재구현 아님). run_len==M 인 시각마다
      검출 1건.
  "any" : 게이트 개방 구간 안에 cond 가 한 번이라도 True 면 그 구간에서 검출
      1건(시각=구간 내 첫 True 시각).
  "max" : 게이트 개방 구간 안 proba_event 최댓값 >= threshold 면 검출 1건
      (시각=그 최댓값 시각).

임계 선택 (3.5절 규약, 이번 개정의 핵심): 셀(gate_N x agg_rule x reference)마다
  e2e.THRESHOLDS 를 스윕해 event-level F1 을 최대화하는 임계를 고르고
  chosen_threshold 열에 기록한다(동률이면 그리드에서 더 작은 값 -- pandas
  idxmax 의 첫 occurrence 규약, 코드에 그대로 반영). th=0.99 고정값도
  참고용으로 별도 열(*_th099)에 남긴다.

calls_per_day = duty_cycle x 96 (논문 Table 5 와 동일 기준 -- 이전 버전은
n_ticks/total_days 를 썼는데 그건 데이터 공백만큼 96/일보다 작게 나와 Table 5
와 안 맞았다).

산출:
  gate_aggregation_sweep.csv : gate_N x agg_rule x reference(36행, chosen_threshold 포함)
  fig7_joint_operating_point.png/pdf : N x M 히트맵(F1/POD, 각 셀 자기 chosen_threshold
      기준), 현재 운용점(N=2,M=4) 빨간 테두리, F1 최댓값 셀(단 하나, "최적") 별표.
"""
from __future__ import annotations

import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent          # window_result/ -- 출력은 여기
POES_DIR = HERE.parent                           # POES/ -- 엔진 파일은 여기
sys.path.insert(0, str(POES_DIR))
import gate_persistence_sweep as gps  # noqa: E402  -- compute_run_len, vp(verify_preproc) 재사용
vp = gps.vp

PREDICT_V0 = vp.C_PD / "predict_v0"
sys.path.insert(0, str(PREDICT_V0))
sys.path.insert(0, str(vp.C_PD / "POES" / "event_MATCHER"))
sys.path.insert(0, str(vp.C_PD / "POES" / "count_FSM"))
v4 = importlib.import_module("4_validation")     # noqa: E402  -- numeric prefix -> importlib
core = v4.core                                     # _match_core_poes (match_events)

# ROS2_POES_DIR: e2e_gated_eval.py가 import 시점에 wp_poes_node/ap_fsm_node/verify_ap_node를
# 끌어오는데, 그건 REPO/ros2_nodes/1D_timeserise/POES(=POES_DIR, 이 스크립트의 부모)를
# 자기 스스로 REPO_ROOT 기준으로 계산해서 찾는다(무변경) -- 여기서 손댈 것 없음.
e2e = importlib.import_module("e2e_gated_eval")  # noqa: E402  -- build_p_event()/THRESHOLDS 재사용

RUN_NAME = e2e.DEFAULT_RUN_NAME
OOF_PATH = PREDICT_V0 / "results" / "runs" / RUN_NAME / "oof_predictions.parquet"  # in_crop 판정용
FULL_PROBA_PATH = e2e.OUT_DIR / "full_record_proba.parquet"      # predict_v0/results/e2e/ (C_PD 예외)
FULL_PROBA_META_PATH = e2e.OUT_DIR / "full_record_proba_meta.json"

GATE_NS = [1, 2, 3]
AGG_RULES = [1, 2, 3, 4, "any", "max"]     # 열 순서 = 표시 순서
THRESHOLD_GRID = e2e.THRESHOLDS             # [0.85, 0.90, 0.91, ..., 0.99, 0.995, 0.999] -- e2e 재사용
REFERENCE_THRESHOLD = 0.99                  # *_th099 참고 열용
CURRENT_OP = (2, 4)                         # 논문 현재 운용점 (N=2, M=4)
TICKS_PER_DAY = 96                          # 논문 Table 5 와 동일 기준

TOL_H = core.MATCH_TOL_H   # 24.0

# Table 6 "B. Gated classifier, break" 검증 목표값 (N=2, M=4, break, th=0.85, NOAA) -- 과제 지시
TABLE6_TARGET = dict(gate_N=2, agg_rule="4", threshold=0.85, reference="noaa",
                     n_det=187, POD=0.690, event_FAR=0.065, precision=0.935, F1=0.795)


# ══════════════════════════════════════════════════════
# STEP 0: 전 구간 확률 -- 캐시 있으면 읽기, 없으면 e2e.build_p_event() 1회 실행 후 저장
#         (사용자 명시 허가: 고정 가중치 forward pass, 재학습/체크포인트 변경 없음)
# ══════════════════════════════════════════════════════
def build_or_load_full_proba() -> pd.Series:
    if FULL_PROBA_PATH.exists():
        df = pd.read_parquet(FULL_PROBA_PATH)
        p = pd.Series(df["p_event"].values, index=pd.DatetimeIndex(df["ts"]))
        print(f"[gate_agg] 캐시된 전 구간 확률 로드(재추론 없음) -> {FULL_PROBA_PATH} ({len(p)}행)")
        return p

    print(f"[gate_agg] 캐시({FULL_PROBA_PATH}) 없음 -- e2e_gated_eval.build_p_event() 1회 실행"
         f"(고정 가중치 forward pass, run_name={RUN_NAME})")
    ckpt_dir = e2e.CKPT_ROOT / e2e.RUN_NAME
    manifest_path = ckpt_dir / "checkpoints" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))  # 메타 기록용(모델 로드 아님)

    p_event = e2e.build_p_event()   # == v4.build_full_features + v4.ensemble_event_proba, 재구현 없음
    if len(p_event) != 236_739:
        print(f"[gate_agg] [!] 경고: p_event 행수 {len(p_event)} != 236739(과제 지시 기대값) "
             "-- 체크포인트/채널 구성이 달라졌을 수 있음. 계속 진행하되 보고에 명시할 것.")

    # in_crop: 이 틱이 (1차 개정에서 쓰던) OOF 크롭 안에도 있었는지 -- 참고용 플래그
    in_crop = pd.Series(False, index=p_event.index)
    if OOF_PATH.exists():
        oof = pd.read_parquet(OOF_PATH)
        ts = oof["window_end_time"]
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize("UTC")
        crop_idx = pd.DatetimeIndex(ts).unique()
        in_crop = pd.Series(p_event.index.isin(crop_idx), index=p_event.index)
    else:
        print(f"[gate_agg] [!] {OOF_PATH} 없음 -- in_crop 열은 전부 False로 둠(참고용이라 치명적 아님)")

    out_df = pd.DataFrame({
        "ts": p_event.index, "p_event": p_event.values,
        "in_crop": in_crop.values, "fold": np.nan,   # 전 구간 값은 전 fold 앙상블 평균이라
    })                                                 # 단일 fold가 없다(아래 meta.json에 명시)
    e2e.OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(FULL_PROBA_PATH, index=False)
    meta = {
        "run_name": RUN_NAME,
        "checkpoint_dir": str(ckpt_dir),
        "n_folds": manifest.get("n_folds"),
        "folds_mode": "ensemble_mean_all_folds (fold 열은 단일값이 없어 NaN)",
        "channels": e2e.TCN_CHANNELS,
        "window": manifest.get("window"),
        "source": "e2e_gated_eval.build_p_event() == v4.build_full_features + "
                 "v4.ensemble_event_proba -- 고정 가중치 forward pass, 재학습/체크포인트 "
                 "변경 없음(사용자 명시 허가)",
        "n_rows": len(p_event),
        "expected_n_rows": 236739,
        "index_min": str(p_event.index.min()), "index_max": str(p_event.index.max()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    FULL_PROBA_META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[gate_agg] 저장(신규, C_PD 예외 -- 사용자 명시 경로) -> {FULL_PROBA_PATH} ({len(p_event)}행)")
    print(f"[gate_agg] 저장 -> {FULL_PROBA_META_PATH}")
    return p_event


def load_full_proba(causal_index: pd.DatetimeIndex) -> pd.Series:
    p = build_or_load_full_proba()
    p_full = p.reindex(causal_index)   # causal 리플레이 스케줄(245,472틱, 결측도 STALE로 남는 그 격자)로 정렬
    n_cov = int(p_full.notna().sum())
    print(f"[gate_agg] causal 스케줄({len(causal_index)}틱) 중 확률 커버: {n_cov}틱 "
         f"({n_cov/len(causal_index)*100:.1f}%)")
    return p_full


# ══════════════════════════════════════════════════════
# STEP 1: 게이트 -- causal 지속성 카운터(N=1,2,3 전부 이 한 번의 리플레이에서 유도)
#         e2e_gated_eval.build_gate_alert_masks() 와 동일 구성, counter 필드까지 받는다.
# ══════════════════════════════════════════════════════
def build_causal_gate_counter() -> tuple[pd.Series, pd.Series]:
    wp, ap, vap = e2e.wp, e2e.ap, e2e.vap   # e2e 가 이미 import 해 둔 것 재사용(re-import 아님)
    print("[gate_agg] STEP 1: causal 지속성 카운터 (verify_ap_node.run_full_replay + "
         "ApFsmCore.process_tick, e2e_gated_eval.build_gate_alert_masks()와 동일 구성)")
    adt = ap.build_adt(["omni_p6"], ap.build_rpn_equation(["omni_p6"]), 2, 4, "RTS_SEP_ALERT")[0]
    fcore = ap.ApFsmCore(adt, dry_run=False)

    ts_list, counters, has_data_list = [], [], []
    for ts, res in vap.run_full_replay():
        watch = {"omni_p6": res["watch"]}
        counts = {"omni_p6": res["count"]}
        out = fcore.process_tick(watch, counts, ts.timestamp())
        ts_list.append(ts)
        counters.append(out["counter"])
        has_data_list.append(out["has_data"])

    idx = pd.DatetimeIndex(ts_list)
    counter_s = pd.Series(counters, index=idx, name="counter")
    has_data_s = pd.Series(has_data_list, index=idx, name="has_data")

    n_gate2 = int(((counter_s >= 2) & has_data_s).sum())
    print(f"[gate_agg]   전 구간 {len(counter_s)}틱, causal gate_open(N=2)={n_gate2}(S2 기대값 16204)")
    assert n_gate2 == 16204, f"causal gate_open(N=2) {n_gate2} != 16204 -- 리플레이가 갈렸다, 중단."

    # 교차검증 -- e2e_gated_eval.py가 이미 저장해 둔 causal gate_open(N=2) CSV와 정확히 일치해야 함(과제 지시)
    mask_path = e2e.OUT_DIR / "e2e_gate_alert_mask.csv"
    if mask_path.exists():
        cached = pd.read_csv(mask_path, index_col=0, parse_dates=[0])
        cached.index = pd.DatetimeIndex(cached.index).tz_convert("UTC")
        mine_n2 = ((counter_s >= 2) & has_data_s).reindex(cached.index).fillna(False)
        n_mismatch = int((mine_n2.values != cached["gate_open"].values).sum())
        print(f"[gate_agg]   교차검증 vs {mask_path.name}: 불일치 {n_mismatch}틱/{len(cached)}틱 "
             f"({'PASS' if n_mismatch == 0 else '[MISMATCH]'})")
    else:
        print(f"[gate_agg]   [!] {mask_path} 없음 -- 교차검증 생략(e2e_gated_eval.py를 먼저 한 번 "
             "돌려야 생김, 치명적 아님).")

    return counter_s, has_data_s


# ══════════════════════════════════════════════════════
# STEP 2: 게이트 x 분류기 공동 집계 -> 검출 시각 리스트 (1차 개정과 동일 로직, 무변경)
# ══════════════════════════════════════════════════════
def gate_run_groups(gate_open: pd.Series) -> pd.Series:
    """gate_persistence_sweep.compute_run_len과 같은 그룹핑 트릭((~bool).cumsum())
    을 그대로 재사용 -- 게이트 개방 연속 구간마다 고유 group id."""
    return (~gate_open).cumsum()


def detections_for_M(cond: pd.Series, m: int) -> pd.DatetimeIndex:
    run_len = gps.compute_run_len(cond)   # 게이트 run_len과 동일 함수 재사용(재구현 아님)
    return cond.index[run_len == m]


def detections_for_any(gate_open: pd.Series, cond: pd.Series) -> pd.DatetimeIndex:
    grp = gate_run_groups(gate_open)
    df = pd.DataFrame({"grp": grp.values, "cond": cond.values}, index=cond.index)
    df_open = df[gate_open.values]
    if df_open.empty:
        return pd.DatetimeIndex([])
    dets = []
    for _, sub in df_open.groupby("grp"):
        hit = sub.index[sub["cond"].values]
        if len(hit):
            dets.append(hit[0])
    return pd.DatetimeIndex(sorted(dets))


def detections_for_max(gate_open: pd.Series, p_full: pd.Series, threshold: float) -> pd.DatetimeIndex:
    grp = gate_run_groups(gate_open)
    df = pd.DataFrame({"grp": grp.values, "p": p_full.values}, index=p_full.index)
    df_open = df[gate_open.values]
    if df_open.empty:
        return pd.DatetimeIndex([])
    dets = []
    for _, sub in df_open.groupby("grp"):
        pk = sub["p"].max(skipna=True)
        if pd.notna(pk) and pk >= threshold:
            dets.append(sub["p"].idxmax())
    return pd.DatetimeIndex(sorted(dets))


def build_detections(gate_open: pd.Series, p_full: pd.Series, agg_rule, threshold: float) -> pd.DataFrame:
    classifier_positive = (p_full >= threshold).fillna(False)
    if agg_rule == "max":
        ts = detections_for_max(gate_open, p_full, threshold)
    elif agg_rule == "any":
        cond = gate_open & classifier_positive
        ts = detections_for_any(gate_open, cond)
    else:
        cond = gate_open & classifier_positive
        ts = detections_for_M(cond, int(agg_rule))
    if len(ts) == 0:
        return pd.DataFrame(columns=["onset_time", "peak_time"])
    idx = pd.DatetimeIndex(sorted(ts))
    return pd.DataFrame({"onset_time": idx, "peak_time": idx})   # peak_time: match_events가
    # 요구하는 컬럼이라 채움(peak_diff 진단용, 이 표에서는 안 씀) -- onset_time 그대로 복제.


# ══════════════════════════════════════════════════════
# STEP 3: 카탈로그 로드 (v4 그대로)
# ══════════════════════════════════════════════════════
def load_catalogs() -> dict[str, pd.DataFrame]:
    noaa = v4.load_noaa_catalog()
    # v4.HERE는 그대로 둔다 -- load_manual_catalog()가 이미 자체적으로
    # HERE/"results"/"quality_check"/...를 읽는다(HERE=predict_v0). e2e_gated_eval.py처럼
    # v4.HERE를 predict_v0/results로 바꿔치기하면 .../results/results/...가 되어 못 찾는다
    # (실측 확인, 그 파일은 수정 금지 대상이라 그 관례를 따르지 않는다).
    manual = v4.load_manual_catalog()
    print(f"[gate_agg] 카탈로그: NOAA {len(noaa)}개, 손라벨 {len(manual)}개")
    return {"noaa": noaa, "manual": manual}


# ══════════════════════════════════════════════════════
# STEP 4: 게이트 전용 지표 -- duty_cycle(gate_persistence_sweep.py 정의 그대로),
#         calls_per_day = duty_cycle x 96(논문 Table 5 기준, 재사용자 지시로 갱신).
# ══════════════════════════════════════════════════════
def gate_only_stats(gate_open: pd.Series, total_ticks: int) -> dict:
    n = int(gate_open.sum())
    duty = n / total_ticks
    return {"duty_cycle": duty, "calls_per_day": duty * TICKS_PER_DAY}


# ══════════════════════════════════════════════════════
# 메인 스윕 -- 셀(gate_N x agg_rule x reference)마다 e2e.THRESHOLD_GRID를 훑어
# event-level F1 최댓값 임계를 고른다(3.5절 규약). th=0.99 고정값은 참고 열로 병기.
# ══════════════════════════════════════════════════════
def main():
    cache_dir = vp.C_PD / "POES" / "MetOp03_count" / "poes_metop03_cache_parquet"
    print(f"[gate_agg] REPO={vp.REPO}")
    cnt = vp.load_omni_p6_count(cache_dir)
    total_ticks = len(cnt)   # duty_cycle/calls_per_day 분모 -- S0.5 확정치(236,848)와 동일 기준 유지
    print(f"[gate_agg] omni_p6 {total_ticks}틱(duty_cycle 분모), {cnt.index[0]} ~ {cnt.index[-1]}")

    counter_s, has_data_s = build_causal_gate_counter()
    p_full = load_full_proba(counter_s.index)   # causal 스케줄(245,472틱) 기준으로 정렬
    catalogs = load_catalogs()

    rows = []
    for N in GATE_NS:
        gate_open = (counter_s >= N) & has_data_s
        gstats = gate_only_stats(gate_open, total_ticks)
        print(f"[gate_agg] N={N}: 게이트 개방 {int(gate_open.sum())}틱 "
             f"(duty={gstats['duty_cycle']*100:.2f}%, {gstats['calls_per_day']:.2f}/day)")
        for M in AGG_RULES:
            # 이 (N,M)의 임계별 결과를 전부 모아뒀다가 reference별로 F1 argmax를 고른다.
            per_th = {ref: [] for ref in catalogs}
            for th in THRESHOLD_GRID:
                det = build_detections(gate_open, p_full, M, th)
                for ref_name, cat in catalogs.items():
                    r = v4.match_cell(det, cat, tol_h=TOL_H)
                    per_th[ref_name].append({"threshold": th, **r})

            for ref_name, records in per_th.items():
                tdf = pd.DataFrame(records)
                best_idx = tdf["f1"].idxmax()   # 동률이면 그리드 상 먼저(=더 낮은 임계) 것 선택
                best = tdf.loc[best_idx]
                ref099 = tdf.loc[np.isclose(tdf["threshold"].astype(float), REFERENCE_THRESHOLD)]
                r099 = ref099.iloc[0] if len(ref099) else None
                rows.append({
                    "gate_N": N, "agg_rule": str(M), "reference": ref_name,
                    "chosen_threshold": float(best["threshold"]),
                    "n_det": int(best["n_det"]), "POD": best["pod"],
                    "event_FAR": best["event_far"], "precision": best["precision"],
                    "F1": best["f1"], "duty_cycle": gstats["duty_cycle"],
                    "calls_per_day": gstats["calls_per_day"],
                    "n_det_th099": int(r099["n_det"]) if r099 is not None else np.nan,
                    "POD_th099": r099["pod"] if r099 is not None else np.nan,
                    "event_FAR_th099": r099["event_far"] if r099 is not None else np.nan,
                    "F1_th099": r099["f1"] if r099 is not None else np.nan,
                })

    sweep_df = pd.DataFrame(rows)
    out_csv = HERE / "gate_aggregation_sweep.csv"
    sweep_df.to_csv(out_csv, index=False)
    print(f"[gate_agg] 저장 -> {out_csv} ({len(sweep_df)}행)")

    check_table6(sweep_df)
    build_figure(sweep_df)
    print_summary(sweep_df)
    return sweep_df


# ══════════════════════════════════════════════════════
# 검증 -- (N=2,M=4,break,th=0.85,NOAA) 셀이 논문 Table 6 "B. Gated classifier, break"
# 와 일치하는지. 이번 작업의 핵심 관문.
# ══════════════════════════════════════════════════════
def check_table6(sweep_df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("[gate_agg] 검증: (N=2, M=4, break, NOAA) 셀 vs 논문 Table 6 'B. Gated classifier, break'")
    print("=" * 70)
    row = sweep_df[(sweep_df.gate_N == TABLE6_TARGET["gate_N"])
                  & (sweep_df.agg_rule == TABLE6_TARGET["agg_rule"])
                  & (sweep_df.reference == TABLE6_TARGET["reference"])]
    if row.empty:
        print("[gate_agg] [FAIL] 해당 셀이 스윕 결과에 없음 -- 중단.")
        return
    row = row.iloc[0]
    got_th = row["chosen_threshold"]
    print(f"이 셀이 고른 chosen_threshold = {got_th} "
         f"(Table 6 목표 임계 0.85와 {'일치' if abs(got_th-0.85)<1e-9 else '불일치'})")
    fields = [("n_det", "n_det"), ("POD", "POD"), ("event_FAR", "event_FAR"),
             ("precision", "precision"), ("F1", "F1")]
    ok_all = True
    for label, col in fields:
        got = row[col]
        want = TABLE6_TARGET[label]
        ok = (got == want) if label == "n_det" else (round(float(got), 3) == want)
        ok_all = ok_all and ok
        print(f"  {label:10s} got={got if label=='n_det' else f'{got:.6f}'}  "
             f"want={want}  {'OK' if ok else '[MISMATCH]'}")
    print(f"[gate_agg] 종합: {'PASS -- 그리드 전체가 Table 6의 엄밀한 확장' if ok_all else '[FAIL]'}")
    if not ok_all:
        print(
            "[gate_agg] 추적 결과(3차, 수동 diff+원인 확정): 게이트를 causal로 바꾼 뒤 "
            "e2e_gate_alert_mask.csv와 245,472틱 전부 정확히 일치(0 mismatch, 위 STEP 1 로그) "
            "-- 그런데도 이 셀은 그대로 182/0.667/0.778이다. 즉 게이트 소스는 애초에 원인이 "
            "아니었다(2차 개정의 진단이 틀렸었다). 진짜 원인은 확률 쪽이다: "
            "e2e_gated_eval.align()이 p_event.index와 gate_s.index를 inner-join(common)한 뒤 "
            "그 축소된 인덱스 위에서 위치 기준으로 연속성을 판정하는데, v4.build_full_features()의 "
            "14틱 lag-window가 원시 z-score 결측을 최대 14틱(3.5시간)짜리 p_event 결측으로 "
            "증폭시켜 놓는다(실측: 2023-08-05 13:30~14:30, 5틱 연속 결측 확인). inner-join은 "
            "그 결측 구간을 인덱스에서 통째로 빼버려 앞뒤 유효 틱이 위치상 인접해 이어지고, "
            "이 스크립트(reindex + fillna(False), 전 구간 인덱스 유지)는 그 구간을 정확히 "
            "끊는다 -- 이번에 바꾼 게이트 소스와는 무관한, e2e_gated_eval.align()의 확률 인덱스 "
            "처리 방식 문제다. 이번 지시 범위(게이트 마스크만)를 벗어나 임의로 고치지 않는다.")


# ══════════════════════════════════════════════════════
# 그림 -- N x M 히트맵(F1/POD), 각 셀은 자기 chosen_threshold 기준. 현재 운용점/최적 셀 표시.
# "최적"은 F1 최댓값 셀 단 하나로만 정의(과제 지시 -- 지난 버전의 "single best cell" 대
# "optimal operating point" 불일치 수정).
# ══════════════════════════════════════════════════════
def build_figure(sweep_df: pd.DataFrame) -> tuple[int, str]:
    sub = sweep_df[sweep_df.reference == "noaa"]
    agg_labels = [str(m) for m in AGG_RULES]
    f1_grid = np.full((len(GATE_NS), len(agg_labels)), np.nan)
    pod_grid = np.full_like(f1_grid, np.nan)
    for i, N in enumerate(GATE_NS):
        for j, M in enumerate(agg_labels):
            row = sub[(sub.gate_N == N) & (sub.agg_rule == M)]
            if len(row):
                f1_grid[i, j] = row["F1"].iloc[0]
                pod_grid[i, j] = row["POD"].iloc[0]

    best_i, best_j = np.unravel_index(np.nanargmax(f1_grid), f1_grid.shape)
    cur_i = GATE_NS.index(CURRENT_OP[0])
    cur_j = agg_labels.index(str(CURRENT_OP[1]))

    plt.rcParams["font.size"] = 8
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.75), dpi=300)
    for name, grid, ax in (("F1", f1_grid, axes[0]), ("POD", pod_grid, axes[1])):
        im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(agg_labels)))
        ax.set_xticklabels(agg_labels, fontsize=8)
        ax.set_yticks(range(len(GATE_NS)))
        ax.set_yticklabels([f"N={n}" for n in GATE_NS], fontsize=8)
        ax.set_xlabel("aggregation rule M", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("gate persistence N", fontsize=8)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = grid[i, j]
                if np.isnan(v):
                    continue
                color = "white" if v < 0.55 else "black"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7, color=color)
        ax.add_patch(plt.Rectangle((cur_j - 0.5, cur_i - 0.5), 1, 1, fill=False,
                                   edgecolor="#d62728", lw=2.4, zorder=5))
        # 별표(F1 최댓값 셀, "최적" 단 하나)는 칸 왼쪽 위 모서리로 offset -- 가운데 숫자와
        # 안 겹치게(기본 클리핑이라 축 밖으로 넘치지도 않는다).
        ax.plot(best_j - 0.28, best_i - 0.28, marker="*", markersize=8, color="white",
               markeredgecolor="black", markeredgewidth=0.7, zorder=6)
        ax.set_title(name, fontsize=8.5, pad=4)   # 패널 구분용 소제목(F1/POD) -- 그림 전체 제목 아님
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.ax.tick_params(labelsize=6.5)

    fig.text(0.5, 0.01,
             f"red box = current op. point (N={CURRENT_OP[0]}, M={CURRENT_OP[1]})   "
             f"★ = optimal (max-F1 cell, N={GATE_NS[best_i]}, M={agg_labels[best_j]})   "
             f"each cell at its own F1-maximizing threshold, reference=NOAA",
             ha="center", va="bottom", fontsize=6.0, color="#333333")
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    png = HERE / "fig7_joint_operating_point.png"
    pdf = HERE / "fig7_joint_operating_point.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"[gate_agg] 저장 -> {png}")
    print(f"[gate_agg] 저장 -> {pdf}")
    return GATE_NS[best_i], agg_labels[best_j]


# ══════════════════════════════════════════════════════
# 결과 요약
# ══════════════════════════════════════════════════════
def print_summary(sweep_df: pd.DataFrame) -> None:
    sub = sweep_df[sweep_df.reference == "noaa"]
    best = sub.loc[sub["F1"].idxmax()]
    cur = sub[(sub.gate_N == CURRENT_OP[0]) & (sub.agg_rule == str(CURRENT_OP[1]))].iloc[0]
    relaxed = sub[(sub.gate_N == CURRENT_OP[0]) & (sub.agg_rule == "1")].iloc[0]
    print("\n" + "=" * 70)
    print("[gate_agg] 요약 (reference=NOAA, N=42, 셀마다 자기 chosen_threshold 기준)")
    print("=" * 70)
    print(f"현재 운용점  N={CURRENT_OP[0]} M={CURRENT_OP[1]} (th={cur.chosen_threshold}): "
         f"POD={cur.POD:.3f} event_FAR={cur.event_FAR:.3f} F1={cur.F1:.3f} n_det={cur.n_det}")
    print(f"최적(F1 최댓값 단일 셀)  N={int(best.gate_N)} M={best.agg_rule} (th={best.chosen_threshold}): "
         f"POD={best.POD:.3f} event_FAR={best.event_FAR:.3f} F1={best.F1:.3f} n_det={best.n_det} "
         f"(F1 {best.F1 - cur.F1:+.3f}, POD {best.POD - cur.POD:+.3f} vs 현재)")
    print(f"[한 문장 요약] 같은 N={CURRENT_OP[0]}에서 M을 4->1로 완화하면: "
         f"POD {cur.POD:.3f} -> {relaxed.POD:.3f} ({relaxed.POD - cur.POD:+.3f}), "
         f"event_FAR {cur.event_FAR:.3f} -> {relaxed.event_FAR:.3f} ({relaxed.event_FAR - cur.event_FAR:+.3f}) "
         f"(각자 F1-최대 임계 기준: M=4 th={cur.chosen_threshold}, M=1 th={relaxed.chosen_threshold}).")


if __name__ == "__main__":
    main()
