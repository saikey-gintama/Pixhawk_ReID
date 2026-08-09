"""
e2e_gated_eval.py
==================
작업 2: 게이팅된 파이프라인의 end-to-end 검출 성능 계산 (Windows 오프라인 분석,
젯슨 불필요, 새 학습 없음). 읽기 전용(체크포인트/카탈로그) + --out-dir 아래 4개
파일만 새로 쓴다. C_PD/predict_v0 아래 기존 파일은 전혀 건드리지 않는다.

배경: 논문 아키텍처는 "게이트가 열릴 때만 TCN이 돈다"인데, 기존 표에는 게이트
단독 성능(Table 5)과 TCN 단독 성능(Table 3)만 있고 둘을 연결한 최종 검출
성능이 없었다. 이 스크립트가 그 빈칸을 채운다.

네 구성 (전 구간 공통 시간 인덱스 위에서 A/B, ALERT latch 이벤트 위에서 C/D):
  A  TCN 단독            p_event >= thr                         (연속 확률->세그먼트, 1h 필터)
  B  게이트 제한 TCN     p_event >= thr  AND  gate_open(N=2)     (연속 확률->세그먼트, 1h 필터,
                                                                   break/skip 두 변형)
  C  ALERT ∧ TCN veto    ALERT latch 순간에 p_event(그 틱) >= thr (이산 순간조건, 지속필터 없음)
  D  FSM ALERT 단독      ALERT latch 그 자체                     (이산 latch 전이, 지속필터 없음)

C/D 재정의 이유(범주 오류 수정): ALERT(alert_n=4)는 이 논문에서 "latched transition"으로
정의된 이산 상태 전이다(3.3절) -- 실현되면 그 순간 1건의 확정 이벤트지, 그 뒤로도 계속
지속되는 연속 신호가 아니다. 이전 버전은 alert_open 불리언에 probs_to_onsets 의 1시간(4틱)
연속 세그먼트 필터를 한 번 더 적용해 항상 0건이 나왔다(ALERT 지속시간 중앙값 2틱=30분,
최댓값 3틱=45분 -- alert_n=4 는 "진입 조건"이지 "그 이후 지속시간"이 아니므로 구조적으로
1시간을 못 채운다). 그 필터는 TCN 확률 시계열을 검출로 바꾸는 규칙(A/B에는 맞다)이지,
이미 이산 전이인 ALERT(C/D)에는 적용 대상이 아니었다.

재사용 (재구현 없음 -- import만):
  wp_poes_node.py / ap_fsm_node.py : ChannelState/ApFsmCore -- gate_open/alert_open 마스크와
      ALERT latch 이벤트를 S2(verify_ap_node.py)와 정확히 같은 방식(단일채널 omni_p6, k=7/
      onset_floor=0.1/gate_n=2/alert_n=4)으로 재생성. verify_ap_node.run_full_replay() 를
      그대로 가져다 쓴다(리플레이 드라이버 재구현 없음). ALERT latch 판정 자체는
      ApFsmCore.process_tick 의 기존 latch 로직(다운링크 큐에 최초 확정 1회만 적재)을 그대로
      쓴다 -- 새 latch 판정을 만들지 않는다.
  4_validation.py(as v4) : build_full_features/ensemble_event_proba(TCN p_event 전 구간
      시계열, 채널=omni_p6,pro_tel0_p5,pro_tel90_p5 "승자" 체크포인트) + probs_to_onsets
      (fsm_engine.detect_segments 재사용 경로, A/B 전용) + match_cell + load_tcn_ensemble/
      load_noaa_catalog/load_manual_catalog. 전부 무변경 재사용.
  _match_core_poes.py(as core, v4.core) : det_matched_mask -- 카탈로그 매칭뿐 아니라 "A의
      검출이 B/C에도 살아남았는가"를 판정하는 자기매칭에도 그대로 재사용(cat 인자 자리에
      B/C의 det를 onset_time 인덱스로 바꿔 넣을 뿐, 매칭 로직 자체는 무변경).

경로: --ckpt-root(기본 predict_v0/results/runs) 아래 --run-name(기본
omni_p6_pro_tel0_p5_pro_tel90_p5_binary) 서브디렉토리의 체크포인트를 쓰고,
--out-dir(기본 predict_v0/results/e2e) 에 결과를 쓴다. 손라벨 카탈로그(quality_check/)는
predict_v0/results/quality_check/ 를 그대로 읽는다(4_validation.py 의 load_manual_catalog
자체는 무변경 -- 호출 구간에서만 v4.HERE 를 잠깐 results/ 로 바꿔치기했다가 되돌린다).

게이트 닫힌 틱 처리 (B만 해당): "추론 없음"이지 "임계 미만"이 아니다. 두 방식 모두 낸다.
  break : 조건 계산에서 False로 취급 -- 진행 중이던 세그먼트를 끊는다.
  skip  : 그 틱을 시계열에서 제거 -- 세그먼트 연속성 판정에 나타나지 않으므로 앞뒤의 True
          상태가 이어진다(S1/S2의 STALE 동결과 같은 태도).
  C는 이 구분이 필요 없다(순간조건이라 연속성 개념 자체가 없음).

필수 관문 두 개(둘 다 실패 시 중단):
  ① A(TCN 단독) vs 기존 comparison_table.csv 의 TCN-multi 행.
  ② D(FSM ALERT 단독) vs 논문 4.6절 causal 재현(216 onsets, POD 0.738, event_FAR 0.061, NOAA).

산출: <out-dir>/e2e_gated.csv, e2e_removed.csv, e2e_summary.md, e2e_gate_alert_mask.csv
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent            # Experiment_window/C_PD/predict_v0/
C_PD = HERE.parent                                  # Experiment_window/C_PD
REPO_ROOT = C_PD.parent.parent                      # 레포 루트
ONBOARD_POES_DIR = REPO_ROOT / "ros2_nodes" / "1D_timeserise" / "POES"   # wp/ap/verify_ap_node 위치

sys.path.insert(0, str(ONBOARD_POES_DIR))
import wp_poes_node as wp          # noqa: E402
import ap_fsm_node as ap           # noqa: E402
import verify_ap_node as vap       # noqa: E402  -- run_full_replay() 재사용(단일채널 omni_p6)

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(C_PD / "POES" / "event_MATCHER"))
sys.path.insert(0, str(C_PD / "POES" / "count_FSM"))
v4 = importlib.import_module("4_validation")       # noqa: E402
core = v4.core                                       # _match_core_poes (det_matched_mask 재사용)

TCN_CHANNELS = ["omni_p6", "pro_tel0_p5", "pro_tel90_p5"]
DEFAULT_RUN_NAME = "omni_p6_pro_tel0_p5_pro_tel90_p5_binary"

MIN_DURATION_H = v4.MIN_DURATION_H     # 1h, fsm_engine.MIN_SPE_DURATION_H -- A/B 전용
TOL_H = v4.TOL_H                       # 24h, core.MATCH_TOL_H -- 카탈로그 매칭 & 자기매칭 공용

# 0.9~0.999 조밀 스윕(과제 지시) + 0.85(comparison_table.csv 의 TCN-multi manual max_f1
# 운용점) -- A 대조(필수 관문 ①)에 그대로 쓰기 위해 포함.
THRESHOLDS = [0.85, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999]

# comparison_table.csv(results/validation_v0/validation_ch_20260808_1704) 의 TCN multi 행 --
# 필수 관문 ①.
BASELINE_A = {
    ("noaa", 0.99): dict(n_det=217, TP=40, FP=10, FN=2, pod=0.9523809523809523,
                         event_far=0.2, precision=0.8, f1=0.8695652173913043),
    ("manual", 0.85): dict(n_det=155, TP=46, FP=38, FN=7, pod=0.8679245283018868,
                           event_far=0.4523809523809524, precision=0.5476190476190477,
                           f1=0.6715328467153285),
}

# 논문 4.6절 causal ALERT 재현(NOAA 기준) -- 필수 관문 ②.
PAPER_4_6_ALERT = dict(catalog="noaa", n_det=216, pod=0.738, event_far=0.061)

# 모듈 전역(argparse 로 main() 에서 덮어씀 -- ap_fsm_node.py 의 관례 승계)
CKPT_ROOT = HERE / "results" / "runs"
RUN_NAME = DEFAULT_RUN_NAME
OUT_DIR = HERE / "results" / "e2e"


# ══════════════════════════════════════════════════════
# STEP 1: gate_open/alert_open 마스크 + ALERT latch 이벤트
#         (단일채널 omni_p6, S2 검증 경로 재사용)
# ══════════════════════════════════════════════════════
def build_gate_alert_masks() -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    print("[e2e] STEP 1: gate_open/alert_open 마스크 + ALERT latch 이벤트 "
         "(단일채널 omni_p6, verify_ap_node.run_full_replay 재사용)")
    adt = ap.build_adt(["omni_p6"], ap.build_rpn_equation(["omni_p6"]), 2, 4, "RTS_SEP_ALERT")[0]
    fcore = ap.ApFsmCore(adt, dry_run=False)   # dry_run=False 라야 latch/downlink_queue 가 쌓인다

    ts_list: list = []
    gate_open: list[bool] = []
    alert_open: list[bool] = []
    alert_latch_ts: list = []      # ALERT 최초 확정(latch) 시각 -- C/D 의 원재료

    for ts, res in vap.run_full_replay():
        watch = {"omni_p6": res["watch"]}
        counts = {"omni_p6": res["count"]}
        out = fcore.process_tick(watch, counts, ts.timestamp())
        ts_list.append(ts)
        # has_data 가드 필수: STALE 틱은 counter 가 동결돼 state 가 PRE_ALERT/ALERT 로
        # "남아있을" 수 있지만 실제 관측이 없으므로 AI 는 트리거하지 않는다
        # (ai_tcn_node.py:237 `if ap_state not in (...) or not has_data` 와 동일 조건).
        gate_open.append(out["has_data"] and out["state"] in (ap.STATE_PRE_ALERT, ap.STATE_ALERT))
        alert_open.append(out["has_data"] and out["state"] == ap.STATE_ALERT)
        if out["transitioned"] and out["state"] == ap.STATE_ALERT and out["from_state"] != ap.STATE_ALERT:
            alert_latch_ts.append(ts)   # latch 순간의 실제 Timestamp 를 그대로 보존(재인코딩 없음)

    idx = pd.DatetimeIndex(ts_list)
    gate_s = pd.Series(gate_open, index=idx, name="gate_open")
    alert_s = pd.Series(alert_open, index=idx, name="alert_open")

    n_gate_open = int(gate_s.sum())
    n_alert_transitions = len(alert_latch_ts)
    print(f"[e2e]   gate_open 틱={n_gate_open}(기대 16204)  alert 전이={n_alert_transitions}(기대 216)  "
          f"core.n_ticks_gate_open={fcore.n_ticks_gate_open}  core.n_ticks_alert_open={fcore.n_ticks_alert_open}  "
          f"downlink_queue={len(fcore.downlink_queue)}(교차확인, alert 전이와 같아야 함)")
    ok = (n_gate_open == 16204 and n_alert_transitions == 216 and fcore.n_ticks_gate_open == 16204
         and len(fcore.downlink_queue) == 216)
    if not ok:
        raise SystemExit("[e2e] 게이트/ALERT 마스크가 S2 검증값(16204/216)과 불일치 -- 중단. "
                         "재계산 로직이 verify_ap_node.py 와 갈렸는지 먼저 확인할 것.")
    print("[e2e]   S2 검증값과 정확히 일치 확인.")

    # ALERT latch 이벤트 -- 이산 순간 이벤트라 onset=peak=end=latch 시각(과제 지시:
    # "onset = ALERT 진입(latch) 시각"). 지속시간·프로파일 개념이 없으므로 duration_h 없음.
    alert_events = pd.DataFrame({
        "onset_time": pd.DatetimeIndex(alert_latch_ts),
        "peak_time": pd.DatetimeIndex(alert_latch_ts),
        "end_time": pd.DatetimeIndex(alert_latch_ts),
    })

    mask_path = OUT_DIR / "e2e_gate_alert_mask.csv"
    pd.DataFrame({"gate_open": gate_s, "alert_open": alert_s}).to_csv(mask_path)
    print(f"[e2e]   마스크 CSV 덤프 -> {mask_path} ({len(gate_s)}틱)")
    return gate_s, alert_s, alert_events


# ══════════════════════════════════════════════════════
# STEP 2: TCN p_event 전 구간 시계열 (build_full_features/ensemble_event_proba 재사용)
# ══════════════════════════════════════════════════════
def build_p_event() -> pd.Series:
    ckpt_dir = CKPT_ROOT / RUN_NAME
    print(f"[e2e] STEP 2: TCN p_event 전 구간 시계열 (채널={TCN_CHANNELS}, run={ckpt_dir})")
    loaded = v4.load_tcn_ensemble(ckpt_dir)
    if loaded is None:
        raise SystemExit(f"[e2e] TCN 체크포인트 없음: {ckpt_dir}")
    models, manifest = loaded
    X, time_idx, _ = v4.build_full_features("metop03", TCN_CHANNELS, manifest["window"])
    p_event = pd.Series(v4.ensemble_event_proba(models, manifest, X), index=time_idx)
    print(f"[e2e]   p_event {len(p_event)}틱, 범위 {p_event.index.min()} ~ {p_event.index.max()}")
    return p_event


# ══════════════════════════════════════════════════════
# STEP 3: 카탈로그 (경로 보정만, 로직 무변경)
# ══════════════════════════════════════════════════════
def load_catalogs() -> dict[str, pd.DataFrame]:
    print("[e2e] STEP 3: 카탈로그 로드")
    noaa = v4.load_noaa_catalog()
    orig_here = v4.HERE
    try:
        v4.HERE = HERE / "results"   # quality_check/ 가 results/ 아래로 이동됨(경로만 보정)
        manual = v4.load_manual_catalog()
    finally:
        v4.HERE = orig_here
    print(f"[e2e]   NOAA {len(noaa)}개, 손라벨 {len(manual)}개")
    return {"noaa": noaa, "manual": manual}


# ══════════════════════════════════════════════════════
# STEP 4: 인덱스 정렬 (timestamp inner join, 위치 정렬 금지) -- A/B 전용.
#         C/D는 ALERT latch 이벤트가 원재료라 이 정렬과 무관(D는 전 구간 그대로 써야
#         논문 4.6절의 216건을 재현한다 -- 공통 인덱스로 자르면 안 됨).
# ══════════════════════════════════════════════════════
def align(p_event: pd.Series, gate_s: pd.Series):
    common = p_event.index.intersection(gate_s.index).sort_values()
    print(f"[e2e] STEP 4: 인덱스 정렬(A/B용) -- p_event {len(p_event)}틱, gate마스크 {len(gate_s)}틱, "
          f"공통 틱 {len(common)}개 (timestamp 기준 inner join)")
    dropped = len(p_event) - len(common)
    print(f"[e2e]   p_event 중 게이트 마스크 범위 밖(드롭) {dropped}개")
    p_c = p_event.loc[common]
    gate_c = gate_s.loc[common]
    return p_c, gate_c, common


# ══════════════════════════════════════════════════════
# 세그먼트 빌더 (A/B) -- v4.probs_to_onsets(=fsm_engine.detect_segments) 재사용
# ══════════════════════════════════════════════════════
def det_from_bool(bool_series: pd.Series) -> pd.DataFrame:
    """불리언 시리즈 -> 검출 구간. 0/1 시리즈 + 고정임계 0.5 로 probs_to_onsets 를
    재사용(세그먼트 알고리즘 자체는 무변경)."""
    return v4.probs_to_onsets(bool_series.astype(float), 0.5, MIN_DURATION_H)


def config_A(p_c: pd.Series, th: float) -> pd.DataFrame:
    return v4.probs_to_onsets(p_c, th, MIN_DURATION_H)


def config_B_break(p_c: pd.Series, gate_c: pd.Series, th: float) -> pd.DataFrame:
    cond = (p_c >= th) & gate_c
    return det_from_bool(cond)


def config_B_skip(p_c: pd.Series, gate_c: pd.Series, th: float) -> pd.DataFrame:
    sub = p_c.loc[gate_c[gate_c].index]   # 게이트 닫힌 틱을 인덱스에서 제거(건너뜀)
    return v4.probs_to_onsets(sub, th, MIN_DURATION_H)


# ══════════════════════════════════════════════════════
# C/D -- ALERT latch(이산 전이) 기반. 지속 필터 없음(범주 오류 수정).
# ══════════════════════════════════════════════════════
def config_D(alert_events: pd.DataFrame) -> pd.DataFrame:
    """FSM ALERT 단독: latch 이벤트 그대로(216건), 추가 필터 없음."""
    return alert_events.copy()


def config_C(p_event_full: pd.Series, alert_events: pd.DataFrame, th: float) -> pd.DataFrame:
    """ALERT latch 순간의 p_event(그 틱, timestamp 정확 일치 -- 위치 정렬 아님) >= thr 인
    latch 만 남긴다. 순간조건이라 지속 필터가 없다(따라서 break/skip 구분도 없다)."""
    if alert_events.empty:
        return alert_events.copy()
    p_at_latch = p_event_full.reindex(alert_events["onset_time"])   # timestamp 라벨 매칭
    keep = np.isfinite(p_at_latch.values) & (p_at_latch.values >= th)
    return alert_events[keep].reset_index(drop=True)


# ══════════════════════════════════════════════════════
# STEP 5: 임계 스윕 + 카탈로그 매칭 (match_cell 재사용)
# ══════════════════════════════════════════════════════
def sweep_all(p_c, gate_c, p_event_full, alert_events, catalogs):
    print("[e2e] STEP 5: 임계 스윕 + 카탈로그 매칭")
    rows = []
    det_cache: dict[tuple, pd.DataFrame] = {}

    det_D = config_D(alert_events)
    det_cache[("D", "na", None)] = det_D
    for cat_name, cat in catalogs.items():
        r = v4.match_cell(det_D, cat)
        rows.append({"config": "D_alert_only", "variant": "na", "threshold": np.nan,
                    "ground_truth": cat_name, "n_det_raw": len(det_D), **r})

    for th in THRESHOLDS:
        det_A = config_A(p_c, th)
        det_Bb = config_B_break(p_c, gate_c, th)
        det_Bs = config_B_skip(p_c, gate_c, th)
        det_C = config_C(p_event_full, alert_events, th)   # p_event 전체(공통 인덱스 제한 안함)로 순간조회

        det_cache[("A", "na", th)] = det_A
        det_cache[("B", "break", th)] = det_Bb
        det_cache[("B", "skip", th)] = det_Bs
        det_cache[("C", "na", th)] = det_C

        for cfg_label, variant, det in (
            ("A_tcn_only", "na", det_A),
            ("B_gated_tcn", "break", det_Bb),
            ("B_gated_tcn", "skip", det_Bs),
            ("C_alert_and_tcn", "na", det_C),
        ):
            for cat_name, cat in catalogs.items():
                r = v4.match_cell(det, cat)
                rows.append({"config": cfg_label, "variant": variant, "threshold": th,
                            "ground_truth": cat_name, "n_det_raw": len(det), **r})

    sweep_df = pd.DataFrame(rows)
    print(f"[e2e]   스윕 완료: {len(sweep_df)}행")
    return sweep_df, det_cache


# ══════════════════════════════════════════════════════
# 필수 관문 ①: A vs 기존 comparison_table.csv 대조
# ══════════════════════════════════════════════════════
def check_baseline_a(sweep_df: pd.DataFrame) -> list[str]:
    print("[e2e] 필수 관문 ①: A(TCN 단독) vs 기존 comparison_table.csv 대조")
    lines = []
    ok_all = True
    for (cat_name, th), exp in BASELINE_A.items():
        row = sweep_df[(sweep_df.config == "A_tcn_only") & (sweep_df.ground_truth == cat_name)
                       & (np.isclose(sweep_df.threshold.astype(float), th))]
        if row.empty:
            msg = f"  [FAIL] {cat_name}@th={th}: 스윕 결과에 해당 임계 행 없음"
            print(msg); lines.append(msg); ok_all = False
            continue
        row = row.iloc[0]
        mism = {}
        for k, v in exp.items():
            got = row[k]
            bad = (abs(got - v) > 1e-6) if isinstance(v, float) else (got != v)
            if bad:
                mism[k] = (got, v)
        status = "PASS" if not mism else "FAIL"
        msg = f"  [{status}] {cat_name}@th={th}: " + ("전부 일치" if not mism else f"불일치 {mism}")
        print(msg); lines.append(msg)
        ok_all = ok_all and not mism
    print(f"[e2e]   관문 ① 종합: {'PASS' if ok_all else 'FAIL'}")
    if not ok_all:
        raise SystemExit("[e2e] A 가 기존 baseline 과 불일치 -- 중단. "
                         "인덱스 정렬/임계 처리 점검 필요(B/C/D 결과는 신뢰 불가).")
    return lines


# ══════════════════════════════════════════════════════
# 필수 관문 ②: D vs 논문 4.6절 causal ALERT 재현
# ══════════════════════════════════════════════════════
def check_baseline_d(sweep_df: pd.DataFrame) -> list[str]:
    print("[e2e] 필수 관문 ②: D(FSM ALERT 단독) vs 논문 4.6절 causal 재현 대조")
    lines = []
    row = sweep_df[(sweep_df.config == "D_alert_only") & (sweep_df.ground_truth == PAPER_4_6_ALERT["catalog"])]
    if row.empty:
        raise SystemExit("[e2e] D 결과 행이 없음 -- 중단.")
    row = row.iloc[0]
    got_n, got_pod, got_far = int(row["n_det"]), float(row["pod"]), float(row["event_far"])
    ok_n = (got_n == PAPER_4_6_ALERT["n_det"])
    ok_pod = (round(got_pod, 3) == PAPER_4_6_ALERT["pod"])
    ok_far = (round(got_far, 3) == PAPER_4_6_ALERT["event_far"])
    ok = ok_n and ok_pod and ok_far
    msg = (f"  [{'PASS' if ok else 'FAIL'}] D vs 논문 4.6절(causal ALERT, {PAPER_4_6_ALERT['catalog']}): "
          f"n_det={got_n}(기대 {PAPER_4_6_ALERT['n_det']}, {'OK' if ok_n else 'FAIL'})  "
          f"POD={got_pod:.4f}->round3={round(got_pod,3)}(기대 {PAPER_4_6_ALERT['pod']}, {'OK' if ok_pod else 'FAIL'})  "
          f"event_FAR={got_far:.4f}->round3={round(got_far,3)}(기대 {PAPER_4_6_ALERT['event_far']}, {'OK' if ok_far else 'FAIL'})")
    print(msg); lines.append(msg)
    print(f"[e2e]   관문 ② 종합: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit("[e2e] D 가 논문 4.6절 causal 재현과 불일치 -- 중단. "
                         "매칭 tolerance 또는 causal/offline 카탈로그 구분을 점검할 것.")
    return lines


# ══════════════════════════════════════════════════════
# 파생 지표: A 대비 게이팅이 지운 검출 (자기매칭에 det_matched_mask 재사용)
# ══════════════════════════════════════════════════════
def removed_analysis(det_cache, catalogs, sweep_df):
    print("[e2e] 파생 지표: A 최적임계(F1 최대)에서 게이팅이 지운 검출")
    removed_parts = []
    summary_lines = []

    for cat_name, cat in catalogs.items():
        a_rows = sweep_df[(sweep_df.config == "A_tcn_only") & (sweep_df.ground_truth == cat_name)]
        best = a_rows.loc[a_rows["f1"].astype(float).idxmax()]
        th_opt = float(best["threshold"])
        det_A = det_cache[("A", "na", th_opt)]
        n_det_A = len(det_A)
        summary_lines.append(f"\n### 기준: {cat_name} (A 최적임계, F1={best['f1']:.4f}, th={th_opt})\n")

        for cfg_key, cfg_label, variant_label in (
            (("B", "break", th_opt), "B_gated_tcn", "break"),
            (("B", "skip", th_opt), "B_gated_tcn", "skip"),
            (("C", "na", th_opt), "C_alert_and_tcn", "na"),
        ):
            det_X = det_cache[cfg_key]
            n_det_X = len(det_X)
            if n_det_X > 0:
                survived = core.det_matched_mask(det_A, det_X.set_index("onset_time"), tol_h=TOL_H)
            else:
                survived = np.zeros(len(det_A), dtype=bool)
            removed = det_A[~survived].copy()
            n_removed = len(removed)
            if n_removed > 0:
                true_mask = core.det_matched_mask(removed, cat, tol_h=TOL_H)
            else:
                true_mask = np.zeros(0, dtype=bool)
            n_true = int(true_mask.sum())
            n_false = n_removed - n_true

            if n_removed > 0:
                removed = removed.assign(ground_truth=cat_name, config=cfg_label,
                                         variant=variant_label, cfg_threshold=th_opt,
                                         matched_catalog=true_mask)
                removed_parts.append(removed)

            line = (f"- n_det_A={n_det_A}, n_det_{cfg_label}({variant_label})={n_det_X}, "
                   f"n_removed={n_removed} (true={n_true}, false={n_false})")
            print(f"[e2e]   {cat_name}/{cfg_label}/{variant_label}: {line[2:]}")
            summary_lines.append(line)

    removed_df = pd.concat(removed_parts, ignore_index=True) if removed_parts else pd.DataFrame()
    return removed_df, summary_lines


# ══════════════════════════════════════════════════════
# 출력
# ══════════════════════════════════════════════════════
def write_summary_md(common_index, gate_a_lines, gate_d_lines, removed_lines):
    path = OUT_DIR / "e2e_summary.md"
    lines = []
    lines.append("# E2E 게이팅 파이프라인 검출 성능 요약\n")
    lines.append(f"공통 시간 인덱스(A/B, timestamp inner join): **{len(common_index)}틱** "
                f"({common_index.min()} ~ {common_index.max()}).\n")

    lines.append("## 구성 정의\n")
    lines.append("- **A** TCN 단독: `p_event >= thr` (연속 확률 -> 세그먼트, 1시간 필터)\n")
    lines.append("- **B** 게이트 제한 TCN: `p_event >= thr AND gate_open(N=2)` "
                "(연속 확률 -> 세그먼트, 1시간 필터, break/skip 두 변형)\n")
    lines.append("- **C** ALERT ∧ TCN veto: ALERT latch 순간에 `p_event(그 틱) >= thr` "
                "(이산 순간조건, 지속필터 없음)\n")
    lines.append("- **D** FSM ALERT 단독: ALERT latch 그 자체 (이산 전이, 지속필터 없음)\n")

    lines.append("\n## C/D 재정의 (범주 오류 수정)\n")
    lines.append("ALERT(alert_n=4)는 latched transition(이산 상태 전이)이지 연속 신호가 아니다. "
                "이전 버전은 여기에 probs_to_onsets 의 1시간 연속 세그먼트 필터를 한 번 더 적용해 "
                "항상 0건이 나왔다(ALERT 지속시간 중앙값 30분·최댓값 45분, 1시간을 구조적으로 "
                "못 채움). C/D는 이제 ALERT latch 이벤트(216건)를 원재료로 쓴다: D는 그대로, "
                "C는 latch 순간의 p_event(그 틱)가 thr 이상인지만 순간적으로 확인한다.\n")

    lines.append("\n## 게이트 닫힌 틱의 세그먼트 연속성 처리 (B만 해당)\n")
    lines.append("게이트가 닫힌 틱에서 p_event는 '임계 미만'이 아니라 '추론 없음'이다. "
                "두 해석을 모두 계산했다:\n")
    lines.append("- **break**: 조건 계산에서 False로 취급 -- 진행 중이던 세그먼트를 끊는다.\n")
    lines.append("- **skip**: 그 틱을 시계열에서 제거 -- 연속성 판정에 나타나지 않으므로 "
                "앞뒤의 True 상태가 이어진다(S1/S2 STALE 동결과 동일한 태도).\n")
    lines.append("C는 순간조건이라 연속성 개념 자체가 없어 이 구분이 필요 없다.\n")

    lines.append("\n## 필수 관문 ①: A 대 기존 comparison_table.csv 대조\n")
    lines.extend(f"{ln}\n" for ln in gate_a_lines)

    lines.append("\n## 필수 관문 ②: D 대 논문 4.6절 causal ALERT 재현\n")
    lines.extend(f"{ln}\n" for ln in gate_d_lines)

    lines.append("\n## 게이팅이 제거한 검출 (A 최적임계, F1 최대 기준)\n")
    lines.extend(f"{ln}\n" for ln in removed_lines)

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[e2e] 저장 -> {path}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="e2e_gated_eval -- 게이팅된 파이프라인 end-to-end 검출 성능 (오프라인)")
    p.add_argument("--ckpt-root", type=Path, default=HERE / "results" / "runs",
                   help="TCN 체크포인트 루트(아래 --run-name/checkpoints/manifest.json). "
                        "기본: predict_v0/results/runs")
    p.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME,
                   help="ckpt-root 아래 TCN run 디렉토리 이름(3채널 승자 binary 체크포인트)")
    p.add_argument("--out-dir", type=Path, default=HERE / "results" / "e2e",
                   help="e2e_gated.csv/e2e_removed.csv/e2e_summary.md 출력 디렉토리. "
                        "기본: predict_v0/results/e2e")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    global CKPT_ROOT, RUN_NAME, OUT_DIR
    CKPT_ROOT = args.ckpt_root
    RUN_NAME = args.run_name
    OUT_DIR = args.out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[e2e] ckpt_root={CKPT_ROOT}  run_name={RUN_NAME}  out_dir={OUT_DIR}")

    gate_s, alert_s, alert_events = build_gate_alert_masks()
    p_event = build_p_event()
    catalogs = load_catalogs()
    p_c, gate_c, common = align(p_event, gate_s)

    sweep_df, det_cache = sweep_all(p_c, gate_c, p_event, alert_events, catalogs)
    gate_a_lines = check_baseline_a(sweep_df)
    gate_d_lines = check_baseline_d(sweep_df)

    gated_csv = OUT_DIR / "e2e_gated.csv"
    sweep_df.to_csv(gated_csv, index=False)
    print(f"[e2e] 저장 -> {gated_csv} ({len(sweep_df)}행)")

    removed_df, removed_lines = removed_analysis(det_cache, catalogs, sweep_df)
    removed_csv = OUT_DIR / "e2e_removed.csv"
    if not removed_df.empty:
        removed_df.to_csv(removed_csv, index=False)
    print(f"[e2e] 저장 -> {removed_csv} ({len(removed_df)}행)")

    write_summary_md(common, gate_a_lines, gate_d_lines, removed_lines)
    print("\n[e2e] 완료.")


if __name__ == "__main__":
    main()
