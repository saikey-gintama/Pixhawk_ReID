"""
evaluate.py  —  BL-C2 지상 검증 전용 평가 스크립트
=====================================================
ksem_bl_c2.py 에서 분리된 ground truth 의존 지표를 계산한다.

[ 위성 탑재 불필요 이유 ]
  - FAR / Detection Rate : known_events(ground truth) 필요 → 지상 전용
  - Lead Time            : ground truth onset 기준 계산 → 지상 전용
  - 데이터 절감률        : duration_* 결과를 입력으로 사용하는 후처리 지표

[ 입력 ]
  ksem_bl_c2.py 가 출력하는 두 파일:
    1. bl_c2_states_{tag}.csv    — FSM 상태 시계열 (voted, PD* 컬럼 포함)
    2. bl_c2_metrics_{tag}.json  — duration_* 등 알고리즘 지표

[ 출력 ]
  evaluate_{tag}.json  — FAR / Detection Rate / Lead Time / 데이터 절감률

지표 정의:
  FAR (논문 Eq. 1):
    FAR = false_trigger / (false_trigger + correct_trigger)
    트리거(PRE_ALERT onset)가 onset 이전 30분 ~ event end 사이이면 correct.

  Detection Rate:
    detected_gt_events / total_gt_events

  Lead Time (논문 Figure 3 right):
    PRE_ALERT onset → event_cumsum 90% 누적 시각까지 남은 시간(분).
    PD1 event_cumsum 컬럼 사용.

  데이터 절감률:
    ALERT+PRE_ALERT 구간만 전송 시 줄어드는 포인트 비율.
    = (duration_NOMINAL_pts) / total_pts × 100 (%)
    NOMINAL 구간 포인트는 전송 불필요 → 절감 대상.

사용법:
  # JSON ground truth 파일로 평가
  python evaluate.py \\
      --states  ./output_bl_c2/bl_c2_states_201905_202412.csv \\
      --metrics ./output_bl_c2/bl_c2_metrics_201905_202412.json \\
      --events  known_events.json \\
      --output  ./output_bl_c2/evaluate_201905_202412.json

  # known_events 인라인 입력 (빠른 테스트)
  python evaluate.py \\
      --states  ./output_bl_c2/bl_c2_states_20240510.csv \\
      --metrics ./output_bl_c2/bl_c2_metrics_20240510.json \\
      --event_onset "2024-05-10 01:30" --event_end "2024-05-10 09:30"

known_events JSON 형식:
  [
    {"onset": "2024-05-10 01:30", "end": "2024-05-10 09:30"},
    {"onset": "2024-06-02 14:00", "end": "2024-06-03 02:00"}
  ]
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── 논문 고정값 (ksem_bl_c2.py 와 동일하게 유지) ──────────────────────────
RESAMPLE_MIN         = 15   # 15분 cadence
TRIGGER_MATCH_MARGIN = 30   # PRE_ALERT onset이 gt onset 이전 N분까지 correct로 인정


# ============================================================
# 1.  입력 로더
# ============================================================

def load_states(states_csv: Path) -> pd.DataFrame:
    """bl_c2_states_{tag}.csv → DataFrame (index=Timestamp)."""
    df = pd.read_csv(states_csv, index_col=0, parse_dates=True)
    df.index.name = "Time"
    return df


def load_metrics(metrics_json: Path) -> dict:
    """bl_c2_metrics_{tag}.json → dict."""
    with open(metrics_json, encoding="utf-8") as f:
        return json.load(f)


def load_known_events(events_json: Path) -> List[dict]:
    """
    known_events JSON 파일 로드.

    형식:
      [{"onset": "YYYY-MM-DD HH:MM", "end": "YYYY-MM-DD HH:MM"}, ...]
    """
    with open(events_json, encoding="utf-8") as f:
        raw = json.load(f)
    for ev in raw:
        if "onset" not in ev or "end" not in ev:
            raise ValueError(f"이벤트 항목에 'onset' / 'end' 키 필요: {ev}")
    return raw


# ============================================================
# 2.  트리거 이벤트 추출  (ksem_bl_c2.py 의 extract_trigger_events 와 동일)
# ============================================================

def extract_trigger_events(states: pd.DataFrame
                           ) -> Tuple[List[pd.Timestamp],
                                      List[pd.Timestamp],
                                      List[dict]]:
    """
    voted / voted_state 컬럼에서 트리거 onset과 이벤트 구간을 추출.

    state_name 컬럼(문자열)이 있으면 그것을 사용하고,
    없으면 state_val(int)에서 복원한다.

    Returns
    -------
    pre_alert_times : PRE_ALERT onset 시각 리스트
    alert_times     : ALERT onset 시각 리스트
    detected_events : [{"pre_alert": t, "alert": t, "end": t}, ...]
    """
    # state_name 컬럼 결정
    if "state_name" in states.columns:
        state_col = states["state_name"]
    elif "voted_state" in states.columns:
        state_col = states["voted_state"]
    else:
        # state_val(int) 에서 문자열로 복원
        val_map = {0: "NOMINAL", 1: "PRE_ALERT", 2: "ALERT"}
        state_col = states["state_val"].map(val_map)

    pre_alert_times: List[pd.Timestamp] = []
    alert_times:     List[pd.Timestamp] = []
    detected_events: List[dict]         = []

    prev      = "NOMINAL"
    cur_event: Optional[dict] = None

    for t, s in state_col.items():
        s = str(s)  # Enum repr 안전 처리

        if prev == "NOMINAL" and s == "PRE_ALERT":
            pre_alert_times.append(t)
            cur_event = {"pre_alert": t, "alert": None, "end": None}

        if prev != "ALERT" and s == "ALERT":
            alert_times.append(t)
            if cur_event is not None:
                cur_event["alert"] = t

        if prev != "NOMINAL" and s == "NOMINAL":
            if cur_event is not None:
                cur_event["end"] = t
                detected_events.append(cur_event)
                cur_event = None

        prev = s

    if cur_event is not None:
        cur_event["end"] = state_col.index[-1]
        detected_events.append(cur_event)

    return pre_alert_times, alert_times, detected_events


# ============================================================
# 3.  FAR / Detection Rate  (논문 Section 4.1, Eq. 1)
# ============================================================

def compute_far_detection(
        pre_alert_times: List[pd.Timestamp],
        known_events: List[dict],
        margin_min: int = TRIGGER_MATCH_MARGIN
) -> dict:
    """
    FAR  = false_trigger / (false_trigger + correct_trigger)
    DR   = detected_gt_events / total_gt_events

    매칭 기준:
      PRE_ALERT onset이 gt onset 이전 margin_min 분 ~ gt end 사이이면 correct.
      하나의 gt 이벤트에 여러 트리거가 매칭되면 첫 번째만 correct로 계산.
    """
    correct        = 0
    false_t        = 0
    detected_gt    = set()

    for pre_t in pre_alert_times:
        matched = False
        for i, ev in enumerate(known_events):
            onset = pd.Timestamp(ev["onset"])
            end   = pd.Timestamp(ev["end"])
            window_start = onset - pd.Timedelta(minutes=margin_min)
            if window_start <= pre_t <= end:
                if i not in detected_gt:   # 중복 correct 방지
                    correct += 1
                    detected_gt.add(i)
                matched = True
                break
        if not matched:
            false_t += 1

    denom = false_t + correct
    far   = round(false_t / denom, 4) if denom > 0 else 0.0
    dr    = round(len(detected_gt) / len(known_events), 4) if known_events else None

    return {
        "FAR":                  far,
        "Detection_Rate":       dr,
        "n_correct_triggers":   correct,
        "n_false_triggers":     false_t,
        "n_detected_gt_events": len(detected_gt),
        "n_total_gt_events":    len(known_events),
    }


# ============================================================
# 4.  Lead Time  (논문 Figure 3 right)
# ============================================================

def compute_lead_times(
        states: pd.DataFrame,
        detected_events: List[dict]
) -> dict:
    """
    Lead Time = PRE_ALERT onset → event_cumsum 90% 누적 시각까지 걸린 시간(분).

    event_cumsum 컬럼은 bl_c2_states CSV에서 PD1_* 컬럼군을 사용한다.
    PD1_value 컬럼이 있을 경우 background 제거 후 직접 누적 계산도 가능하나,
    FSM 내부에서 이미 계산된 event_cumsum을 사용하는 것이 논문 정의에 부합.

    event_cumsum 컬럼이 없으면 PD1_value - PD1_background로 대체 계산.
    """
    # event_cumsum 열 특정
    cumsum_col = None
    if "PD1_event_cumsum" in states.columns:
        cumsum_col = "PD1_event_cumsum"
    elif "event_cumsum" in states.columns:
        cumsum_col = "event_cumsum"
    else:
        # PD1_value - PD1_background 로 누적 대체
        if "PD1_value" in states.columns and "PD1_background" in states.columns:
            excess = (states["PD1_value"] - states["PD1_background"]).clip(lower=0)
            states = states.copy()
            states["_fallback_cumsum"] = excess.cumsum()
            cumsum_col = "_fallback_cumsum"

    lead_times: List[float] = []

    if cumsum_col is None:
        print("  [WARN] event_cumsum 컬럼 없음 → Lead Time 계산 불가")
    else:
        for ev in detected_events:
            pre_t = ev["pre_alert"]
            end_t = ev["end"]
            if end_t is None:
                continue

            segment = states.loc[pre_t:end_t, cumsum_col]
            if len(segment) < 2:
                continue

            # 누적값을 구간 내 상대 누적으로 환산
            seg_vals = segment.values.astype(float)
            seg_min  = seg_vals[0]
            seg_max  = seg_vals[-1]
            total_dose = seg_max - seg_min
            if total_dose <= 0:
                continue

            thresh_90 = seg_min + total_dose * 0.90
            idx_90    = segment[segment >= thresh_90].index
            if len(idx_90) == 0:
                continue

            t_90     = idx_90[0]
            lead_min = (t_90 - pre_t).total_seconds() / 60.0
            lead_times.append(lead_min)

    if lead_times:
        return {
            "lead_time_min_avg": round(float(np.mean(lead_times)), 1),
            "lead_time_min_min": round(float(np.min(lead_times)),  1),
            "lead_time_min_max": round(float(np.max(lead_times)),  1),
            "n_lead_time_events": len(lead_times),
        }
    else:
        return {
            "lead_time_min_avg":  None,
            "lead_time_min_min":  None,
            "lead_time_min_max":  None,
            "n_lead_time_events": 0,
        }


# ============================================================
# 5.  데이터 절감률
# ============================================================

def compute_data_reduction(alg_metrics: dict) -> dict:
    """
    NOMINAL 구간은 이벤트가 없으므로 전송 불필요 → 절감 대상.
    ALERT + PRE_ALERT 구간만 전송 시의 절감률을 계산한다.

    입력: ksem_bl_c2.py 가 출력한 metrics dict (duration_* 키 포함)

    reduction_pct = duration_NOMINAL_pts / total_pts × 100
    """
    nom_pts  = alg_metrics.get("duration_NOMINAL_pts",   0)
    pre_pts  = alg_metrics.get("duration_PRE_ALERT_pts", 0)
    alt_pts  = alg_metrics.get("duration_ALERT_pts",     0)
    total    = nom_pts + pre_pts + alt_pts

    if total == 0:
        return {
            "data_reduction_pct":       None,
            "transmitted_pts":          None,
            "total_pts":                0,
            "nominal_pts_saved":        0,
        }

    transmitted = pre_pts + alt_pts
    reduction   = round(nom_pts / total * 100, 2)

    return {
        "data_reduction_pct":   reduction,
        "transmitted_pts":      int(transmitted),
        "nominal_pts_saved":    int(nom_pts),
        "total_pts":            int(total),
        "transmitted_min":      int(transmitted * RESAMPLE_MIN),
        "nominal_saved_min":    int(nom_pts     * RESAMPLE_MIN),
    }


# ============================================================
# 6.  결과 출력 / 저장
# ============================================================

def print_results(results: dict):
    """콘솔 출력."""
    w = 40
    print("\n" + "=" * (w + 20))
    print("  evaluate.py  —  BL-C2 지상 검증 결과")
    print("=" * (w + 20))

    sections = {
        "[ FAR / Detection Rate ]": [
            "FAR", "Detection_Rate",
            "n_correct_triggers", "n_false_triggers",
            "n_detected_gt_events", "n_total_gt_events",
        ],
        "[ Lead Time ]": [
            "lead_time_min_avg", "lead_time_min_min",
            "lead_time_min_max", "n_lead_time_events",
        ],
        "[ 데이터 절감률 ]": [
            "data_reduction_pct",
            "transmitted_pts", "transmitted_min",
            "nominal_pts_saved", "nominal_saved_min",
            "total_pts",
        ],
    }

    for title, keys in sections.items():
        print(f"\n{title}")
        print("─" * (w + 4))
        for k in keys:
            v = results.get(k, "—")
            if isinstance(v, float):
                print(f"  {k:<{w}}: {v:.4f}" if "pct" not in k and "Rate" not in k and "FAR" not in k
                      else f"  {k:<{w}}: {v:.4f}")
            else:
                print(f"  {k:<{w}}: {v}")

    print("\n" + "=" * (w + 20))


def save_results(results: dict, output_path: Path):
    """JSON 저장."""
    clean = {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
             for k, v in results.items()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[Output] {output_path}")


# ============================================================
# 7.  MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="BL-C2 지상 검증 평가 스크립트 (FAR / Lead Time / 데이터 절감률)")

    parser.add_argument(
        "--states",  required=True,
        help="bl_c2_states_{tag}.csv 경로 (ksem_bl_c2.py 출력)")
    parser.add_argument(
        "--metrics", required=True,
        help="bl_c2_metrics_{tag}.json 경로 (ksem_bl_c2.py 출력)")

    # ground truth 입력: 파일 또는 인라인
    gt_group = parser.add_mutually_exclusive_group(required=True)
    gt_group.add_argument(
        "--events",
        help="ground truth JSON 파일 경로 "
             '(형식: [{"onset": "YYYY-MM-DD HH:MM", "end": "YYYY-MM-DD HH:MM"}, ...])')
    gt_group.add_argument(
        "--event_onset", metavar="ONSET",
        help="단일 이벤트 onset (--event_end 와 함께 사용): YYYY-MM-DD HH:MM")

    parser.add_argument(
        "--event_end", metavar="END",
        help="단일 이벤트 end (--event_onset 과 함께 사용): YYYY-MM-DD HH:MM")
    parser.add_argument(
        "--output",  default=None,
        help="결과 JSON 저장 경로 (기본: states CSV 와 같은 폴더에 evaluate_{tag}.json)")
    parser.add_argument(
        "--margin_min", type=int, default=TRIGGER_MATCH_MARGIN,
        help=f"FAR 매칭 허용 마진(분) (기본: {TRIGGER_MATCH_MARGIN}분)")

    args = parser.parse_args()

    # ── 인자 검증 ──
    if args.event_onset and not args.event_end:
        parser.error("--event_onset 을 사용할 때는 --event_end 도 필요합니다.")
    if args.event_end and not args.event_onset:
        parser.error("--event_end 를 사용할 때는 --event_onset 도 필요합니다.")

    # ── 파일 로드 ──
    states_path  = Path(args.states)
    metrics_path = Path(args.metrics)

    print(f"[Load] states  : {states_path}")
    print(f"[Load] metrics : {metrics_path}")

    states      = load_states(states_path)
    alg_metrics = load_metrics(metrics_path)

    # ── ground truth ──
    if args.events:
        known_events = load_known_events(Path(args.events))
        print(f"[Load] events  : {args.events}  ({len(known_events)}건)")
    else:
        known_events = [{"onset": args.event_onset, "end": args.event_end}]
        print(f"[Load] events  : 인라인 1건  "
              f"({args.event_onset} ~ {args.event_end})")

    # ── 트리거 추출 ──
    pre_times, alert_times, detected = extract_trigger_events(states)
    print(f"\n[Triggers] PRE-ALERT: {len(pre_times)}건  "
          f"ALERT: {len(alert_times)}건  "
          f"이벤트 구간: {len(detected)}건")

    # ── 지표 계산 ──
    far_result      = compute_far_detection(pre_times, known_events, args.margin_min)
    lead_result     = compute_lead_times(states, detected)
    reduction_result = compute_data_reduction(alg_metrics)

    results = {
        "source_states":  str(states_path),
        "source_metrics": str(metrics_path),
        "n_gt_events":    len(known_events),
        "match_margin_min": args.margin_min,
        **far_result,
        **lead_result,
        **reduction_result,
    }

    # ── 출력 ──
    print_results(results)

    # ── 저장 ──
    if args.output:
        out_path = Path(args.output)
    else:
        tag      = states_path.stem.replace("bl_c2_states_", "")
        out_path = states_path.parent / f"evaluate_{tag}.json"

    save_results(results, out_path)


if __name__ == "__main__":
    main()
