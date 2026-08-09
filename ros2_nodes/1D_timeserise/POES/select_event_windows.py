"""
select_event_windows.py
========================
S5 리플레이 구간 선정. 전 구간(236,848틱)은 1800배로도 33시간이라 리소스 실험에
못 쓴다 -- 손라벨 이벤트 중 3개를 골라 [onset-10일, onset+7일] 구간(앞 7일은 배경
워밍업, 나머지가 측정 대상)으로 대체한다. 읽기 전용, C_PD 아래 아무것도 쓰지 않는다.

선정 기준(결과 JSON에 그대로 기록 -- "실제 어떤 이벤트를 골랐는지" 추적용):
  strong  : peak_count 최댓값인 완결 이벤트(가장 강한 SPE).
  weak    : peak_count 최솟값인 완결 이벤트(오프라인 FSM 이 검출은 하되 약하게
            잡힌 경계 사례 -- 게이트가 자주 열고닫히는 쪽 대표).
  cluster : 고정 [t-10일, t+7일] 창에 다른 완결 이벤트가 가장 많이 들어오는
            기준 시각 t(=어떤 이벤트의 onset_time). 처음엔 2_build_dataset.
            build_episodes() 의 무제한 연쇄 병합(겹치면 계속 합침)을 썼는데,
            실제로 돌려보니 7개 이벤트가 연쇄 병합돼 102일짜리 창이 나와
            "구간당 약 1,630틱" 예산을 6배 넘겼다(리소스 실험 불가) -- 그래서
            strong/weak 와 같은 고정 17일 폭으로 바꿨다. build_episodes 자체는
            무제한 병합이 맞는 동작이라 그대로 두고(다른 용도에 재사용 가능),
            여기서는 고정폭 밀도 스캔을 새로 쓴다(기존 검출 알고리즘의 재구현이
            아니라 이 스크립트 전용 선정 휴리스틱).

재사용(재구현 없음): 1_check_labels.py(load_manual_labels/report_completeness/
build_reconciled_events), 2_build_dataset.py(build_event_spans).
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import pandas as pd


def _default_repo() -> Path:
    if os.name == "nt":
        return Path("D:/VS_code/Pixhawk_ReID")
    return Path.home() / "jeongin" / "Pixhawk_ReID"


REPO = Path(os.environ.get("REPO", _default_repo()))
C_PD = REPO / "Experiment_window" / "C_PD"

sys.path.insert(0, str(C_PD / "predict_v0"))
_check = importlib.import_module("1_check_labels")     # 숫자 접두 모듈 -> importlib
_dataset = importlib.import_module("2_build_dataset")

PAD_BEFORE_DAYS = 10   # md 지시: 이벤트-10일
PAD_AFTER_DAYS = 7     # md 지시: 이벤트+7일 (2_build_dataset 기본 대칭 10/10 과 다름, S5 전용)

OUT_PATH = Path(__file__).resolve().parent / "event_windows.json"


def _iso(ts) -> str:
    return pd.Timestamp(ts).isoformat()


def _date(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def select_windows(detector: str = "metop03", primary: str = "omni_p6") -> dict:
    labels_path = C_PD / "predict_v0" / "manual_labels" / f"manual_labels_{detector}_{primary}.csv"
    df = _check.load_manual_labels(labels_path)
    comp = _check.report_completeness(df)
    events = _check.build_reconciled_events(df, comp["split_pairs"])
    peak_lt2_ids = set(events.loc[events["peak_count"] < 2, "event_id"])
    spans = _dataset.build_event_spans(events, peak_lt2_ids)

    complete = events.dropna(subset=["onset_time", "peak_time", "end_time"]).copy()
    if complete.empty:
        raise SystemExit("[select_event_windows] 완결 이벤트가 없음 -- manual_labels 확인 필요")

    # ── strong: peak_count 최댓값 ──
    row_strong = complete.loc[complete["peak_count"].idxmax()]
    strong = {
        "event_id": int(row_strong["event_id"]),
        "onset_time": _iso(row_strong["onset_time"]),
        "peak_count": float(row_strong["peak_count"]),
        "start": _date(pd.Timestamp(row_strong["onset_time"]) - pd.Timedelta(days=PAD_BEFORE_DAYS)),
        "end": _date(pd.Timestamp(row_strong["onset_time"]) + pd.Timedelta(days=PAD_AFTER_DAYS)),
        "criterion": "peak_count 최댓값(완결 이벤트 중 가장 강한 SPE)",
    }

    # ── weak: peak_count 최솟값 ──
    row_weak = complete.loc[complete["peak_count"].idxmin()]
    weak = {
        "event_id": int(row_weak["event_id"]),
        "onset_time": _iso(row_weak["onset_time"]),
        "peak_count": float(row_weak["peak_count"]),
        "start": _date(pd.Timestamp(row_weak["onset_time"]) - pd.Timedelta(days=PAD_BEFORE_DAYS)),
        "end": _date(pd.Timestamp(row_weak["onset_time"]) + pd.Timedelta(days=PAD_AFTER_DAYS)),
        "criterion": "peak_count 최솟값(완결 이벤트 중 가장 약한 SPE, 경계 검출 사례)",
    }

    # ── cluster: 고정 [t-10일, t+7일] 창에 다른 완결 이벤트가 가장 많이 들어오는 t ──
    onsets = pd.DatetimeIndex(complete["onset_time"])
    counts = []
    for t in onsets:
        lo = t - pd.Timedelta(days=PAD_BEFORE_DAYS)
        hi = t + pd.Timedelta(days=PAD_AFTER_DAYS)
        counts.append(int(((onsets >= lo) & (onsets <= hi)).sum()))
    counts = pd.Series(counts, index=complete.index)
    best_idx = counts.idxmax()
    row_cluster = complete.loc[best_idx]
    lo = pd.Timestamp(row_cluster["onset_time"]) - pd.Timedelta(days=PAD_BEFORE_DAYS)
    hi = pd.Timestamp(row_cluster["onset_time"]) + pd.Timedelta(days=PAD_AFTER_DAYS)
    member_ids = sorted(int(e) for e in complete.loc[(onsets >= lo) & (onsets <= hi), "event_id"])
    cluster = {
        "event_id": int(row_cluster["event_id"]),
        "onset_time": _iso(row_cluster["onset_time"]),
        "n_events_in_window": int(counts.loc[best_idx]),
        "member_event_ids": member_ids,
        "start": _date(lo), "end": _date(hi),
        "criterion": f"고정 [onset-{PAD_BEFORE_DAYS}d, onset+{PAD_AFTER_DAYS}d] 창에 완결 이벤트가 "
                     f"{int(counts.loc[best_idx])}개(자기 포함) 들어오는 최다밀집 기준 이벤트",
    }

    result = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "detector": detector, "primary_channel": primary,
        "pad_before_days": PAD_BEFORE_DAYS, "pad_after_days": PAD_AFTER_DAYS,
        "windows": {"strong": strong, "weak": weak, "cluster": cluster},
    }
    return result


def main():
    result = select_windows()
    OUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[select_event_windows] 저장 -> {OUT_PATH}")
    for name, w in result["windows"].items():
        print(f"  {name}: {w['start']} ~ {w['end']}  ({w['criterion']})")


if __name__ == "__main__":
    main()
