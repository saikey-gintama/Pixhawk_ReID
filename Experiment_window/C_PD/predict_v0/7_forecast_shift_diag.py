"""
7_forecast_shift_diag.py
==========================
Δt(--forecast-min) 시프트가 라벨을 실제로 얼마나 바꿨는지 정량화하는 순수 데이터셋
진단(학습 없음). train_experiment.py의 nowcast macro-F1(0.847)과 forecast Δt=15/30/60
macro-F1(0.850/0.851/0.849)이 사실상 평평했던 이유가 (a) 이벤트가 며칠 단위로 길어
Δt<=60분 시프트로는 라벨이 거의 안 바뀌어 사실상 같은 문제를 푼 것인지, (b) 시프트
로직 자체가 잘못돼 실제로 미래 라벨이 아닌지를 판별한다.

재사용 (재구현 없음 -- import만):
  train_experiment.py : load_windows(단일채널 nowcast windows 그대로 로드).
      "train_experiment"는 숫자로 시작하지 않아 일반 import로 로드 가능.
  timeseries_{detector}_{channel}.parquet : t+Δt 시점 라벨 조회(train_experiment.py의
      apply_forecast_shift와 동일한 reindex 방식 -- 여기서는 now/future 라벨을 둘 다
      남겨야 해서 그 로직만 인라인으로 재현, 필터링(keep) 기준은 동일).

한계(중요): 이 스크립트는 라벨 시프트가 옳은지만 검사한다(1,2번). 3번(전환
구간에서 모델이 실제로 얼마나 맞혔는지)은 각 run 폴더에 per-window OOF 예측이나
모델 체크포인트가 저장돼 있지 않아(metrics.csv는 fold 집계치, confusion_matrix.png는
이미지일 뿐) 학습·추론을 다시 돌리지 않고는 계산할 수 없다 -- "학습 재실행 금지"
지시와 정면으로 충돌해서 이 스크립트는 3번을 계산하지 않는다(주석 참고).

사용:
  python 7_forecast_shift_diag.py --forecast-min 15,30,60
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import train_experiment as te  # 숫자로 시작하지 않아 일반 import 가능

HERE = Path(__file__).resolve().parent


def diag_one(windows0: pd.DataFrame, ts: pd.DataFrame, forecast_min: int) -> dict:
    future_time = windows0["window_end_time"] + pd.Timedelta(minutes=forecast_min)
    future_label_raw = ts["label"].reindex(future_time).values   # NaN = t+Δt 라벨 없음
    now_label_raw = windows0["label"].values                      # 3클래스(0/1/2) 원본
    keep = ~pd.isna(future_label_raw)                              # apply_forecast_shift와 동일 필터

    now_bin = (now_label_raw[keep] > 0).astype(int)                # --binary와 동일 규칙
    fut_bin = (future_label_raw[keep] > 0).astype(int)

    n_total = int(keep.sum())
    changed = now_bin != fut_bin
    n_changed = int(changed.sum())
    n_q2e = int(((now_bin == 0) & (fut_bin == 1)).sum())   # quiet -> event
    n_e2q = int(((now_bin == 1) & (fut_bin == 0)).sum())   # event -> quiet

    return {
        "forecast_min": forecast_min,
        "n_windows": n_total,
        "n_excluded_no_future_label": int((~keep).sum()),
        "n_changed": n_changed,
        "pct_changed": n_changed / n_total * 100,
        "n_quiet_to_event": n_q2e,
        "n_event_to_quiet": n_e2q,
        "n_unchanged": n_total - n_changed,
    }


def main():
    ap = argparse.ArgumentParser(description="forecast Δt 라벨 시프트 정량화(학습 없음, 데이터셋만)")
    ap.add_argument("--detector", default="metop03")
    ap.add_argument("--channel", default="omni_p6")
    ap.add_argument("--forecast-min", default="15,30,60", help="콤마구분 Δt(분) 목록")
    args = ap.parse_args()

    windows0 = te.load_windows(args.detector, [args.channel])  # nowcast 원본(3클래스 라벨, 미시프트)
    ts = pd.read_parquet(HERE / "dataset_v0" / f"timeseries_{args.detector}_{args.channel}.parquet")
    ts = ts.set_index("time")

    forecast_mins = [int(x) for x in args.forecast_min.split(",") if x.strip()]
    rows = [diag_one(windows0, ts, dt) for dt in forecast_mins]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    print("\n[7_forecast_shift_diag] 한계: 3번(전환 구간에서 모델 recall/precision)은 "
          "per-window OOF 예측/체크포인트가 저장돼 있지 않아 계산 불가(학습 재실행 없이는 "
          "구할 수 없음) -- 위 표(1,2번)만으로 (a)/(b) 판정.")

    out_path = HERE / "runs" / "forecast_label_shift_diag.csv"
    df.to_csv(out_path, index=False)
    print(f"[7_forecast_shift_diag] 저장 -> {out_path}")


if __name__ == "__main__":
    main()
