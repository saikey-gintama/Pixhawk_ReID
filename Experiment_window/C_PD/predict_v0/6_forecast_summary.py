"""
6_forecast_summary.py
=======================
train_experiment.py --forecast-min 스윕(0/15/30/60...)으로 나온 여러
runs/<exp-name>/ 의 metrics.csv(mean 행)와 run_config.json(forecast_min)을
모아 Δt별 macro-F1 + 클래스별 precision/recall/f1 표 하나로 취합한다.
"Δt vs 성능" trade-off를 사람이 그래프로 보기 위한 취합 도구 -- 학습은 안 함,
이미 끝난 실험들의 산출물만 읽는다.

재사용 (재구현 없음 -- import만): 없음(순수 집계, 이미 저장된 CSV/JSON만 읽음).

사용:
  python 6_forecast_summary.py --runs omni_p6_binary,omni_p6_binary_fc15,omni_p6_binary_fc30,omni_p6_binary_fc60
  (각 이름은 predict_v0/runs/ 아래 실제 폴더명과 일치해야 함 -- train_experiment.py
  실행 시 --exp-name을 생략했다면 자동생성 규칙: {channels}_{binary|3class}[_fc{Δt}])
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def load_run(run_dir: Path) -> dict:
    metrics_path = run_dir / "metrics.csv"
    config_path = run_dir / "run_config.json"
    if not metrics_path.exists() or not config_path.exists():
        raise SystemExit(f"[forecast_summary] {run_dir} 에 metrics.csv/run_config.json이 없습니다 "
                         f"-- train_experiment.py를 이 exp-name으로 먼저 돌렸는지 확인하세요.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metrics = pd.read_csv(metrics_path)
    mean_row = metrics.loc[metrics["fold"] == "mean"].iloc[0].to_dict()
    return {"exp_name": run_dir.name, "forecast_min": config["forecast_min"],
            "label_names": config["label_names"], "n_windows": config["n_windows"],
            **mean_row}


def main():
    ap = argparse.ArgumentParser(description="forecast Δt 스윕 metrics 취합 -- Δt vs 성능 표")
    ap.add_argument("--runs", required=True,
                    help="콤마구분 exp-name 목록(predict_v0/runs/ 아래 실제 폴더명)")
    ap.add_argument("--out", default=None, help="기본 predict_v0/runs/forecast_summary.csv")
    args = ap.parse_args()

    run_names = [r.strip() for r in args.runs.split(",") if r.strip()]
    rows = [load_run(HERE / "runs" / name) for name in run_names]

    label_names_set = {tuple(r["label_names"]) for r in rows}
    if len(label_names_set) > 1:
        raise SystemExit(f"[forecast_summary] 취합 대상 run들의 클래스 구성이 서로 다릅니다: "
                         f"{label_names_set} -- 같은 --binary 여부/클래스 수끼리만 비교하세요.")

    for r in rows:
        r.pop("label_names")
    df = pd.DataFrame(rows).sort_values("forecast_min").reset_index(drop=True)
    cols = ["forecast_min", "exp_name", "n_windows", "fold", "macro_f1"] + \
        [c for c in df.columns if c not in ("forecast_min", "exp_name", "n_windows", "fold", "macro_f1")]
    df = df[cols]

    out_path = Path(args.out) if args.out else HERE / "runs" / "forecast_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"[forecast_summary] Δt별 성능 취합 ({len(df)}개 실험):")
    print(df.to_string(index=False))
    print(f"\n[forecast_summary] 저장 -> {out_path}")


if __name__ == "__main__":
    main()
