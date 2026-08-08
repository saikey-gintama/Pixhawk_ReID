"""
5_forecast_summary.py
=======================
3_train_experiment.py --forecast-min 스윕(0/15/30/60...)으로 나온 여러
runs/<exp-name>/ 의 metrics.csv(mean 행)와 run_config.json(forecast_min)을
모아 Δt별 macro-F1 + 클래스별 precision/recall/f1 표 하나로 취합한다.
"Δt vs 성능" trade-off를 사람이 그래프로 보기 위한 취합 도구 -- 학습은 안 함,
이미 끝난 실험들의 산출물만 읽는다.

전체 평균 macro-F1(위 metrics.csv 기반 컬럼들)은 now_label==forecast_label인
윈도우가 99%대라 거의 안 바뀐 라벨에 희석돼 forecast 능력을 잘 못 보여준다
(6_forecast_shift_diag.py에서 Δt<=60분의 라벨 변경 비율이 0.1~0.4%뿐임을 확인).
그래서 oof_predictions.parquet(now_label, forecast_label, y_pred, proba_*)가
있는 run에 한해 now_label != forecast_label인 "전환 구간"만 따로 뽑아
recall/precision/f1을 transition_* 컬럼으로 추가한다 -- 실제 forecast 능력
판정은 이 transition_* 쪽을 봐야 한다. oof_predictions.parquet이 없는 run
(--forecast-min 도입 이전에 돌렸거나 Δt=0이라 애초에 전환 구간이 없는 run)은
transition_* 컬럼이 NaN으로 남는다(0으로 채우지 않음 -- "측정 안 됨"과
"전환 구간 자체가 0개"를 구분하기 위해).

재사용 (재구현 없음 -- import만): 없음(순수 집계, 이미 저장된 CSV/JSON/parquet만 읽음).

사용:
  python 5_forecast_summary.py --runs omni_p6_binary,omni_p6_binary_fc15,omni_p6_binary_fc30,omni_p6_binary_fc60
  (각 이름은 predict_v0/runs/ 아래 실제 폴더명과 일치해야 함 -- 3_train_experiment.py
  실행 시 --exp-name을 생략했다면 자동생성 규칙: {channels}_{binary|3class}[_fc{Δt}])
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

HERE = Path(__file__).resolve().parent


def load_transition_metrics(run_dir: Path, label_names: list[str]) -> dict:
    """oof_predictions.parquet에서 now_label != forecast_label인 전환 구간만 뽑아
    recall/precision/f1(macro 포함)을 계산. 파일이 없으면(이 기능 이전 run) 빈 dict --
    호출부에서 컬럼 자체가 NaN으로 남아 "측정 안 됨"이 "전환 구간 0개"와 구분됨."""
    oof_path = run_dir / "oof_predictions.parquet"
    if not oof_path.exists():
        return {}
    oof = pd.read_parquet(oof_path)
    sub = oof.loc[oof["now_label"] != oof["forecast_label"]]
    n_classes = len(label_names)
    out = {"n_transition_windows": int(len(sub))}
    if len(sub) == 0:
        return out   # Δt=0(nowcast)는 정의상 전환 구간이 항상 0개
    p, r, f1, support = precision_recall_fscore_support(
        sub["forecast_label"], sub["y_pred"], labels=list(range(n_classes)), zero_division=0)
    out["transition_macro_f1"] = f1_score(sub["forecast_label"], sub["y_pred"],
                                          average="macro", zero_division=0)
    for k, name in enumerate(label_names):
        out[f"transition_precision_{name}"] = p[k]
        out[f"transition_recall_{name}"] = r[k]
        out[f"transition_f1_{name}"] = f1[k]
        out[f"transition_support_{name}"] = int(support[k])
    return out


def load_run(run_dir: Path) -> dict:
    metrics_path = run_dir / "metrics.csv"
    config_path = run_dir / "run_config.json"
    if not metrics_path.exists() or not config_path.exists():
        raise SystemExit(f"[5_forecast_summary] {run_dir} 에 metrics.csv/run_config.json이 없습니다 "
                         f"-- 3_train_experiment.py를 이 exp-name으로 먼저 돌렸는지 확인하세요.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metrics = pd.read_csv(metrics_path)
    mean_row = metrics.loc[metrics["fold"] == "mean"].iloc[0].to_dict()
    # forecast_min은 --forecast-min 인자 도입(이진 forecast 기능) 이전에 돌린 run에는
    # run_config.json에 키 자체가 없다 -- 그 시절 실험은 전부 nowcast였으므로 0으로 간주.
    row = {"exp_name": run_dir.name, "forecast_min": config.get("forecast_min", 0),
          "label_names": config["label_names"], "n_windows": config["n_windows"],
          **mean_row}
    row.update(load_transition_metrics(run_dir, config["label_names"]))
    return row


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
        raise SystemExit(f"[5_forecast_summary] 취합 대상 run들의 클래스 구성이 서로 다릅니다: "
                         f"{label_names_set} -- 같은 --binary 여부/클래스 수끼리만 비교하세요.")

    for r in rows:
        r.pop("label_names")
    df = pd.DataFrame(rows).sort_values("forecast_min").reset_index(drop=True)
    front = ["forecast_min", "exp_name", "n_windows", "fold", "macro_f1",
            "n_transition_windows", "transition_macro_f1"]
    cols = [c for c in front if c in df.columns] + [c for c in df.columns if c not in front]
    df = df[cols]

    missing_oof = [r["exp_name"] for r in df.to_dict("records")
                  if pd.isna(r.get("n_transition_windows"))]
    if missing_oof:
        print(f"[5_forecast_summary] 참고: {missing_oof}는 oof_predictions.parquet이 없어 "
              f"transition_* 지표를 못 냄(--forecast-min 도입 이전에 돌린 run -- 재실행해야 나옴).\n")

    out_path = Path(args.out) if args.out else HERE / "runs" / "forecast_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[5_forecast_summary] Δt별 성능 취합 ({len(df)}개 실험):")
    print(df.to_string(index=False))
    print(f"\n[5_forecast_summary] 저장 -> {out_path}")


if __name__ == "__main__":
    main()
