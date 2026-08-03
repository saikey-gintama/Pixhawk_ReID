"""
train_experiment.py
=====================
predict_v0 v0 실험 3종(단일채널 3클래스/단일채널 이진/다채널 3클래스)을 인자 조합
하나로 표현하는 통합 러너. Step 1(3_train.py)/Step 2(4_eval.py)/Step 3(5_diag.py)의
TCN 학습·OOF 평가·진단 그림 코드를 그대로 재사용(재구현 없음) -- 이 스크립트는
그 위에 (a) 이진 라벨 병합, (b) 다채널 피처 스태킹, (c) epochs/patience 조기종료를
CLI 인자로 노출하는 오케스트레이션만 새로 짠다.

재사용 (재구현 없음 -- import만, 전부 숫자로 시작하는 모듈이라 importlib):
  3_train.py : load_dataset/feature_cols/train_tcn(epochs/patience/input_size/
      n_classes로 일반화됨)/LABEL_NAMES -- 5_diag.py를 통해 접근.
  5_diag.py  : run_oof(n_splits/n_classes/input_size/epochs/patience로 일반화),
      plot_learning_curves, plot_confusion_matrix(label_names로 일반화),
      plot_event_overlay(label_names/label_colors/label_transform로 일반화),
      pick_representatives, MAX_EVENT_PLOTS, LABEL_COLORS.
  2_build_dataset.py : --channels 다채널 확장판이 만든 windows_{detector}_{tag}.parquet
      (tag="-".join(channels)) -- 다채널 실험(--channels 2개 이상)은 이 파일이 미리
      있어야 함(없으면 에러 메시지로 안내).

실험 3종 (실행 예시는 모듈 맨 아래 "사용" 참고):
  실험1 단일채널 3클래스 : --channels omni_p6
  실험2 단일채널 이진     : --channels omni_p6 --binary
      (--binary는 데이터셋 재생성 없이 라벨만 여기서 병합: label = (label>0).astype(int)
      quiet=0, event(rising+decreasing)=1 -- 실험1과 완전히 같은 windows.parquet을
      공유해야 입력이 동일해 비교가 깨끗하다는 요구사항 그대로.)
  실험3 다채널 3클래스   : --channels omni_p6,omni_p7,pro_tel0_p5
      (TCN input_size=len(channels). 각 채널의 z_ch{i}_lag* 컬럼을 쌓아 (n, n_channels,
      window) 텐서를 만듦 -- tcn.py의 TCNClassifier.forward가 2D/3D 입력 모두 받도록
      이미 일반화돼 있어 추가 변경 없이 그대로 동작.)

forecast (--forecast-min, 기본 0=nowcast): 이진(--binary)이 가장 잘 되므로(nowcast
macro-F1 0.85) 이걸 미래로 밀어 "지금 quiet여도 Δt분 뒤 event가 시작되는가"를 미리
예측할 수 있는지 본다. 시각 t의 입력(z_lag..)으로 t+Δt 시점의 라벨을 예측 -- 라벨만
시프트하고 입력 윈도우·모델·episode 분할·GroupKFold는 전부 그대로(스펙 그대로).
구현: windows['label']을 t 시점 값 대신 timeseries_*.parquet에서 조회한 t+Δt 시점
값으로 치환(apply_forecast_shift). t+Δt가 timeseries 인덱스에 없는 행(각 episode
크롭의 맨 끝 Δt 구간, 또는 우연히 실데이터 공백과 겹치는 경우 전부 포함)은 그
윈도우째로 제외 -- reindex가 없는 시각에 자동으로 NaN을 주므로 그 자체가 배제
조건이 된다. Δt는 15분 격자이므로 15의 배수만 허용. 3클래스 forecast도 기술적으로
되지만(라벨 시프트가 클래스 수와 무관) 이번 목적은 이진.

출력 (predict_v0/runs/<exp-name>/, 기존 실험 폴더 보호를 위해 이미 있으면 --force
없이는 중단 -- 4_summarize.py의 --force 관례와 동일):
  learning_curves.png, confusion_matrix.png, events/event{id}_*.png(최대 100장),
  metrics.csv(fold별+mean/std, 클래스별 P/R/F1 + macro-F1 + best/stopped epoch),
  run_config.json(실행 인자 전부 -- 재현용)
  exp-name 자동생성 시 forecast-min>0이면 "_fc{Δt}" 접미사가 붙어 nowcast 결과와
  안 겹침(예: omni_p6_binary_fc15).

사용:
  # 실험1: 단일채널 3클래스
  python train_experiment.py --channels omni_p6 --epochs 100 --patience 15
  # 실험2: 단일채널 이진 (nowcast, Δt=0 기준선)
  python train_experiment.py --channels omni_p6 --binary --epochs 100 --patience 15
  # 실험3: 다채널 3클래스 (먼저 다채널 데이터셋 생성 필요)
  python 2_build_dataset.py --channels omni_p6,omni_p7,pro_tel0_p5
  python train_experiment.py --channels omni_p6,omni_p7,pro_tel0_p5 --epochs 100 --patience 15
  # 이진 forecast Δt 스윕 (15/30/60분)
  python train_experiment.py --channels omni_p6 --binary --forecast-min 15 --epochs 100 --patience 15
  python train_experiment.py --channels omni_p6 --binary --forecast-min 30 --epochs 100 --patience 15
  python train_experiment.py --channels omni_p6 --binary --forecast-min 60 --epochs 100 --patience 15
  # 스모크 테스트(긴 학습 없이 파이프라인만 확인)
  python train_experiment.py --channels omni_p6 --epochs 2 --folds 1 --exp-name smoke1 --force
  python train_experiment.py --channels omni_p6 --binary --forecast-min 15 --epochs 2 --folds 1 --exp-name smoke_fc15 --force
"""
from __future__ import annotations
import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# "5_diag"/"3_train"은 숫자로 시작해 import 문 대신 importlib로 로드(0_label_events_gui.py
# 참고). 5_diag가 내부에서 3_train을 이미 importlib로 로드해 자기 모듈 이름공간에
# 노출해 두므로(diag.feature_cols 등), 여기서는 5_diag 하나만 가져오면 충분하다.
diag = importlib.import_module("5_diag")

BINARY_LABEL_NAMES = ["quiet", "event"]
BINARY_LABEL_COLORS = {0: "#999999", 1: "tab:orange"}


def load_windows(detector: str, channels: list[str]) -> pd.DataFrame:
    if len(channels) == 1:
        return diag.load_dataset(detector, channels[0])
    tag = "-".join(channels)
    path = HERE / "dataset_v0" / f"windows_{detector}_{tag}.parquet"
    if not path.exists():
        raise SystemExit(
            f"[train_experiment] {path} 없음 -- 다채널 실험은 먼저 데이터셋을 만들어야 합니다:\n"
            f"  python 2_build_dataset.py --detector {detector} --channels {','.join(channels)}")
    return pd.read_parquet(path)


def infer_window(windows: pd.DataFrame, channels: list[str]) -> int:
    prefix = "z_lag" if len(channels) == 1 else "z_ch0_lag"
    return sum(1 for c in windows.columns if c.startswith(prefix))


def build_X(windows: pd.DataFrame, channels: list[str], window: int) -> np.ndarray:
    if len(channels) == 1:
        cols = diag.feature_cols(window)
        return windows[cols].values.astype(np.float32)          # (n, window)
    arrs = []
    for i in range(len(channels)):
        cols = [f"z_ch{i}_lag{k}" for k in range(window - 1, -1, -1)]
        arrs.append(windows[cols].values.astype(np.float32))
    return np.stack(arrs, axis=1)                                 # (n, n_channels, window)


def apply_forecast_shift(windows: pd.DataFrame, ts: pd.DataFrame, forecast_min: int) -> pd.DataFrame:
    """windows['label']을 t(window_end_time) 시점 값 대신 t+forecast_min 시점 값으로
    치환(nowcast -> forecast). event_id/episode_id/ambiguous_peak(그룹핑·진단용)는
    건드리지 않고 t 기준 그대로 유지 -- GroupKFold 분할은 입력 시각의 이벤트 소속을
    그대로 쓴다. t+Δt가 timeseries에 없는 행(각 episode 크롭 맨 끝 Δt 구간, 또는
    실데이터 공백과 겹치는 경우)은 reindex가 NaN을 주므로 그 자체로 걸러져 윈도우째
    제외된다. forecast_min<=0이면 그대로(nowcast, 무변경) 반환."""
    if forecast_min <= 0:
        return windows
    if forecast_min % 15 != 0:
        raise SystemExit(f"[train_experiment] --forecast-min은 15의 배수여야 합니다(15분 격자): {forecast_min}")

    future_time = windows["window_end_time"] + pd.Timedelta(minutes=forecast_min)
    future_label = ts["label"].reindex(future_time).values
    now_label = windows["label"].values
    keep = ~pd.isna(future_label)

    n_before = len(windows)
    out = windows.loc[keep].reset_index(drop=True).copy()
    out["label"] = future_label[keep].astype(np.int64)
    print(f"[train_experiment] forecast +{forecast_min}min 라벨 시프트: {n_before}개 -> {len(out)}개 "
          f"윈도우(미래 라벨 없음 {n_before - len(out)}개 제외)")

    sample_idx = np.where(keep)[0][:3]
    for i in sample_idx:
        t = windows["window_end_time"].iloc[i]
        print(f"    검증 예시: t={t}  now_label={int(now_label[i])}(t 시점)  ->  "
              f"forecast_label={int(future_label[i])}(t+{forecast_min}min={t + pd.Timedelta(minutes=forecast_min)} 시점)")
    return out


def build_y(windows: pd.DataFrame, binary: bool):
    y = windows["label"].values.astype(np.int64)
    if binary:
        return (y > 0).astype(np.int64), BINARY_LABEL_NAMES
    return y, diag.LABEL_NAMES


def build_metrics_table(windows: pd.DataFrame, y: np.ndarray, oof_pred: np.ndarray,
                        fold_of_episode: dict, fold_histories: list[dict],
                        label_names: list[str], n_splits: int) -> pd.DataFrame:
    groups = windows["episode_id"].values
    n_classes = len(label_names)
    rows = []
    for fold in range(n_splits):
        val_mask = np.array([fold_of_episode.get(g) == fold for g in groups])
        yt, yp = y[val_mask], oof_pred[val_mask]
        p, r, f1, support = precision_recall_fscore_support(
            yt, yp, labels=list(range(n_classes)), zero_division=0)
        macro_f1 = f1_score(yt, yp, average="macro", zero_division=0) if val_mask.any() else float("nan")
        row = {
            "fold": fold,
            "n_val_windows": int(val_mask.sum()),
            "n_val_episodes": len(set(groups[val_mask])),
            "best_epoch": fold_histories[fold]["best_epoch"],
            "stopped_epoch": fold_histories[fold]["stopped_epoch"],
            "macro_f1": macro_f1,
        }
        for k, name in enumerate(label_names):
            row[f"precision_{name}"] = p[k]
            row[f"recall_{name}"] = r[k]
            row[f"f1_{name}"] = f1[k]
            row[f"support_{name}"] = int(support[k])
        rows.append(row)
    df = pd.DataFrame(rows)
    numeric_cols = [c for c in df.columns if c != "fold"]
    mean_row = {"fold": "mean", **df[numeric_cols].mean(numeric_only=True).to_dict()}
    std_row = {"fold": "std", **df[numeric_cols].std(numeric_only=True).to_dict()}
    return pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description="predict_v0 통합 실험 러너 (실험1/2/3을 인자 조합으로)")
    ap.add_argument("--detector", default="metop03")
    ap.add_argument("--channels", default="omni_p6",
                    help="콤마구분. 1개=단일채널(기존 windows 재사용), 2개 이상=다채널 "
                         "(2_build_dataset.py --channels로 미리 생성 필요)")
    ap.add_argument("--binary", action="store_true",
                    help="quiet=0, event(rising+decreasing)=1로 라벨 병합(데이터셋 재생성 없음)")
    ap.add_argument("--forecast-min", type=int, default=0,
                    help="0=nowcast(기본). >0이면 라벨을 t+Δt분 시점 값으로 시프트해 forecast "
                         "학습(15의 배수만 허용, --binary와 함께 쓰는 걸 기본 상정)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15,
                    help="val macro-F1 개선 없이 버티는 epoch 수(조기종료 기준). 0 이하면 비활성화")
    ap.add_argument("--folds", type=int, default=5,
                    help="GroupKFold 수. 1은 스모크 테스트 전용 특수경로(80/20 1회 분할, "
                         "OOF 부분 커버리지 -- 5_diag.run_oof 참고, 성능 판단용 아님)")
    ap.add_argument("--exp-name", default=None, help="predict_v0/runs/<exp-name>/. 생략 시 "
                    "channels+binary로 자동 생성(예: omni_p6_3class, omni_p6_binary)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None, help="기본 predict_v0/runs/<exp-name>/")
    ap.add_argument("--force", action="store_true", help="runs/<exp-name>/ 이미 있어도 덮어쓰기 허용")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    patience = args.patience if args.patience and args.patience > 0 else None

    fc_suffix = f"_fc{args.forecast_min}" if args.forecast_min > 0 else ""
    exp_name = args.exp_name or ("_".join(channels) + ("_binary" if args.binary else "_3class") + fc_suffix)
    out_dir = Path(args.out_dir) if args.out_dir else HERE / "runs" / exp_name
    if out_dir.exists() and not args.force:
        raise SystemExit(f"[train_experiment] {out_dir} 이미 존재합니다 -- 다른 --exp-name을 쓰거나 "
                         f"--force로 덮어쓰세요(기존 실험 결과 보호).")
    events_dir = out_dir / "events"
    events_dir.mkdir(parents=True, exist_ok=True)

    # 이벤트 오버레이 상단 z-score와 forecast 라벨 시프트 둘 다 라벨 앵커(첫) 채널의
    # timeseries가 필요해서 여기서 한 번만 로드해 재사용.
    primary = channels[0]
    ts = pd.read_parquet(HERE / "dataset_v0" / f"timeseries_{args.detector}_{primary}.parquet")
    ts = ts.set_index("time")

    windows = load_windows(args.detector, channels)
    windows = apply_forecast_shift(windows, ts, args.forecast_min)
    window = infer_window(windows, channels)
    X = build_X(windows, channels, window)
    y, label_names = build_y(windows, args.binary)
    n_classes = len(label_names)
    label_colors = BINARY_LABEL_COLORS if args.binary else diag.LABEL_COLORS
    label_transform = (lambda arr: (arr > 0).astype(int)) if args.binary else None

    print(f"[train_experiment] exp={exp_name}  channels={channels}  binary={args.binary}  "
          f"forecast_min={args.forecast_min}  n_classes={n_classes}  windows={len(windows)}  "
          f"window_len={window}  X.shape={X.shape}  folds={args.folds}  epochs={args.epochs}  "
          f"patience={patience}")

    oof_proba, fold_histories, fold_of_episode = diag.run_oof(
        windows, X, y, n_splits=args.folds, n_classes=n_classes,
        input_size=len(channels), epochs=args.epochs, patience=patience)
    oof_pred = oof_proba.argmax(axis=1)
    covered = ~np.isnan(oof_proba).any(axis=1)   # folds=1 스모크 경로는 일부만 커버됨

    diag.plot_learning_curves(fold_histories, out_dir / "learning_curves.png")
    diag.plot_confusion_matrix(y[covered], oof_pred[covered], out_dir / "confusion_matrix.png",
                               label_names=label_names)

    metrics_df = build_metrics_table(windows, y, oof_pred, fold_of_episode, fold_histories,
                                     label_names, args.folds)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    print(f"[train_experiment] 저장 -> {out_dir / 'metrics.csv'}")
    print(metrics_df.to_string(index=False))

    run_config = {
        **vars(args), "channels_parsed": channels, "n_classes": n_classes,
        "label_names": label_names, "window_len": window, "X_shape": list(X.shape),
        "n_windows": int(len(windows)), "n_episodes": int(windows["episode_id"].nunique()),
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[train_experiment] 저장 -> {out_dir / 'run_config.json'}")

    # 이벤트 오버레이: 상단 z-score+라벨 음영은 forecast 여부와 무관하게 항상 실제
    # 물리 상태(nowcast, t 시점) 기준 -- ts는 위에서 이미 로드해 둔 것 그대로 재사용.
    events = pd.read_csv(HERE / "quality_check" / "events_reconciled.csv",
                        parse_dates=["onset_time", "peak_time", "end_time"])
    episode_of_event = windows[windows["event_id"] >= 0].groupby("event_id")["episode_id"].first().to_dict()
    amb_of_event = windows[windows["event_id"] >= 0].groupby("event_id")["ambiguous_peak"].any().to_dict()
    reps = diag.pick_representatives(events, windows)
    print(f"[train_experiment] 대표 이벤트: {reps}")

    ev_ids = sorted(episode_of_event.keys())[:diag.MAX_EVENT_PLOTS]
    saved = []
    for eid in ev_ids:
        row = events.loc[events["event_id"] == eid].iloc[0]
        tag = ""
        if eid == reps["big"]:
            tag = " [rep: big event]"
        elif eid == reps["ambiguous"]:
            tag = " [rep: ambiguous(weak) event]"
        elif eid == reps["holdout"]:
            tag = " [rep: time-holdout 2024-2025 event]"
        p = diag.plot_event_overlay(eid, row, ts, windows, oof_proba, fold_of_episode, episode_of_event,
                                    amb_of_event, events_dir, tag=tag, label_names=label_names,
                                    label_colors=label_colors, label_transform=label_transform)
        saved.append(p)
    print(f"[train_experiment] 이벤트 오버레이 {len(saved)}장 저장 -> {events_dir}")
    print(f"\n[train_experiment] 전체 산출물 -> {out_dir}")


if __name__ == "__main__":
    main()
