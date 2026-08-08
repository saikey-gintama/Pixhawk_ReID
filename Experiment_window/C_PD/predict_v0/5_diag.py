"""
5_diag.py
==========
예측 모델 v0 -- Step 3: 진단 그림. rising recall 0.47(홀드아웃 0.38)이
"z-score 자체가 안 두꺼워져서 못 잡는 것"(신호/피처 한계)인지 "rising/decreasing
경계 근처에서만 헷갈리는 것"(라벨 정의 문제)인지 눈으로 판별한다. 이 판단이
다음 단계(v1 힐베르트 포락선 vs 라벨 정의 수정)를 정한다.

재사용 (재구현 없음 -- import만):
  3_train.py : feature_cols/train_tcn/load_dataset/LABEL_NAMES/WINDOW/N_SPLITS
      (Step 1/2와 동일 TCN 구조·하이퍼파라미터. "3_train"은 숫자로 시작해
      import 문 대신 importlib.import_module로 로드)
  2_build_dataset.py가 만든 timeseries_*.parquet(zscore/label/event_id 전
      구간)과 1_check_labels.py 산출물 quality_check/events_reconciled.csv
      (onset/peak/end 시각) -- 새로 계산하지 않고 그대로 읽어 오버레이에 사용.

out-of-fold(OOF) 원칙: 4_eval.py와 동일한 5-fold GroupKFold(group=episode_id)를
다시 학습하되, 이번엔 예측 확률과 학습곡선을 저장해야 해서 별도 루프로 작성
(4_eval.py의 run_5fold와 로직은 같지만 반환값이 다름). 모든 이벤트는 자신이
val이었던 fold의 모델로만 예측 -- train에 쓰인 모델로 그린 그림은 무의미하므로.

출력 (predict_v0/diag_v0/):
  learning_curves.png        : 5-fold 각 학습곡선(train/val loss, val macro-F1)
  confusion_matrix_5fold.png : 5-fold 합산 confusion matrix (OOF)
  events/event{id:03d}_*.png : 이벤트별 오버레이(상단 z-score+3상태 라벨 음영+
      onset/peak/end 수직선, 하단 OOF 3클래스 확률), 최대 100장. 대표 3종
      (큰 이벤트/약한 이벤트/시간홀드아웃 이벤트)은 제목에 표시.

사용:
  python 5_diag.py --detector metop03 --channel omni_p6
"""
from __future__ import annotations
import argparse
import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_train = importlib.import_module("3_train")  # "3_train"은 숫자로 시작 -- importlib 필요
feature_cols = _train.feature_cols
train_tcn = _train.train_tcn
load_dataset = _train.load_dataset
LABEL_NAMES = _train.LABEL_NAMES
WINDOW = _train.WINDOW
N_SPLITS = _train.N_SPLITS

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

TEST_YEARS = (2024, 2025)   # 4_eval.py의 시간홀드아웃과 동일 정의(대표 이벤트 선정용)
DISPLAY_PAD_DAYS = 3         # 오버레이에 보여줄 이벤트 앞뒤 여유(학습용 크롭 10일보다 좁게)
MAX_EVENT_PLOTS = 100

# quiet은 음영 없음(배경), rising/decreasing만 칠함. 하단 확률 곡선도 같은 색으로
# 맞춰 상/하단을 눈으로 바로 대응시킨다.
LABEL_COLORS = {0: "#999999", 1: "tab:orange", 2: "tab:purple"}


def run_oof(windows: pd.DataFrame, X: np.ndarray, y: np.ndarray, *,
           n_splits: int = N_SPLITS, n_classes: int = 3, input_size: int = 1,
           epochs: int = _train.EPOCHS, patience: int | None = None,
           checkpoint_dir: Path | None = None):
    """5-fold GroupKFold(group=episode_id)를 학습해 OOF 확률·fold별 학습곡선·
    episode_id->held-out fold 매핑을 반환. 4_eval.py의 run_5fold와 같은 분할
    로직이지만 이번엔 예측 확률/학습곡선을 저장해야 해서 별도로 돈다.
    n_splits/n_classes/input_size/epochs/patience 기본값은 5_diag.py 자체 실행(Step 3,
    3클래스·단일채널·30epoch·조기종료 없음)과 동일 -- train_experiment.py가 다채널·
    이진·긴 학습+조기종료로 이 함수를 그대로 재사용하기 위한 일반화.

    checkpoint_dir 지정 시 fold마다 학습된(best epoch로 복원된) 모델의 state_dict를
    checkpoint_dir/fold{k}.pt로 저장(학습 로직 자체는 변경 없음 -- train_tcn 호출 뒤
    이미 메모리에 있는 model을 디스크에 남기기만 함). 크롭 밖 전 구간 추론
    (8_validation.py) 등 재학습 없이 나중에 모델을 다시 쓰기 위한 용도.

    n_splits=1은 GroupKFold가 지원하지 않는 값(sklearn 최소 2)이라 스모크 테스트 전용
    특수 경로로 처리한다 -- episode를 80/20으로 한 번만 나눠 파이프라인만 빠르게
    확인(성능 판단용 아님, val 20% 밖 윈도우는 OOF 커버리지 없이 NaN으로 남음)."""
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    groups = windows["episode_id"].values
    n = len(windows)
    oof_proba = np.full((n, n_classes), np.nan, dtype=np.float64)
    fold_histories = []
    fold_of_episode: dict[int, int] = {}

    if n_splits == 1:
        uniq_eps = np.unique(groups)
        rng = np.random.RandomState(0)
        rng.shuffle(uniq_eps)
        n_val_eps = max(1, int(round(len(uniq_eps) * 0.2)))
        val_eps = set(uniq_eps[:n_val_eps].tolist())
        val_idx = np.where(np.isin(groups, list(val_eps)))[0]
        train_idx = np.where(~np.isin(groups, list(val_eps)))[0]
        splits = [(train_idx, val_idx)]
        print(f"[run_oof] n_splits=1 스모크 경로: episode {len(uniq_eps)}개 중 {len(val_eps)}개만 "
              f"val(OOF 부분 커버리지, 성능 판단용 아님)")
    else:
        splits = list(GroupKFold(n_splits=n_splits).split(windows, groups=groups))

    for fold, (train_idx, val_idx) in enumerate(splits):
        train_eps, val_eps = set(groups[train_idx]), set(groups[val_idx])
        assert not (train_eps & val_eps), f"fold {fold} episode 누수: {train_eps & val_eps}"
        for eid in val_eps:
            fold_of_episode[int(eid)] = fold

        y_train, y_val = y[train_idx], y[val_idx]
        class_weight = compute_class_weight("balanced", classes=np.arange(n_classes), y=y_train)
        print(f"\n--- OOF fold {fold}/{n_splits} --- train {len(train_idx)}개 / val {len(val_idx)}개")
        model, history, best_f1 = train_tcn(X[train_idx], y_train, X[val_idx], y_val, class_weight,
                                            epochs=epochs, patience=patience,
                                            input_size=input_size, n_classes=n_classes)
        if checkpoint_dir is not None:
            torch.save(model.state_dict(), checkpoint_dir / f"fold{fold}.pt")
        with torch.no_grad():
            proba = torch.softmax(model(torch.tensor(X[val_idx], dtype=torch.float32)), dim=1).numpy()
        oof_proba[val_idx] = proba
        fold_histories.append(history)
        print(f"[OOF fold {fold}] best val macro-F1 = {best_f1:.4f} "
              f"(stopped @ epoch {history['stopped_epoch']}, best @ epoch {history['best_epoch']})")

    if n_splits > 1:
        assert not np.isnan(oof_proba).any(), "OOF 확률 누락된 윈도우 있음 -- 분할 버그"
    return oof_proba, fold_histories, fold_of_episode


def plot_learning_curves(fold_histories: list[dict], out_path: Path):
    n_folds = len(fold_histories)
    fig, axes = plt.subplots(2, n_folds, figsize=(4 * n_folds, 6), sharex=True, squeeze=False)
    for fold, hist in enumerate(fold_histories):
        epochs = np.arange(1, len(hist["train_loss"]) + 1)
        ax0, ax1 = axes[0, fold], axes[1, fold]
        ax0.plot(epochs, hist["train_loss"], label="train_loss", color="tab:blue")
        ax0.plot(epochs, hist["val_loss"], label="val_loss", color="tab:red")
        ax0.set_title(f"fold {fold}")
        ax0.grid(alpha=0.3)
        ax1.plot(epochs, hist["val_macro_f1"], color="tab:green")
        ax1.set_xlabel("epoch")
        ax1.grid(alpha=0.3)
        if fold == 0:
            ax0.set_ylabel("loss")
            ax1.set_ylabel("val macro-F1")
            ax0.legend(fontsize=8)
    fig.suptitle(f"{n_folds}-fold learning curves (TCN, group=episode_id)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[diag] 저장 -> {out_path}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path,
                          label_names: list[str] | None = None) -> np.ndarray:
    names = list(label_names) if label_names is not None else LABEL_NAMES
    n = len(names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_xticklabels(names)
    ax.set_yticks(range(n)); ax.set_yticklabels(names)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"{n}-class summed confusion matrix (OOF)")
    # 3클래스(rising 존재)일 때만 기존처럼 rising 행의 off-diagonal만 강조하고,
    # 그 외(이진 등)는 모든 off-diagonal을 일반적으로 강조.
    rising_row = names.index("rising") if "rising" in names else None
    for i in range(n):
        for j in range(n):
            highlight = (i != j) and (rising_row is None or i == rising_row)
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="crimson" if highlight else "black",
                    fontweight="bold" if highlight else "normal")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[diag] 저장 -> {out_path}")
    return cm


def _shade_labels(ax, ts_win: pd.DataFrame, label_colors: dict | None = None):
    colors = label_colors if label_colors is not None else LABEL_COLORS
    lab = ts_win["label"].values
    idx = ts_win.index
    if len(lab) == 0:
        return
    start = 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[start]:
            l = int(lab[start])
            if l != 0:
                ax.axvspan(idx[start], idx[i - 1], color=colors[l], alpha=0.25, lw=0)
            start = i


def plot_event_overlay(event_id: int, ev_row: pd.Series, ts: pd.DataFrame, windows: pd.DataFrame,
                       oof_proba: np.ndarray, fold_of_episode: dict, episode_of_event: dict,
                       amb_of_event: dict, out_dir: Path, tag: str = "",
                       label_names: list[str] | None = None, label_colors: dict | None = None,
                       label_transform=None):
    """label_names/label_colors 미지정 시 기존 3클래스(quiet/rising/decreasing) 기본값.
    label_transform(raw_label_array)->array 는 상단 음영용 라벨을 바꿔치기(예: 이진
    실험에서 rising+decreasing을 event로 합치는 (label>0).astype(int)) -- 지정 시
    하단 확률 패널의 클래스 수(label_names 길이)와 반드시 맞아야 한다."""
    names = list(label_names) if label_names is not None else LABEL_NAMES
    colors = label_colors if label_colors is not None else LABEL_COLORS
    onset, peak, end = ev_row["onset_time"], ev_row["peak_time"], ev_row["end_time"]
    t0 = onset - pd.Timedelta(days=DISPLAY_PAD_DAYS)
    t1 = end + pd.Timedelta(days=DISPLAY_PAD_DAYS)

    ts_win = ts.loc[t0:t1].copy()
    if label_transform is not None:
        ts_win["label"] = label_transform(ts_win["label"].values)
    w_mask = (windows["window_end_time"] >= t0) & (windows["window_end_time"] <= t1)
    w_win = windows.loc[w_mask]
    proba_win = oof_proba[w_win.index.values]

    fig, (ax, axp) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1]})
    _shade_labels(ax, ts_win, colors)
    ax.plot(ts_win.index, ts_win["zscore"], color="black", lw=0.8, label="z-score")
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    for t, color in ((onset, "green"), (peak, "red"), (end, "blue")):
        if pd.notna(t) and t0 <= t <= t1:
            ax.axvline(t, color=color, lw=1.3, alpha=0.85)
    ep = episode_of_event.get(event_id)
    fold = fold_of_episode.get(ep, -1)
    amb = bool(amb_of_event.get(event_id, False))
    ax.set_ylabel("z-score")
    ax.set_title(f"event {event_id}{tag}  onset={onset}  peak_count={ev_row['peak_count']:.1f}  "
                f"ambiguous_peak={amb}  (held out in fold {fold})", fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    for k, name in enumerate(names):
        axp.plot(w_win["window_end_time"], proba_win[:, k], color=colors[k], lw=1.2, label=name)
    axp.axhline(0.5, color="gray", lw=0.5, ls=":")
    for t, color in ((onset, "green"), (peak, "red"), (end, "blue")):
        if pd.notna(t) and t0 <= t <= t1:
            axp.axvline(t, color=color, lw=1.3, alpha=0.85)
    axp.set_ylabel("OOF probability")
    axp.set_ylim(-0.02, 1.02)
    axp.legend(loc="upper right", fontsize=8)
    axp.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    out_path = out_dir / f"event{event_id:03d}_{onset:%Y-%m-%d}.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def pick_representatives(events: pd.DataFrame, windows: pd.DataFrame) -> dict:
    big_cands = events[events["peak_count"] > 1000]
    big_id = 49 if 49 in big_cands["event_id"].values else int(
        big_cands.sort_values("peak_count", ascending=False)["event_id"].iloc[0])

    amb_ids = sorted(windows.loc[windows["ambiguous_peak"] & (windows["event_id"] >= 0),
                                 "event_id"].unique())
    amb_id = int(amb_ids[0]) if amb_ids else None

    ep_start_year = windows.groupby("episode_id")["window_end_time"].min().dt.year
    test_eps = set(ep_start_year[ep_start_year.isin(range(TEST_YEARS[0], TEST_YEARS[1] + 1))].index)
    test_events = events[events["event_id"].isin(
        windows.loc[windows["episode_id"].isin(test_eps) & (windows["event_id"] >= 0), "event_id"].unique())]
    test_events = test_events[~test_events["event_id"].isin([big_id, amb_id])]
    holdout_id = int(test_events.sort_values("peak_count", ascending=False)["event_id"].iloc[0]) \
        if len(test_events) else None

    return {"big": big_id, "ambiguous": amb_id, "holdout": holdout_id}


def main():
    ap = argparse.ArgumentParser(description="예측 모델 v0 Step 3: 진단 그림")
    ap.add_argument("--detector", default="metop03")
    ap.add_argument("--channel", default="omni_p6")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else HERE / "diag_v0"
    events_dir = out_dir / "events"
    events_dir.mkdir(parents=True, exist_ok=True)

    windows = load_dataset(args.detector, args.channel)
    cols = feature_cols(WINDOW)
    X = windows[cols].values.astype(np.float32)
    y = windows["label"].values.astype(np.int64)
    ts = pd.read_parquet(HERE / "dataset_v0" / f"timeseries_{args.detector}_{args.channel}.parquet")
    ts = ts.set_index("time")
    events = pd.read_csv(HERE / "quality_check" / "events_reconciled.csv",
                        parse_dates=["onset_time", "peak_time", "end_time"])
    print(f"[diag] windows {len(windows)}개, episode {windows['episode_id'].nunique()}개, "
          f"events {len(events)}개")

    oof_proba, fold_histories, fold_of_episode = run_oof(windows, X, y)
    oof_pred = oof_proba.argmax(axis=1)

    plot_learning_curves(fold_histories, out_dir / "learning_curves.png")
    cm = plot_confusion_matrix(y, oof_pred, out_dir / "confusion_matrix_5fold.png")

    n_rising = cm[1].sum()
    print("\n[diag] rising 행 off-diagonal 분해 (OOF, 5-fold 합산):")
    print(f"  rising -> quiet      : {cm[1,0]}개 ({cm[1,0]/n_rising*100:.1f}%)")
    print(f"  rising -> rising(정답): {cm[1,1]}개 ({cm[1,1]/n_rising*100:.1f}%)")
    print(f"  rising -> decreasing : {cm[1,2]}개 ({cm[1,2]/n_rising*100:.1f}%)")

    episode_of_event = windows[windows["event_id"] >= 0].groupby("event_id")["episode_id"].first().to_dict()
    amb_of_event = windows[windows["event_id"] >= 0].groupby("event_id")["ambiguous_peak"].any().to_dict()
    reps = pick_representatives(events, windows)
    print(f"\n[diag] 대표 이벤트: {reps}")

    ev_ids = sorted(episode_of_event.keys())[:MAX_EVENT_PLOTS]
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
        p = plot_event_overlay(eid, row, ts, windows, oof_proba, fold_of_episode, episode_of_event,
                               amb_of_event, events_dir, tag=tag)
        saved.append(p)
    print(f"\n[diag] 이벤트 오버레이 {len(saved)}장 저장 -> {events_dir}")
    print(f"[diag] 전체 산출물 -> {out_dir}")


if __name__ == "__main__":
    main()
