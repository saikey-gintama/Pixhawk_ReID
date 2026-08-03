"""
4_eval.py
==========
예측 모델 v0 -- Step 2: 5-fold GroupKFold 전체 평가 + 시간 홀드아웃(2019~2023
train / 2024~2025 test) 1건. Step 1(3_train.py, fold 0 단독)이 PASS(val
macro-F1 0.602)한 다음 단계 -- fold 0의 성능이 우연인지, 특히 rising recall이
fold마다 얼마나 흔들리는지가 이번 실행의 핵심 질문.

재사용 (재구현 없음 -- import만):
  3_train.py : feature_cols/train_tcn/load_dataset/LABEL_NAMES/WINDOW/N_SPLITS
      (Step 1과 똑같은 TCN 구조·하이퍼파라미터로 fold마다 학습해야 fold 간
      비교가 공정하다 -- "3_train"은 숫자로 시작해 import 문으로 바로 못 써서
      importlib.import_module로 로드)

시간 홀드아웃 연도 배정 규칙: episode_id별 window_end_time 최솟값의 연도를 그
episode의 "시작 연도"로 정의(=crop_start 연도의 사실상 동일한 프록시 -- 데이터
맨 앞부분에서만 발생하는 bg 미확보 dropna 때문에 아주 드물게 며칠 밀릴 수 있지만
연도 판정에는 영향 없음). 시작 연도가 2019~2023이면 그 episode 전체를 train,
2024~2025면 test로 통째로 배정 -- episode를 쪼개서 나누지 않음(GroupKFold와
동일한 시간 누수 방지 원칙).

사용:
  python 4_eval.py --detector metop03 --channel omni_p6
"""
from __future__ import annotations
import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
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

TRAIN_YEARS = (2019, 2023)
TEST_YEARS = (2024, 2025)


def fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    return {"macro_f1": macro_f1, "precision": p, "recall": r, "f1": f1, "support": support}


def class_counts(y: np.ndarray) -> dict:
    return {LABEL_NAMES[k]: int((y == k).sum()) for k in range(3)}


def run_one_fold(X_train, y_train, X_val, y_val, tag: str) -> dict:
    class_weight = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=y_train)
    model, _, _ = train_tcn(X_train, y_train, X_val, y_val, class_weight)
    with torch.no_grad():
        pred = model(torch.tensor(X_val, dtype=torch.float32)).argmax(dim=1).numpy()
    m = fold_metrics(y_val, pred)
    cm = confusion_matrix(y_val, pred, labels=[0, 1, 2])
    print(f"\n[{tag}] classification report")
    print(classification_report(y_val, pred, target_names=LABEL_NAMES, zero_division=0, digits=3))
    print(f"[{tag}] confusion matrix (rows=true, cols=pred) {LABEL_NAMES}")
    print(cm)
    m["cm"] = cm
    return m


def run_5fold(windows: pd.DataFrame, X: np.ndarray, y: np.ndarray) -> list[dict]:
    print("\n" + "#" * 70)
    print("# 1) 5-fold GroupKFold 전체 (group=episode_id)")
    print("#" * 70)
    groups = windows["episode_id"].values
    gkf = GroupKFold(n_splits=N_SPLITS)
    results = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(windows, groups=groups)):
        train_eps, val_eps = set(groups[train_idx]), set(groups[val_idx])
        assert not (train_eps & val_eps), f"fold {fold} episode 누수: {train_eps & val_eps}"

        y_train, y_val = y[train_idx], y[val_idx]
        cc = class_counts(y_val)
        print(f"\n--- fold {fold}/{N_SPLITS} --- train {len(train_idx)}개({len(train_eps)} episode) / "
              f"val {len(val_idx)}개({len(val_eps)} episode), val 클래스 분포: {cc}")

        m = run_one_fold(X[train_idx], y_train, X[val_idx], y_val, tag=f"fold {fold}")
        m["fold"] = fold
        m["n_val_episodes"] = len(val_eps)
        m["val_class_counts"] = cc
        results.append(m)
    return results


def summarize_5fold(results: list[dict]) -> np.ndarray:
    print("\n" + "=" * 70)
    print("[5-fold 요약] fold별 상세")
    print("=" * 70)
    header = (f"{'fold':>4} {'val_eps':>7} {'quiet':>7} {'rising':>7} {'decr':>7} "
              f"{'macro_f1':>9} {'recall_q':>9} {'recall_r':>9} {'recall_d':>9}")
    print(header)
    for m in results:
        cc = m["val_class_counts"]
        r = m["recall"]
        print(f"{m['fold']:>4} {m['n_val_episodes']:>7} {cc['quiet']:>7} {cc['rising']:>7} "
              f"{cc['decreasing']:>7} {m['macro_f1']:>9.4f} {r[0]:>9.4f} {r[1]:>9.4f} {r[2]:>9.4f}")

    macro_f1s = np.array([m["macro_f1"] for m in results])
    recalls = np.stack([m["recall"] for m in results])   # (5,3)
    precisions = np.stack([m["precision"] for m in results])
    f1s = np.stack([m["f1"] for m in results])

    print("\n[5-fold 평균 ± 표준편차]")
    print(f"  macro-F1        : {macro_f1s.mean():.4f} +/- {macro_f1s.std():.4f}")
    for k, name in enumerate(LABEL_NAMES):
        print(f"  {name:<11} precision={precisions[:,k].mean():.4f}+/-{precisions[:,k].std():.4f}  "
              f"recall={recalls[:,k].mean():.4f}+/-{recalls[:,k].std():.4f}  "
              f"f1={f1s[:,k].mean():.4f}+/-{f1s[:,k].std():.4f}")

    cm_sum = np.sum([m["cm"] for m in results], axis=0)
    print("\n[5-fold 합산 confusion matrix] (rows=true, cols=pred)", LABEL_NAMES)
    print(cm_sum)

    # 진단: rising 샘플이 극단적으로 적거나 recall이 튀는 fold 짚어내기
    rising_support = np.array([m["val_class_counts"]["rising"] for m in results])
    lo, hi = rising_support.argmin(), rising_support.argmax()
    print(f"\n[진단] fold별 val rising 샘플 수: {rising_support.tolist()} "
          f"(최소 fold {lo}={rising_support[lo]}개, 최대 fold {hi}={rising_support[hi]}개)")
    if rising_support.min() < rising_support.mean() * 0.3:
        print(f"  -> fold {lo}은 다른 fold 대비 rising 샘플이 크게 적음(episode 22개를 5로 나눠 "
              f"fold당 val 4~5episode뿐이라, 우연히 rising 구간이 긴/짧은 이벤트가 한 fold에 "
              f"몰린 결과로 보임 -- recall 표본이 애초에 적어 그 fold의 rising recall 값 자체가 "
              f"불안정할 수 있음).")
    f1_range = recalls[:, 1].max() - recalls[:, 1].min()
    print(f"[진단] rising recall 범위: {recalls[:,1].min():.3f} ~ {recalls[:,1].max():.3f} "
          f"(폭 {f1_range:.3f}) -- 폭이 넓을수록 fold 0 하나만으로 판단한 0.46이 대표값이 아닐 수 있음.")
    return cm_sum


def run_time_holdout(windows: pd.DataFrame, X: np.ndarray, y: np.ndarray):
    print("\n" + "#" * 70)
    print(f"# 2) 시간 홀드아웃: train {TRAIN_YEARS[0]}~{TRAIN_YEARS[1]} / test {TEST_YEARS[0]}~{TEST_YEARS[1]}")
    print("#" * 70)
    ep_start_year = windows.groupby("episode_id")["window_end_time"].min().dt.year
    train_eps = set(ep_start_year[(ep_start_year >= TRAIN_YEARS[0]) & (ep_start_year <= TRAIN_YEARS[1])].index)
    test_eps = set(ep_start_year[(ep_start_year >= TEST_YEARS[0]) & (ep_start_year <= TEST_YEARS[1])].index)
    uncovered = set(ep_start_year.index) - train_eps - test_eps
    assert not uncovered, f"연도 배정 안 된 episode: {uncovered}"
    assert not (train_eps & test_eps)

    print(f"[time-holdout] episode 시작 연도: {ep_start_year.sort_index().to_dict()}")
    print(f"[time-holdout] train episode {len(train_eps)}개, test episode {len(test_eps)}개")

    train_mask = windows["episode_id"].isin(train_eps).values
    test_mask = windows["episode_id"].isin(test_eps).values
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[test_mask], y[test_mask]

    print(f"[time-holdout] train 윈도우 {len(X_train)}개 클래스 분포: {class_counts(y_train)}")
    print(f"[time-holdout] test  윈도우 {len(X_val)}개 클래스 분포: {class_counts(y_val)}")

    m = run_one_fold(X_train, y_train, X_val, y_val, tag="time-holdout")
    print("\n[time-holdout 요약]")
    print(f"  macro-F1 = {m['macro_f1']:.4f}")
    for k, name in enumerate(LABEL_NAMES):
        print(f"  {name:<11} precision={m['precision'][k]:.4f}  recall={m['recall'][k]:.4f}  f1={m['f1'][k]:.4f}")
    return m


def main():
    ap = argparse.ArgumentParser(description="예측 모델 v0 Step 2: 5-fold 전체 + 시간 홀드아웃")
    ap.add_argument("--detector", default="metop03")
    ap.add_argument("--channel", default="omni_p6")
    args = ap.parse_args()

    windows = load_dataset(args.detector, args.channel)
    cols = feature_cols(WINDOW)
    X = windows[cols].values.astype(np.float32)
    y = windows["label"].values.astype(np.int64)
    print(f"[data] windows {len(windows)}개, episode {windows['episode_id'].nunique()}개")

    fold_results = run_5fold(windows, X, y)
    summarize_5fold(fold_results)

    holdout_result = run_time_holdout(windows, X, y)

    print("\n" + "=" * 70)
    print("[Step 2 종합]")
    print("=" * 70)
    macro_f1s = np.array([m["macro_f1"] for m in fold_results])
    rising_recalls = np.array([m["recall"][1] for m in fold_results])
    print(f"  5-fold macro-F1 평균 {macro_f1s.mean():.4f} +/- {macro_f1s.std():.4f} "
          f"(fold 0 단독값과 비교: Step 1 = 0.602)")
    print(f"  5-fold rising recall 평균 {rising_recalls.mean():.4f} +/- {rising_recalls.std():.4f} "
          f"(fold 0 단독값과 비교: Step 1 = 0.46)")
    print(f"  시간 홀드아웃({TEST_YEARS[0]}~{TEST_YEARS[1]} test) macro-F1 {holdout_result['macro_f1']:.4f}, "
          f"rising recall {holdout_result['recall'][1]:.4f}")


if __name__ == "__main__":
    main()
