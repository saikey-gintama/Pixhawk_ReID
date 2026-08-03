"""
train_v0.py
============
예측 모델 v0 -- Step 1: "raw z-score만으로 포락선(rising/decreasing)이 학습되는가"를
GroupKFold 중 fold 하나로 먼저 확인한다(5-fold 전체/시간홀드아웃/진단그림은 이 스크립트를
통과 판정한 뒤의 Step 2/3 -- 이번 실행 범위 아님).

split의 group은 build_dataset_v0.py가 만든 episode_id를 쓴다(event_id가 아님) --
quiet 윈도우는 event_id=-1 하나로 뭉쳐 있어 그걸 그대로 GroupKFold group으로 쓰면
quiet 전부가 항상 통째로 같은 fold에만 들어가 버린다(스플릿이 사실상 망가짐).
episode_id는 이벤트+앞뒤 pad 구간을 하나로 묶은 그룹이라 quiet도 가까운 이벤트에
귀속되어 정상적으로 5-fold에 흩어진다 -- 시간적으로 인접한(거의 중복인) 슬라이딩
윈도우가 train/val에 걸쳐 나뉘는 누수도 이 그룹 정의가 막아준다.

판정선(1-fold): val macro-F1이 quiet-only 자명 baseline(대략 1/3 수준)을 유의미하게
넘는가. 못 넘으면 여기서 멈추고 데이터/증강/윈도우 재검토(TCN 구조 자체를 더 손보기
전에).

사용:
  python train_v0.py --detector metop03 --channel omni_p6
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tcn import TCNClassifier  # noqa: E402

SEED = 0
WINDOW = 14
N_SPLITS = 5
FOLD_TO_RUN = 0  # Step 1: 이 fold 하나만 학습/평가
LABEL_NAMES = ["quiet", "rising", "decreasing"]
EPOCHS = 30
BATCH_SIZE = 256
LR = 1e-3

torch.manual_seed(SEED)
np.random.seed(SEED)


def load_dataset(detector: str, channel: str) -> pd.DataFrame:
    path = HERE.parent / "manual_labels" / "dataset_v0" / f"windows_{detector}_{channel}.parquet"
    return pd.read_parquet(path)


def feature_cols(window: int) -> list[str]:
    return [f"z_lag{k}" for k in range(window - 1, -1, -1)]  # 과거(lag13) -> 현재(lag0) 순


def make_split(windows: pd.DataFrame, n_splits: int, fold: int):
    groups = windows["episode_id"].values
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(windows, groups=groups))
    train_idx, val_idx = splits[fold]
    train_eps = set(groups[train_idx])
    val_eps = set(groups[val_idx])
    assert not (train_eps & val_eps), f"episode 누수: {train_eps & val_eps}"
    return train_idx, val_idx


def train_tcn(X_train, y_train, X_val, y_val, class_weight):
    model = TCNClassifier(input_size=1, num_channels=(16, 16, 16), kernel_size=3, dropout=0.2, n_classes=3)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weight, dtype=torch.float32))

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.long)
    Xva = torch.tensor(X_val, dtype=torch.float32)
    yva = torch.tensor(y_val, dtype=torch.long)

    n = len(Xtr)
    history = {"train_loss": [], "val_loss": [], "val_macro_f1": []}
    best_f1, best_state = -1.0, None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb, yb = Xtr[idx], ytr[idx]
            opt.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        train_loss = total_loss / n

        model.eval()
        with torch.no_grad():
            val_out = model(Xva)
            val_loss = criterion(val_out, yva).item()
            val_pred = val_out.argmax(dim=1).numpy()
        val_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(val_f1)
        print(f"[tcn] epoch {epoch:02d}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_macro_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, history, best_f1


def evaluate(y_true, y_pred, name: str) -> float:
    print(f"\n[{name}] classification report")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES, zero_division=0, digits=3))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    print(f"[{name}] confusion matrix (rows=true, cols=pred) {LABEL_NAMES}")
    print(cm)
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def main():
    ap = argparse.ArgumentParser(description="예측 모델 v0 Step 1: 1-fold 학습/평가")
    ap.add_argument("--detector", default="metop03")
    ap.add_argument("--channel", default="omni_p6")
    args = ap.parse_args()

    windows = load_dataset(args.detector, args.channel)
    cols = feature_cols(WINDOW)
    X = windows[cols].values.astype(np.float32)
    y = windows["label"].values.astype(np.int64)

    train_idx, val_idx = make_split(windows, N_SPLITS, FOLD_TO_RUN)
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    episodes = windows["episode_id"].values
    print(f"[split] fold {FOLD_TO_RUN}/{N_SPLITS} (episode_id 그룹): "
          f"train {len(train_idx)}개 / val {len(val_idx)}개")
    print(f"[split] train episodes {len(set(episodes[train_idx]))}개, "
          f"val episodes {len(set(episodes[val_idx]))}개")
    for name, yy in (("train", y_train), ("val", y_val)):
        counts = {LABEL_NAMES[k]: int((yy == k).sum()) for k in range(3)}
        print(f"[split] {name} 클래스 분포: {counts}")

    class_weight = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=y_train)
    print(f"[split] class_weight (balanced, train 기준): "
          f"{dict(zip(LABEL_NAMES, np.round(class_weight, 3)))}")

    # quiet-only 자명 baseline
    quiet_pred = np.zeros_like(y_val)
    quiet_f1 = evaluate(y_val, quiet_pred, "quiet-only baseline")

    # 로지스틱 회귀 대조 baseline (같은 z-score 입력, 시퀀스 flatten)
    logreg = LogisticRegression(max_iter=2000, class_weight="balanced")
    logreg.fit(X_train, y_train)
    logreg_pred = logreg.predict(X_val)
    logreg_f1 = evaluate(y_val, logreg_pred, "logistic regression baseline")

    # TCN
    model, history, _ = train_tcn(X_train, y_train, X_val, y_val, class_weight)
    with torch.no_grad():
        tcn_pred = model(torch.tensor(X_val, dtype=torch.float32)).argmax(dim=1).numpy()
    tcn_f1 = evaluate(y_val, tcn_pred, "TCN")

    print("\n" + "=" * 70)
    print("[Step 1 판정] val macro-F1 비교")
    print("=" * 70)
    print(f"  quiet-only baseline : {quiet_f1:.4f}")
    print(f"  logistic regression : {logreg_f1:.4f}")
    print(f"  TCN                 : {tcn_f1:.4f}")
    margin = tcn_f1 - quiet_f1
    verdict = "PASS" if margin > 0.10 else "FAIL"
    print(f"  TCN - quiet-only baseline = {margin:+.4f}  -> {verdict} "
          f"(판정 기준: baseline 대비 +0.10 이상 유의미 개선)")


if __name__ == "__main__":
    main()
