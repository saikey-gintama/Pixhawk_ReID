"""
4_validation.py
=================
FSM(규칙 트리거)과 TCN(학습 분류기)을 동일 조건에서 두 정답(NOAA SPE 카탈로그,
손라벨)에 대해 각각 평가해 2x2 비교표를 만든다. 기존 산출물만 소비 -- 학습·재학습
없음(TCN은 3_train_experiment.py가 저장한 checkpoints/fold{k}.pt + manifest.json을
불러와 추론만 한다).

재사용 (재구현 없음 -- import만):
  _match_core_poes.py(POES/event_MATCHER, as `core`) : det_matched_mask/_cluster_indices를
      내부에서 쓰는 match_events(det, cat, tol_h) -- POD/FAR(단순), event 단위 클러스터링
      기반 event_FAR/precision/recall/f1/TP/FP/FN을 한 번에 계산. MATCH_TOL_H(24h)/ERA도
      그대로 재사용. FSM/TCN 양쪽 다 이 함수 하나로 평가 -- 클러스터링(24h)은
      match_events 내부의 _cluster_events(_cluster_indices 재사용)가 처리하므로 이
      스크립트에서 별도 사전 클러스터링을 하지 않는다(det를 raw 그대로 넘김).
  fsm_count_spe_quietoff_mad_poes.py(as fsm_engine) : detect_segments(문턱 넘는
      연속구간 -> onset/peak/end, min_duration_h 적용)를 TCN 확률 시계열에도 그대로
      적용(문턱=고정 threshold 시리즈로만 바꿔치기, 알고리즘 자체는 무변경) --
      MIN_SPE_DURATION_H도 FSM과 동일하게 재사용.
  1_check_labels.py(as check) : _fsm_onset_csv_path(7,7) -- 기존 FSM onset CSV
      (quietoff_mad w7k7 on0.1 pk0) 경로 조립, 재계산 없이 그대로 읽음.
  2_build_dataset.py(as bd) : compute_channel_zscore/_lag_cols -- 전 구간(크롭 없음)
      z-score 피처를 만들 때 학습 데이터셋 빌더와 동일 공식 재사용(새 추정 없음).
  0_label_events_gui.py : load_channel_series(전 구간 count 로드).
  tcn.py : TCNClassifier -- manifest.json으로 구조 복원, checkpoints/fold{k}.pt로
      가중치 로드(학습 안 함, 추론만).
  diag.py : _shade_labels(대표 이벤트 그림의 정답 구간 음영), pick_representatives.

조건 통일:
  1. 채널: 단일(omni_p6) / 다채널(omni_p6,omni_p7,pro_tel0_p5) 둘 다 평가.
     FSM 다채널 = cFS RPN 논리(채널 OR 결합) -- 3채널 raw onset을 그냥 합쳐서
     match_events에 넘김(내부 24h 클러스터링이 서로 다른 채널의 인접 onset도
     하나의 "사건"으로 묶어줘 OR 결합 효과를 낸다).
  2. 평가 구간: 전 구간(2019~2025, 크롭 없음). TCN은 학습을 이벤트 중심 크롭
     (predict_v0/dataset_v0/windows_*.parquet)에서 했지만 추론은 크롭 밖까지
     포함한 전체 count 시계열에 돌린다 -- 학습 때 못 본 먼 quiet에서의 오경보를
     그대로 드러내는 게 목적. TCN 예측 방법: 크롭 안/밖을 구분하지 않고 5-fold
     체크포인트 전부로 예측한 확률을 단순 평균(앙상블) -- "각 fold로 예측 후
     fold 평균" 방식 채택(반대쪽 "fold별 산출 후 지표 평균" 방식은 안 씀,
     이유: 크롭 밖은 애초에 OOF 정의가 없어 fold별 지표를 낼 기준이 없고,
     크롭 안팎에 다른 방법을 섞으면 경계에서 지표가 불연속으로 튀는 것을
     피하려고 전 구간에 앙상블 하나로 통일).
  3. TCN 확률(quiet 제외 나머지 클래스 합 = "event" 확률) 시계열 -> onset 사건
     변환: fsm_engine.detect_segments(같은 min_duration_h=MIN_SPE_DURATION_H)로
     문턱 이상 연속구간을 검출, match_events 내부 24h 클러스터링을 FSM과 동일하게
     적용(위 참고).
  4. 매칭 tolerance: ±24h 하나로 통일(core.MATCH_TOL_H).
  5. TCN은 out-of-fold만 사용(학습에 쓰인 fold의 모델로 그 fold 데이터를 평가하지
     않음) -- 크롭 안에서는 이 정의가 성립하지만 크롭 밖은 2번 항목의 앙상블로
     대체(엄밀한 OOF가 아님, 결과에 명시).

주의(결과 해석): TCN x 손라벨 칸은 라벨 정의를 공유하므로 구조적으로 유리하다.
교차 칸(TCN x NOAA, FSM x 손라벨)이 더 공정한 비교다.

출력 (predict_v0/validation_v0/):
  comparison_table.csv   : 2x2(+연산점) 비교표(POD/FAR/n_det/n_hit/n_fa/event_FAR/
      precision/recall/f1/TP/FP/FN), channel_config·ground_truth·detector·threshold별.
  threshold_sweep_{channel_config}.csv : TCN 임계 스윕 원본(POD/FAR/F1 등, 정답별).
  pr_curve_{channel_config}.png : precision-recall 곡선(정답별 2선) + 운용점 2개 표시.
  event_overlay/event{id}_*.png : FSM onset vs TCN onset 비교(대표 3개, 정답 구간 음영).

TCN 체크포인트가 없는 run(3_train_experiment.py의 checkpoint 저장 기능 이전에 돌린
run)은 그 채널 구성의 TCN 결과를 건너뛰고 경고만 출력 -- 학습을 대신 돌리지 않는다.

사용:
  python 4_validation.py --single-run omni_p6_binary --multi-run omni_p6_omni_p7_pro_tel0_p5_3class
"""
from __future__ import annotations
import argparse
import importlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent          # C_PD/predict_v0/
C_PD = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(C_PD / "POES"))
sys.path.insert(0, str(C_PD / "POES" / "event_MATCHER"))
sys.path.insert(0, str(C_PD / "POES" / "count_FSM"))

import _match_core_poes as core                              # noqa: E402
import fsm_count_spe_quietoff_mad_poes as fsm_engine          # noqa: E402
import diag                                                    # noqa: E402  -- 숫자로 시작 안 해 일반 import
# "1_check_labels"/"2_build_dataset"/"3_train_experiment"는 숫자로 시작해 import 문
# 대신 importlib.import_module로 로드(0_label_events_gui.py 참고).
check = importlib.import_module("1_check_labels")             # noqa: E402
bd = importlib.import_module("2_build_dataset")                # noqa: E402
te = importlib.import_module("3_train_experiment")             # noqa: E402
load_channel_series = importlib.import_module("0_label_events_gui").load_channel_series  # noqa: E402
from tcn import TCNClassifier                                  # noqa: E402

TOL_H = core.MATCH_TOL_H          # 24.0h, 프로젝트 전역 관례
MIN_DURATION_H = fsm_engine.MIN_SPE_DURATION_H   # FSM과 동일 최소 지속시간
FSM_W, FSM_K = 7, 7                # 기존 quietoff_mad w7k7 on0.1 pk0 재사용
DEFAULT_THRESHOLDS = [round(t, 2) for t in np.arange(0.05, 1.0, 0.05)]


# ── 정답(카탈로그) 로더 ──────────────────────────────────────────────────────

def load_noaa_catalog() -> pd.DataFrame:
    catalog_dir = C_PD / "NOAA_GOES" / "noaa_goes_spe_cache_parquet"
    io = core._import_event_io("noaa_goes_spe_io", str(catalog_dir))
    cat_all, _ = io.load(str(catalog_dir))
    cat = io.filter_by_date(cat_all, *core.ERA)
    if cat.index.tz is None:
        cat.index = cat.index.tz_localize("UTC")
    return cat


def load_manual_catalog() -> pd.DataFrame:
    """quality_check/events_reconciled.csv(손라벨, 재구현 없이 그대로) -> match_events가
    기대하는 형태(index=onset_time, max_time/max_pfu 컬럼)로 변환."""
    events = pd.read_csv(HERE / "results" / "quality_check" / "events_reconciled.csv",
                        parse_dates=["onset_time", "peak_time", "end_time"])
    events = events.dropna(subset=["onset_time", "peak_time"]).set_index("onset_time")
    events["max_time"] = events["peak_time"]
    events["max_pfu"] = events["peak_count"]   # 단위는 다르지만 match_events엔 크기 정보로만 쓰임
    return events


# ── FSM (기존 산출물 재사용, 재계산 없음) ────────────────────────────────────

def load_fsm_onsets(channels: list[str]) -> pd.DataFrame:
    """기존 fsm_onset CSV(quietoff_mad w7k7 on0.1 pk0)에서 channels에 속한 raw onset을
    그대로 모아 반환. 채널 2개 이상이면 concat만 하고 별도 병합 안 함 -- match_events
    내부 24h 클러스터링이 서로 다른 채널의 인접 onset도 하나의 '사건'으로 묶어줘
    OR 결합(cFS RPN) 효과를 낸다."""
    csv_path = check._fsm_onset_csv_path(FSM_W, FSM_K)
    raw = pd.read_csv(csv_path, parse_dates=["onset_time", "peak_time", "end_time"])
    raw = raw[raw["channel"].isin(channels)].reset_index(drop=True)
    if raw["onset_time"].dt.tz is None:
        raw["onset_time"] = raw["onset_time"].dt.tz_localize("UTC")
    return raw.sort_values("onset_time").reset_index(drop=True)


# ── TCN: 체크포인트 로드 + 전 구간 피처 + 앙상블 추론 ────────────────────────

def load_tcn_ensemble(run_dir: Path):
    """checkpoints/fold{k}.pt + manifest.json 로드. 체크포인트가 없으면(이 run이
    3_train_experiment.py의 checkpoint 저장 기능 이전에 돌았으면) None 반환 --
    호출부에서 그 채널 구성의 TCN 평가를 건너뛴다(학습 대신 돌리지 않음)."""
    ckpt_dir = run_dir / "checkpoints"
    manifest_path = ckpt_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"[4_validation] 경고: {ckpt_dir}에 manifest.json 없음 -- 이 run은 "
              f"checkpoint 저장 기능(3_train_experiment.py) 이전에 돌았을 가능성 -- "
              f"TCN 평가를 건너뜁니다. 재실행하면 나옵니다.")
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    models = []
    for fold in range(manifest["n_folds"]):
        fp = ckpt_dir / f"fold{fold}.pt"
        if not fp.exists():
            print(f"[4_validation] 경고: {fp} 없음 -- 이 run의 TCN 평가를 건너뜁니다.")
            return None
        model = TCNClassifier(input_size=manifest["input_size"],
                              num_channels=tuple(manifest["num_channels"]),
                              kernel_size=manifest["kernel_size"], dropout=manifest["dropout"],
                              n_classes=manifest["n_classes"])
        model.load_state_dict(torch.load(fp, weights_only=True))
        model.eval()
        models.append(model)
    print(f"[4_validation] {run_dir.name}: fold {len(models)}개 체크포인트 로드 완료 "
          f"(channels={manifest['channels']}, n_classes={manifest['n_classes']})")
    return models, manifest


def build_full_features(detector: str, channels: list[str], window: int):
    """전 구간(크롭 없음) z-score 슬라이딩 윈도우 피처. timeseries_*.parquet(라벨 앵커
    채널)+2_build_dataset.compute_channel_zscore(추가 채널)를 그대로 재사용 -- 학습용
    windows.parquet은 이벤트 중심으로 크롭돼 있어 전 구간 평가엔 쓸 수 없다."""
    primary = channels[0]
    ts = pd.read_parquet(HERE / "dataset_v0" / f"timeseries_{detector}_{primary}.parquet")
    ts = ts.set_index("time")
    z_series = [ts["zscore"]]
    for ch in channels[1:]:
        print(f"[4_validation] 추가 채널 {ch} 전 구간 z-score 계산 중")
        z_series.append(bd.compute_channel_zscore(detector, ch, 7, ts.index))

    if len(channels) == 1:
        cols = bd._lag_cols(z_series[0].values, window, "z_lag")
    else:
        cols = {}
        for i, zser in enumerate(z_series):
            cols.update(bd._lag_cols(zser.values, window, f"z_ch{i}_lag"))
    feat = pd.DataFrame(cols, index=ts.index[window - 1:])
    feat = feat.dropna()

    if len(channels) == 1:
        X = feat.values.astype(np.float32)
    else:
        arrs = []
        for i in range(len(channels)):
            cs = [f"z_ch{i}_lag{k}" for k in range(window - 1, -1, -1)]
            arrs.append(feat[cs].values.astype(np.float32))
        X = np.stack(arrs, axis=1)
    return X, feat.index, ts


def ensemble_event_proba(models: list, manifest: dict, X: np.ndarray) -> np.ndarray:
    """fold 모델 전부로 예측한 확률을 단순 평균(앙상블) -- 전 구간(크롭 안팎 구분 없이)
    균일하게 적용하는 방식 채택(모듈 docstring '조건 통일' 2번 참고). quiet=항상
    label_names[0]이므로 event 확률 = 1 - P(quiet)."""
    quiet_idx = manifest["label_names"].index("quiet")
    probs = []
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32)
        for model in models:
            p = torch.softmax(model(xb), dim=1).numpy()
            probs.append(p)
    ens = np.mean(probs, axis=0)
    return 1.0 - ens[:, quiet_idx]


# ── 확률 시계열 -> onset 사건 ─────────────────────────────────────────────

def probs_to_onsets(prob: pd.Series, threshold: float, min_duration_h: float) -> pd.DataFrame:
    """fsm_engine.detect_segments 재사용 -- 문턱을 고정 시리즈로 바꿔치기만 하고
    알고리즘(연속구간 검출 + 최소 지속시간)은 무변경."""
    thr_series = pd.Series(threshold, index=prob.index)
    bg_placeholder = pd.DataFrame({"bg_median": np.nan, "bg_std": np.nan}, index=prob.index)
    segs = fsm_engine.detect_segments(prob, thr_series, bg_placeholder, min_duration_h)
    if not segs:
        return pd.DataFrame(columns=["onset_time", "peak_time", "end_time"])
    return pd.DataFrame(segs)


_MATCH_SCALAR_KEYS = ("n_cat", "n_det", "pod", "far", "n_hit", "n_fa", "n_fa_saa",
                     "n_events_total", "n_events_tp", "n_events_fa", "event_far",
                     "TP", "FP", "FN", "precision", "recall", "f1")


def match_cell(det: pd.DataFrame, cat: pd.DataFrame, tol_h: float = TOL_H) -> dict:
    """core.match_events()는 fa_maglat/onset_diff_h 등 배열값 진단 필드도 같이
    반환하는데(POES 지자기 위도 등 별도 진단용, sweep_table도 안 씀) 비교표에는
    스칼라 요약 지표만 필요해서 여기서 골라낸다 -- match_events 자체는 무변경 재사용.

    예외적으로 onset_diff_h 만 분위수로 요약해 살린다 -- forecast(Δt) 모델은
    카탈로그 onset보다 Δt만큼 먼저 발화해야 정상이라 onset_diff_h의 중앙값이
    실제 리드타임 확보 여부를 직접 보여준다(배열 자체는 CSV 칸에 못 넣으니
    median/p25/p75/n 넉 점으로 요약).

    부호 규약(_match_core_poes.py:136, match_events() 안 `onset_diff.append(
    (on[jc]-b)/np.timedelta64(1,"h"))`, on=검출 onset, b=카탈로그 begin_time으로
    확인): onset_diff_h = 검출 onset - 카탈로그 onset. **음수 = 검출이 카탈로그보다
    선행(리드타임 확보), 양수 = 검출이 후행.** 404행 plot_pod_far_scatter()의
    "neg = POES leads" 축 라벨과 일치."""
    if len(det) == 0:
        return {"n_cat": len(cat), "pod": 0.0, "far": np.nan, "n_det": 0, "n_hit": 0,
                "n_fa": 0, "n_fa_saa": np.nan, "n_events_total": 0, "n_events_tp": 0,
                "n_events_fa": 0, "event_far": np.nan, "TP": 0, "FP": 0, "FN": len(cat),
                "precision": np.nan, "recall": 0.0, "f1": np.nan,
                "onset_diff_h_median": np.nan, "onset_diff_h_p25": np.nan,
                "onset_diff_h_p75": np.nan, "onset_diff_h_n": 0}
    r = core.match_events(det, cat, tol_h)
    out = {k: r[k] for k in _MATCH_SCALAR_KEYS}

    od = np.asarray(r.get("onset_diff_h", []), dtype=float)
    od = od[np.isfinite(od)]
    if len(od):
        out["onset_diff_h_median"] = float(np.median(od))
        out["onset_diff_h_p25"] = float(np.percentile(od, 25))
        out["onset_diff_h_p75"] = float(np.percentile(od, 75))
        out["onset_diff_h_n"] = int(len(od))
    else:
        out["onset_diff_h_median"] = np.nan
        out["onset_diff_h_p25"] = np.nan
        out["onset_diff_h_p75"] = np.nan
        out["onset_diff_h_n"] = 0
    return out


# ── 임계 스윕 + 운용점 선택 ───────────────────────────────────────────────

def sweep_thresholds(prob: pd.Series, thresholds: list[float], catalogs: dict) -> pd.DataFrame:
    rows = []
    for th in thresholds:
        det = probs_to_onsets(prob, th, MIN_DURATION_H)
        row = {"threshold": th, "n_det_raw": len(det)}
        for cat_name, cat in catalogs.items():
            r = match_cell(det, cat)
            for k in ("pod", "far", "n_det", "n_hit", "n_fa", "event_far",
                     "precision", "recall", "f1", "n_events_total", "n_events_tp",
                     "n_events_fa", "TP", "FP", "FN",
                     "onset_diff_h_median", "onset_diff_h_n"):
                row[f"{cat_name}_{k}"] = r[k]
        rows.append(row)
    return pd.DataFrame(rows)


def pick_operating_points(sweep_df: pd.DataFrame, cat_name: str) -> dict:
    """규칙 2가지: (1) event 단위 f1 최대, (2) event_far<=0.10 중 pod(=recall) 최대.
    event 단위 지표(event_far/precision/recall/f1)를 기준으로 삼음 -- 이 프로젝트가
    이미 event_FAR를 표준 FAR로 채택했기 때문(단순 per-detection far는 재검출 반복시
    낮아 보이는 착시가 있음, _match_core_poes.py 모듈 docstring 참고)."""
    valid = sweep_df.dropna(subset=[f"{cat_name}_f1"])
    out = {"max_f1": None, "far_le_010_max_pod": None}
    if not valid.empty:
        out["max_f1"] = valid.loc[valid[f"{cat_name}_f1"].idxmax()].to_dict()
    far_ok = valid[valid[f"{cat_name}_event_far"] <= 0.10]
    if not far_ok.empty:
        out["far_le_010_max_pod"] = far_ok.loc[far_ok[f"{cat_name}_pod"].idxmax()].to_dict()
    return out


# ── 그림 ─────────────────────────────────────────────────────────────────

def plot_pr_curve(sweep_df: pd.DataFrame, cat_names: list[str], ops: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    colors = {"noaa": "tab:blue", "manual": "tab:orange"}
    for cat_name in cat_names:
        valid = sweep_df.dropna(subset=[f"{cat_name}_precision", f"{cat_name}_recall"])
        ax.plot(valid[f"{cat_name}_recall"], valid[f"{cat_name}_precision"], "o-",
                color=colors.get(cat_name, "gray"), ms=3, lw=1, label=f"{cat_name}")
        for rule, marker in (("max_f1", "*"), ("far_le_010_max_pod", "s")):
            pt = ops[cat_name][rule]
            if pt is not None:
                ax.scatter([pt[f"{cat_name}_recall"]], [pt[f"{cat_name}_precision"]],
                          marker=marker, s=120, color=colors.get(cat_name, "gray"),
                          edgecolor="black", zorder=5,
                          label=f"{cat_name} {rule} (th={pt['threshold']})")
    ax.set_xlabel("recall (POD)")
    ax.set_ylabel("precision (event-level, =1-event_FAR)")
    ax.set_title("TCN precision-recall (threshold sweep)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[4_validation] 저장 -> {out_path}")


def plot_detector_overlay(event_id: int, ev_row: pd.Series, ts_primary: pd.DataFrame,
                          fsm_det: pd.DataFrame, tcn_det: pd.DataFrame | None,
                          out_dir: Path, pad_days: int = 3):
    onset, peak, end = ev_row["onset_time"], ev_row["peak_time"], ev_row["end_time"]
    t0 = onset - pd.Timedelta(days=pad_days)
    t1 = end + pd.Timedelta(days=pad_days)
    ts_win = ts_primary.loc[t0:t1]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    diag._shade_labels(ax, ts_win)
    ax.plot(ts_win.index, ts_win["zscore"], color="black", lw=0.8, label="z-score")
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    for t, color in ((onset, "green"), (peak, "red"), (end, "blue")):
        if pd.notna(t) and t0 <= t <= t1:
            ax.axvline(t, color=color, lw=1.5, alpha=0.9)

    fsm_win = fsm_det[(fsm_det["onset_time"] >= t0) & (fsm_det["onset_time"] <= t1)]
    for t in fsm_win["onset_time"]:
        ax.axvline(t, color="tab:blue", lw=1.3, ls="--", alpha=0.85)
    ax.plot([], [], color="tab:blue", ls="--", label=f"FSM onset ({len(fsm_win)})")

    if tcn_det is not None:
        tcn_win = tcn_det[(tcn_det["onset_time"] >= t0) & (tcn_det["onset_time"] <= t1)]
        for t in tcn_win["onset_time"]:
            ax.axvline(t, color="tab:purple", lw=1.3, ls="-.", alpha=0.85)
        ax.plot([], [], color="tab:purple", ls="-.", label=f"TCN onset ({len(tcn_win)})")

    ax.set_ylabel("z-score")
    ax.set_title(f"event {event_id}  onset={onset}  (shading=ground truth 3-state, "
                f"vlines=onset(green)/peak(red)/end(blue))", fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path = out_dir / f"event{event_id:03d}_{onset:%Y-%m-%d}.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


# ── 채널 구성 1개 처리 ────────────────────────────────────────────────────

def evaluate_channel_config(config_name: str, channels: list[str], tcn_run: str | None,
                            detector: str, catalogs: dict, thresholds: list[float],
                            out_dir: Path) -> tuple[list[dict], pd.DataFrame | None, dict]:
    print(f"\n{'#'*70}\n# 채널 구성: {config_name} ({channels})\n{'#'*70}")
    rows = []

    # -- FSM (기존 산출물, 재계산 없음) --
    fsm_det = load_fsm_onsets(channels)
    print(f"[4_validation] FSM raw onset {len(fsm_det)}개(channels={channels})")
    for cat_name, cat in catalogs.items():
        r = match_cell(fsm_det, cat)
        rows.append({"detector": "FSM", "channel_config": config_name, "ground_truth": cat_name,
                    "threshold": np.nan, "operating_point": "N/A(기존 w7k7 on0.1 pk0)", **r})

    # -- TCN (checkpoint 있을 때만) --
    tcn_result = None
    sweep_df = None
    if tcn_run is not None:
        loaded = load_tcn_ensemble(HERE / "results" / "runs" / tcn_run)
        if loaded is not None:
            models, manifest = loaded
            window = manifest["window"]
            X, time_idx, ts_primary = build_full_features(detector, channels, window)
            print(f"[4_validation] 전 구간 피처 {X.shape}, 시간범위 {time_idx.min()} ~ {time_idx.max()}")
            p_event = pd.Series(ensemble_event_proba(models, manifest, X), index=time_idx)

            sweep_df = sweep_thresholds(p_event, thresholds, catalogs)
            ops = {cat_name: pick_operating_points(sweep_df, cat_name) for cat_name in catalogs}

            for cat_name, cat in catalogs.items():
                for rule in ("max_f1", "far_le_010_max_pod"):
                    pt = ops[cat_name][rule]
                    if pt is None:
                        continue
                    th = pt["threshold"]
                    det = probs_to_onsets(p_event, th, MIN_DURATION_H)
                    r = match_cell(det, cat)
                    rows.append({"detector": "TCN", "channel_config": config_name,
                                "ground_truth": cat_name, "threshold": th,
                                "operating_point": rule, **r})
            tcn_result = {"p_event": p_event, "sweep_df": sweep_df, "ops": ops,
                         "ts_primary": ts_primary}
        else:
            for cat_name, cat in catalogs.items():
                rows.append({"detector": "TCN", "channel_config": config_name,
                            "ground_truth": cat_name, "threshold": np.nan,
                            "operating_point": "체크포인트 없음(건너뜀)",
                            "pod": np.nan, "far": np.nan, "n_det": np.nan, "n_hit": np.nan,
                            "n_fa": np.nan, "n_events_total": np.nan, "n_events_tp": np.nan,
                            "n_events_fa": np.nan, "event_far": np.nan, "TP": np.nan,
                            "FP": np.nan, "FN": np.nan, "precision": np.nan, "recall": np.nan,
                            "f1": np.nan})

    return rows, sweep_df, (tcn_result or {})


def main():
    ap = argparse.ArgumentParser(description="FSM vs TCN 동일조건 교차검증(기존 산출물 소비, 학습 없음)")
    ap.add_argument("--detector", default="metop03")
    ap.add_argument("--single-channels", default="omni_p6")
    ap.add_argument("--multi-channels", default="omni_p6,omni_p7,pro_tel0_p5")
    ap.add_argument("--single-run", default="omni_p6_binary",
                    help="predict_v0/runs/ 아래 단일채널 TCN run 이름(checkpoints 필요)")
    ap.add_argument("--multi-run", default="omni_p6_omni_p7_pro_tel0_p5_3class",
                    help="predict_v0/runs/ 아래 다채널 TCN run 이름(checkpoints 필요)")
    ap.add_argument("--thresholds", default=None, help="콤마구분 임계 목록(기본 0.05~0.95 step 0.05)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else HERE / "results" / "validation_v0"
    events_dir = out_dir / "event_overlay"
    events_dir.mkdir(parents=True, exist_ok=True)

    thresholds = ([float(t) for t in args.thresholds.split(",")]
                  if args.thresholds else DEFAULT_THRESHOLDS)

    print("[4_validation] 정답 카탈로그 로드 중")
    catalogs = {"noaa": load_noaa_catalog(), "manual": load_manual_catalog()}
    print(f"[4_validation] NOAA SPE {len(catalogs['noaa'])}개(ERA 필터 후), "
          f"손라벨(완결) {len(catalogs['manual'])}개")

    configs = [
        ("single", [c.strip() for c in args.single_channels.split(",")], args.single_run),
        ("multi", [c.strip() for c in args.multi_channels.split(",")], args.multi_run),
    ]

    all_rows = []
    tcn_results = {}
    for config_name, channels, tcn_run in configs:
        rows, sweep_df, tcn_result = evaluate_channel_config(
            config_name, channels, tcn_run, args.detector, catalogs, thresholds, out_dir)
        all_rows.extend(rows)
        if sweep_df is not None:
            sweep_df.to_csv(out_dir / f"threshold_sweep_{config_name}.csv", index=False)
            print(f"[4_validation] 저장 -> {out_dir / f'threshold_sweep_{config_name}.csv'}")
            plot_pr_curve(sweep_df, list(catalogs), tcn_result["ops"],
                         out_dir / f"pr_curve_{config_name}.png")
        tcn_results[config_name] = tcn_result

    comparison = pd.DataFrame(all_rows)
    comparison.to_csv(out_dir / "comparison_table.csv", index=False)
    print(f"\n[4_validation] 2x2(+연산점) 비교표 저장 -> {out_dir / 'comparison_table.csv'}")
    print(comparison.to_string(index=False))
    print("\n[4_validation] 주의: TCN x 손라벨 칸은 라벨 정의를 공유해 구조적으로 유리함 -- "
          "교차 칸(TCN x NOAA, FSM x 손라벨)이 더 공정한 비교.")

    # -- 대표 이벤트 오버레이 3장 (single 채널 TCN 결과가 있으면 그걸로, 없으면 FSM만) --
    single_tcn = tcn_results.get("single", {})
    windows_meta = te.load_windows(args.detector, [args.single_channels.split(",")[0]])
    events = pd.read_csv(HERE / "results" / "quality_check" / "events_reconciled.csv",
                        parse_dates=["onset_time", "peak_time", "end_time"])
    events = events.dropna(subset=["onset_time", "peak_time", "end_time"])
    reps = diag.pick_representatives(events, windows_meta)
    rep_ids = [v for v in reps.values() if v is not None]
    print(f"[4_validation] 대표 이벤트: {reps}")

    fsm_det_single = load_fsm_onsets([args.single_channels.split(",")[0]])
    tcn_det_single = None
    ts_primary_single = None
    if single_tcn:
        best = single_tcn["ops"]["noaa"]["max_f1"] or single_tcn["ops"]["manual"]["max_f1"]
        if best is not None:
            tcn_det_single = probs_to_onsets(single_tcn["p_event"], best["threshold"], MIN_DURATION_H)
            ts_primary_single = single_tcn["ts_primary"]
    if ts_primary_single is None:
        ts_primary_single = pd.read_parquet(
            HERE / "dataset_v0" / f"timeseries_{args.detector}_{args.single_channels.split(',')[0]}.parquet"
        ).set_index("time")

    saved = []
    for eid in rep_ids:
        row = events.loc[events["event_id"] == eid]
        if row.empty:
            continue
        p = plot_detector_overlay(eid, row.iloc[0], ts_primary_single, fsm_det_single,
                                  tcn_det_single, events_dir)
        saved.append(p)
    print(f"[4_validation] 대표 이벤트 오버레이 {len(saved)}장 저장 -> {events_dir}")
    print(f"\n[4_validation] 전체 산출물 -> {out_dir}")


if __name__ == "__main__":
    main()
