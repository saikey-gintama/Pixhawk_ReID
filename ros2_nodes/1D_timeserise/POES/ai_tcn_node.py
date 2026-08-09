"""
ai_tcn_node.py  --  게이트 추론 층, POES 온보드 파이프라인 S3
======================================================================
링버퍼 축적은 상시(저비용), TCN 추론은 게이트(ap_state>=PRE_ALERT)가 열려 있는
매 틱에만 한다 -- 논문 3.3절 "TCN 은 상시 구동이 아니다" 주장이 코드에서 보이도록
축적 경로(on_wp_results)와 추론 경로(on_sep_alert)를 완전히 분리한다.

배경/z 는 재계산하지 않는다. WP 가 /wp_results 에 실어 보낸 z 를 그대로 링버퍼에
쌓는다 -- "두 계층이 하나의 배경 추정을 공유한다"(논문 3.2절)가 이 노드의 존재
이유이고, 여기서 남는 유일한 신규 비용이 "윈도우 조립 + forward"(논문 4.3절
TCN 한계비용)다.

구독: /wp_results -> 채널별 z 를 ZBuffer(maxlen=14)에 append. 상시, 저비용.
      /sep_alert   -> ap_state 가 PRE_ALERT 이상이면 그 틱에 1회 추론.
발행: /ai_verdict (String JSON):
    {ts, ai_pub_ts, t_sample_rx, ap_state, status, p_event, y_pred, n_models, window_ok}
    -- t_sample_rx 는 WP 가 붙인 원 수신 벽시계를 AP 경유로 변경 없이 전달만 한다(S4 M10용).

추론 트리거 규칙(논문 4.3절 비용 모델과 일치): 게이트가 열려있는 매 틱마다 1회.
게이트 개방 "사건"당 1회가 아니다 -- "TCN 하루비용 = 하루 게이트개방 틱 수 x M9"
이라는 비용 모델이 이 규칙을 전제하므로, 전 구간 리플레이 총 추론 횟수는
AP 의 n_ticks_gate_open 과 정확히 같아야 한다(omni_p6: 16,204회).
결측 슬롯(/sep_alert 의 has_data=False)에서는 상태가 이전 값(PRE_ALERT/ALERT)을
그대로 유지하고 있어도 트리거하지 않는다 -- AP.n_ticks_gate_open 도 결측 슬롯을
빼고 세므로(S2), 여기서 상태만 보고 트리거하면 그 초과분만큼 16,204 를 넘겨버린다
(실측으로 발견된 버그, has_data 를 그대로 물려받아 고침).

윈도우 조립: 링버퍼는 오래된->최신 순(deque 오른쪽에 최신 append, list(deque)가
그대로 lag13..lag0 순서 -- 2_build_dataset._lag_cols 와 동일 배치, 재구현 아님).
채널은 manifest["channels"] 순서에 **이름으로** 매핑(위치 매핑 금지).
버퍼 미충족(<14) 또는 z 에 NaN 있으면 추론하지 않고 status=INSUFFICIENT_DATA 발행
-- 이 경우도 "게이트 열린 틱"으로는 카운트한다(추론 트리거 자체는 발생했으므로).

재사용(재구현 없음): predict_v0/tcn.py 의 TCNClassifier 를 그대로 import.
체크포인트 로드/추론 패턴은 predict_v0/4_validation.py:load_tcn_ensemble /
ensemble_event_proba 를 그대로 따른다(state_dict 로드 -> eval() -> softmax -> 1-P(quiet)).

측정(모두 rclpy 없이 부를 수 있는 함수/클래스에 있음 -- bench_micro 요구사항):
  zbuf_update_ms       : 링버퍼 append (상시 비용, on_wp_results 마다)
  M6 alert_transport_ms: ai 수신시각(wall) - /sep_alert 의 fsm_pub_ts
  M7 preproc_ms        : 윈도우 조립 + 텐서화(모델 호출 직전까지)
  M8 infer_ms          : forward 1회(단일 윈도우). ensemble 이면 모델별 시간도 별도 기록,
                          M8 자체는 전 모델 합계(순차 실행 비용).
  M9 activation_total_ms = M7 + M8(총) + 직렬화. 게이팅 발동 1회당 총비용.
  M8 과 M9 는 반드시 구분 기록(논문에서 다른 문장에 쓰임).
  워밍업(model_load_ms/warmup_ms/...)은 tcn_startup.json 에만 저장하고 per-tick
  CSV 에는 절대 섞지 않는다(첫 forward 의 lazy-init 지연이 집계를 오염시킴).
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    _HAVE_RCLPY = True
except ImportError:                       # Windows 개발 환경 -- 로직 단위 검증용
    _HAVE_RCLPY = False
    rclpy = None
    String = None

    class Node:                            # type: ignore -- 더미 베이스, 인스턴스화 안 함
        pass


def _default_repo() -> Path:
    if os.name == "nt":
        return Path("D:/VS_code/Pixhawk_ReID")
    return Path.home() / "jeongin" / "Pixhawk_ReID"


REPO = Path(os.environ.get("REPO", _default_repo()))
C_PD = REPO / "Experiment_window" / "C_PD"   # 기존 데이터·스크립트 -- 읽기 전용

sys.path.insert(0, str(C_PD / "predict_v0"))
from tcn import TCNClassifier   # noqa: E402  -- 재정의 금지, 그대로 import

RESULT_DIR = os.environ.get("RESULT_DIR", ".")

# ══════════════════════════════════════════════════════
# 파라미터 블록 -- 기본값(argparse 로 런타임 덮어쓰기, cFS TBL 철학)
# ══════════════════════════════════════════════════════
CHANNELS      = "omni_p6"
WINDOW        = 14
CKPT_ROOT     = Path(os.environ.get("CKPT_ROOT", str(C_PD / "predict_v0" / "results" / "runs")))
RUN_NAME      = "omni_p6_binary"
FOLDS_MODE    = "single"      # {single, ensemble}
FOLD          = 0
DEVICE        = "cpu"         # 고정 -- Z7030 GPU 없음
WARMUP_ITERS  = 20

STARTUP_JSON  = str(Path(RESULT_DIR) / "tcn_startup.json")


# ══════════════════════════════════════════════════════
# 모델 로드 -- predict_v0/4_validation.py:load_tcn_ensemble 과 동일 패턴(재사용)
# ══════════════════════════════════════════════════════
def load_single_model(ckpt_dir: Path, fold: int) -> tuple[torch.nn.Module, dict]:
    manifest = json.loads((ckpt_dir / "manifest.json").read_text(encoding="utf-8"))
    fp = ckpt_dir / f"fold{fold}.pt"
    model = TCNClassifier(input_size=manifest["input_size"],
                          num_channels=tuple(manifest["num_channels"]),
                          kernel_size=manifest["kernel_size"], dropout=manifest["dropout"],
                          n_classes=manifest["n_classes"])
    model.load_state_dict(torch.load(fp, map_location="cpu", weights_only=True))
    model.eval()
    return model, manifest


def load_ensemble_models(ckpt_dir: Path) -> tuple[list[torch.nn.Module], dict]:
    manifest = json.loads((ckpt_dir / "manifest.json").read_text(encoding="utf-8"))
    models = []
    for fold in range(manifest["n_folds"]):
        m, _ = load_single_model(ckpt_dir, fold)
        models.append(m)
    return models, manifest


def load_models(ckpt_dir: Path, folds_mode: str, fold: int) -> tuple[list[torch.nn.Module], dict]:
    if folds_mode == "ensemble":
        return load_ensemble_models(ckpt_dir)
    m, manifest = load_single_model(ckpt_dir, fold)
    return [m], manifest


# ══════════════════════════════════════════════════════
# 추론 -- predict_v0/4_validation.py:ensemble_event_proba 와 동일 공식(재사용)
# ══════════════════════════════════════════════════════
def infer_from_tensor(models: list[torch.nn.Module], manifest: dict,
                      xb: torch.Tensor) -> tuple[float, int, list[float]]:
    """xb 는 이미 조립+텐서화된 입력(M7 완료 후). 여기선 forward 만(M8 측정 대상).
    반환: p_event, y_pred, 모델별 forward 시간(ms) 목록."""
    quiet_idx = manifest["label_names"].index("quiet")
    probs = []
    per_model_ms = []
    with torch.no_grad():
        for m in models:
            t0 = perf_counter()
            p = torch.softmax(m(xb), dim=1).numpy()
            per_model_ms.append((perf_counter() - t0) * 1000.0)
            probs.append(p)
    ens = np.mean(probs, axis=0)
    p_event = float(1.0 - ens[0, quiet_idx])
    y_pred = int(np.argmax(ens[0]))
    return p_event, y_pred, per_model_ms


def warmup(models: list[torch.nn.Module], input_shape: tuple, n_iters: int = WARMUP_ITERS) -> list[float]:
    """더미 텐서로 n_iters 회 forward 후 폐기. 첫 forward 의 lazy-init 지연을
    tcn_startup.json 에만 격리(per-tick 집계에서 절대 제외)."""
    dummy = torch.zeros(input_shape, dtype=torch.float32)
    times = []
    with torch.no_grad():
        for _ in range(n_iters):
            t0 = perf_counter()
            for m in models:
                _ = m(dummy)
            times.append((perf_counter() - t0) * 1000.0)
    return times


# ══════════════════════════════════════════════════════
# 링버퍼 -- 채널별 최근 WINDOW개 z. 이름으로 조립(위치 매핑 금지).
# ══════════════════════════════════════════════════════
class ZBuffer:
    def __init__(self, channels: list[str], maxlen: int = WINDOW):
        self.channels = list(channels)
        self.maxlen = maxlen
        self.buf: dict[str, deque] = {ch: deque(maxlen=maxlen) for ch in self.channels}

    def update(self, channel: str, z) -> None:
        """z=None(WP 가 STALE 이라 못 보낸 틱)이면 아무 것도 안 한다 -- 버퍼를 동결한다.
        여기서 NaN 을 push 해버리면 방금까지 유효했던 가장 오래된 표본을 밀어내며
        멀쩡한 창을 깨뜨린다(run_len/counter 가 STALE 에 동결되는 것과 같은 원칙,
        md 4절 정합 항목 ③ 연장선 -- 실측으로 발견된 버그)."""
        if channel not in self.buf or z is None:
            return
        self.buf[channel].append(float(z))

    def is_ready(self) -> bool:
        for ch in self.channels:
            b = self.buf[ch]
            if len(b) < self.maxlen:
                return False
            if any(np.isnan(v) for v in b):
                return False
        return True

    def assemble(self) -> np.ndarray:
        """(1, WINDOW) 단일채널 / (1, n_ch, WINDOW) 다채널. 채널 순서는 self.channels(이름 매핑).
        float64 로 반환한다(float32 로 미리 캐스팅하면 검증 목표 1e-9 를 못 맞춘다 -- 모델
        입력용 float32 변환은 호출부가 torch.tensor(..., dtype=torch.float32) 에서 한다)."""
        rows = [list(self.buf[ch]) for ch in self.channels]   # 오래된->최신, 그대로 lag13..lag0
        arr = np.array(rows, dtype=np.float64)                 # (n_ch, WINDOW)
        if len(self.channels) == 1:
            return arr.reshape(1, -1)
        return arr.reshape(1, len(self.channels), -1)


# ══════════════════════════════════════════════════════
# AI 핵심 로직 -- rclpy 무관. Node 는 이 클래스를 감쌀 뿐(md 2절, bench_micro 요구사항).
# ══════════════════════════════════════════════════════
class AiTcnCore:
    def __init__(self, models: list[torch.nn.Module], manifest: dict, channels: list[str],
                window: int = WINDOW):
        self.models = models
        self.manifest = manifest
        self.zbuf = ZBuffer(channels, maxlen=window)
        self.n_inference_calls = 0    # 게이트 열린 틱마다 1(창 준비 여부와 무관)
        self.n_ok = 0
        self.n_insufficient = 0

    def update_buffer(self, channel: str, z) -> float:
        """zbuf_update_ms(상시 비용) 반환."""
        t0 = perf_counter()
        self.zbuf.update(channel, z)
        return (perf_counter() - t0) * 1000.0

    def maybe_infer(self, ap_state: str, has_data: bool = True) -> dict | None:
        """게이트 닫힘(NOMINAL)이거나 이 틱이 결측 슬롯(has_data=False)이면 None(추론
        카운트도 안 함) -- AP 의 n_ticks_gate_open 과 정확히 같은 집합에서만 추론해야
        전 구간 총 추론 횟수(16,204)가 논문 4.3절 비용 모델의 분자와 일치한다. 결측
        슬롯은 카운터가 동결돼 상태만 이전 값을 유지할 뿐 이 틱 자체엔 새 정보가 없다
        (실측으로 발견 -- AP.has_data 를 안 보고 상태만 보면 결측 중에도 트리거돼
        16,204 보다 많이 나온다). 게이트 열림이면 매 틱 1회 추론 시도 -- 창 미충족/NaN
        이면 INSUFFICIENT_DATA 로 status 만 반환(그래도 카운트)."""
        if ap_state not in ("PRE_ALERT", "ALERT") or not has_data:
            return None
        self.n_inference_calls += 1

        t7 = perf_counter()
        window_ok = self.zbuf.is_ready()
        xb = torch.tensor(self.zbuf.assemble(), dtype=torch.float32) if window_ok else None
        preproc_ms = (perf_counter() - t7) * 1000.0

        if not window_ok:
            self.n_insufficient += 1
            return {"status": "INSUFFICIENT_DATA", "p_event": None, "y_pred": None,
                    "window_ok": False, "preproc_ms": preproc_ms, "infer_ms": 0.0,
                    "per_model_ms": []}

        t8 = perf_counter()
        p_event, y_pred, per_model_ms = infer_from_tensor(self.models, self.manifest, xb)
        infer_ms = (perf_counter() - t8) * 1000.0
        self.n_ok += 1
        return {"status": "OK", "p_event": p_event, "y_pred": y_pred, "window_ok": True,
                "preproc_ms": preproc_ms, "infer_ms": infer_ms, "per_model_ms": per_model_ms}


# ══════════════════════════════════════════════════════
# bench 훅 -- bench_micro.py 가 rclpy/노드 없이 직접 호출 (input_size 스윕 M8 전용)
# ══════════════════════════════════════════════════════
_BENCH_CKPT_BY_INPUT_SIZE = {
    1: "omni_p6_binary",
    3: "omni_p6_pro_tel0_p5_pro_tel90_p5_binary",
}


def bench_model_for_input_size(input_size: int, ckpt_root: Path = CKPT_ROOT,
                               num_channels=(16, 16, 16), kernel_size: int = 3,
                               dropout: float = 0.2, n_classes: int = 2) -> tuple[torch.nn.Module, bool]:
    """input_size(1/3/5) 에 맞는 체크포인트가 있으면 로드(실가중치), 없으면 랜덤
    초기화(두번째 반환값 True="랜덤 가중치, 타이밍 전용"). 없는 체크포인트를
    찾으러 다니지 않는다 -- 존재하는 1/3채널 run 후보만 시도."""
    run_name = _BENCH_CKPT_BY_INPUT_SIZE.get(input_size)
    ckpt_dir = (ckpt_root / run_name / "checkpoints") if run_name else None
    if ckpt_dir is not None and (ckpt_dir / "manifest.json").exists():
        model, _ = load_single_model(ckpt_dir, fold=0)
        return model, False
    model = TCNClassifier(input_size=input_size, num_channels=num_channels,
                          kernel_size=kernel_size, dropout=dropout, n_classes=n_classes)
    model.eval()
    return model, True   # 랜덤 가중치, 타이밍 전용


# ──────────────────────────────────────────────────────
# AiTcnNode -- rclpy 없으면 인스턴스화만 못 함(클래스 정의 자체는 항상 가능)
# ──────────────────────────────────────────────────────
class AiTcnNode(Node):
    def __init__(self):
        super().__init__("ai_tcn_node")
        if String is None:
            raise RuntimeError("rclpy/std_msgs 없음 -- Jetson(ROS2 환경)에서 실행할 것")

        self.sub_wp = self.create_subscription(String, "/wp_results", self.on_wp_results, 10)
        self.sub_alert = self.create_subscription(String, "/sep_alert", self.on_sep_alert, 10)
        self.pub = self.create_publisher(String, "/ai_verdict", 10)

        torch.set_num_threads(1)
        ckpt_dir = CKPT_ROOT / RUN_NAME / "checkpoints"
        t0 = perf_counter()
        models, manifest = load_models(ckpt_dir, FOLDS_MODE, FOLD)
        model_load_ms = (perf_counter() - t0) * 1000.0

        # 채널 순서는 manifest["channels"] 가 유일한 진실의 원천이다 -- 모델의 conv 입력
        # 축(input_size)이 그 순서로 학습됐다. --channels CLI 인자로 링버퍼 순서를
        # 정하면 그 값이 manifest 와 어긋나는 순간(오타/실수) 조용히 잘못된 축으로
        # 텐서가 조립된다(S5.2 에서 지적된 위험, 실제로 여기 있었다 -- 고침).
        # --channels 는 "이 run 이 몇/어떤 채널을 쓰려 했는지" 선언만 남기고,
        # 실제 조립 순서는 항상 manifest 를 따른다.
        cli_channels = [c.strip() for c in CHANNELS.split(",") if c.strip()]
        self.channels = list(manifest["channels"])
        if cli_channels and cli_channels != self.channels:
            self.get_logger().warn(
                f"[AI] --channels={cli_channels} 가 manifest 채널 순서 {self.channels} 와 "
                f"다름 -- manifest 를 따른다(무시된 것은 --channels 쪽).")

        input_shape = (1, WINDOW) if len(self.channels) == 1 else (1, len(self.channels), WINDOW)
        warmup_ms_list = warmup(models, input_shape, WARMUP_ITERS)

        startup = {
            "model_load_ms": round(model_load_ms, 4),
            "warmup_ms": [round(w, 4) for w in warmup_ms_list],
            "warmup_iters": WARMUP_ITERS,
            "torch_threads": torch.get_num_threads(),
            "device": DEVICE,
            "folds_mode": FOLDS_MODE, "fold": FOLD, "n_models": len(models),
            "channels": self.channels, "manifest": manifest,
        }
        with open(STARTUP_JSON, "w", encoding="utf-8") as f:
            json.dump(startup, f, indent=2)
        self.get_logger().info(
            f"[AI] startup 저장 -> {STARTUP_JSON}: model_load={model_load_ms:.1f}ms "
            f"warmup x{WARMUP_ITERS} torch_threads={torch.get_num_threads()} "
            f"folds_mode={FOLDS_MODE} n_models={len(models)}")

        self.core = AiTcnCore(models, manifest, self.channels, WINDOW)

        log_path = os.path.join(RESULT_DIR, "ai_log.csv")
        import csv
        self.csv = open(log_path, "w", newline="")
        self.writer = csv.writer(self.csv)
        self.writer.writerow([
            "phys_ts", "ai_pub_wall_ts", "ap_state", "status",
            "alert_transport_ms", "preproc_ms", "infer_ms", "activation_total_ms",
            "per_model_ms", "n_models", "p_event", "y_pred",
        ])

        # zbuf_log.csv -- 상시 비용(zbuf_update_ms) per-tick 기록. WP 의 모든 틱에서
        # on_wp_results 가 돈다(게이트 상태 무관) -- log.csv(AP 소유, M1~M5) 와는 별개
        # 파일이다: 서로 다른 두 OS 프로세스(AP/AI)가 같은 파일에 동시에 쓰면 위험해
        # 나눴다. 두 파일 다 "ts" 로 조인 가능(md 4절 취지 연장, S7 집계에서 join).
        zbuf_log_path = os.path.join(RESULT_DIR, "zbuf_log.csv")
        self.zbuf_csv = open(zbuf_log_path, "w", newline="")
        self.zbuf_writer = csv.writer(self.zbuf_csv)
        self.zbuf_writer.writerow(["phys_ts", "n_channels", "zbuf_update_ms"])

        self.marker_path = os.path.join(RESULT_DIR, "REPLAY_DONE")
        self.marker_timer = self.create_timer(1.0, self._check_marker)
        self._done = False

        self.get_logger().info(
            f"[AI] ready. channels={self.channels} window={WINDOW} device={DEVICE}. "
            f"Waiting for /wp_results, /sep_alert ...")

    def on_wp_results(self, msg):
        payload = json.loads(msg.data)
        phys_ts = payload.get("ts", 0.0)
        results = payload.get("results", [])
        zbuf_update_ms_total = 0.0
        for r in results:
            zbuf_update_ms_total += self.core.update_buffer(r["channel"], r.get("z"))
        self.zbuf_writer.writerow([phys_ts, len(results), round(zbuf_update_ms_total, 5)])
        self.zbuf_csv.flush()

    def on_sep_alert(self, msg):
        t_recv_wall = time.time()
        payload = json.loads(msg.data)
        ap_state = payload.get("ap_state", "NOMINAL")
        phys_ts = payload.get("ts", 0.0)
        fsm_pub_ts = payload.get("fsm_pub_ts")
        has_data = payload.get("has_data", True)
        t_sample_rx = payload.get("t_sample_rx")   # WP 원 수신 벽시계, 변경 없이 전달(S4 M10용)
        alert_transport_ms = (t_recv_wall - fsm_pub_ts) * 1000.0 if fsm_pub_ts else float("nan")

        out = self.core.maybe_infer(ap_state, has_data)
        if out is None:
            return   # 게이트 닫힘 -- 추론 시도조차 없음(CSV 도 안 남김)

        verdict = {
            "ts": phys_ts, "ai_pub_ts": time.time(), "t_sample_rx": t_sample_rx,
            "ap_state": ap_state,
            "status": out["status"], "p_event": out["p_event"], "y_pred": out["y_pred"],
            "n_models": len(self.core.models), "window_ok": out["window_ok"],
        }
        t_ser = perf_counter()
        vmsg = String()
        vmsg.data = json.dumps(verdict)
        serialize_ms = (perf_counter() - t_ser) * 1000.0
        self.pub.publish(vmsg)

        activation_total_ms = out["preproc_ms"] + out["infer_ms"] + serialize_ms
        self.writer.writerow([
            phys_ts, time.time(), ap_state, out["status"],
            round(alert_transport_ms, 3) if alert_transport_ms == alert_transport_ms else "",
            round(out["preproc_ms"], 4), round(out["infer_ms"], 4),
            round(activation_total_ms, 4),
            json.dumps([round(x, 4) for x in out["per_model_ms"]]),
            len(self.core.models), out["p_event"], out["y_pred"],
        ])
        self.csv.flush()

    def _check_marker(self):
        if self._done or not os.path.exists(self.marker_path):
            return
        self._done = True
        self.get_logger().info(
            f"[AI] REPLAY_DONE marker 감지 -> 종료 (n_inference_calls={self.core.n_inference_calls} "
            f"n_ok={self.core.n_ok} n_insufficient={self.core.n_insufficient})")
        self.csv.close()
        self.zbuf_csv.close()
        rclpy.shutdown()

    def destroy_node(self):
        if not self.csv.closed:
            self.csv.close()
        if not self.zbuf_csv.closed:
            self.zbuf_csv.close()
        super().destroy_node()


def _parse_args(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="ai_tcn_node -- 게이트 TCN 추론")
    p.add_argument("--channels", type=str, default=CHANNELS)
    p.add_argument("--window", type=int, default=WINDOW)
    p.add_argument("--ckpt-root", type=str, default=str(CKPT_ROOT))
    p.add_argument("--run-name", type=str, default=RUN_NAME)
    p.add_argument("--folds-mode", choices=["single", "ensemble"], default=FOLDS_MODE)
    p.add_argument("--fold", type=int, default=FOLD)
    p.add_argument("--device", type=str, default=DEVICE, choices=["cpu"])
    p.add_argument("--warmup-iters", type=int, default=WARMUP_ITERS)
    p.add_argument("--startup-json", type=str, default=STARTUP_JSON)
    return p.parse_known_args(argv)[0]


def main(argv=None):
    args = _parse_args(argv)

    global CHANNELS, WINDOW, CKPT_ROOT, RUN_NAME, FOLDS_MODE, FOLD, DEVICE, WARMUP_ITERS, STARTUP_JSON
    CHANNELS      = args.channels
    WINDOW        = args.window
    CKPT_ROOT     = Path(args.ckpt_root)
    RUN_NAME      = args.run_name
    FOLDS_MODE    = args.folds_mode
    FOLD          = args.fold
    DEVICE        = args.device
    WARMUP_ITERS  = args.warmup_iters
    STARTUP_JSON  = args.startup_json

    if not _HAVE_RCLPY:
        raise SystemExit("[AI] rclpy 없음 -- 이 노드 실행은 Jetson(ROS2 환경)에서만 가능. "
                         "로직 단위 검증은 AiTcnCore/load_models/infer_from_tensor 를 "
                         "직접 import 해서 할 것.")

    rclpy.init()
    node = AiTcnNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
