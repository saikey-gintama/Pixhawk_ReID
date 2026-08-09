"""
bench_micro.py
================
S6 -- 노드를 띄우지 않고 판정 함수를 직접 호출해 순수 연산 비용만 측정한다.
M2 는 마이크로초 단위라 DDS·직렬화가 섞이면 못 잰다(serialize_ms 자체는 예외 --
JSON 직렬화 비용을 따로 격리해서 재는 게 목적이라 여기 포함).

S5(노드 리플레이)가 "실제 환경에서의 지연"을 재는 것과 이건 대체 관계가 아니라
서로 다른 값이다 -- 여기는 "연산 자체의 비용"만 잰다.

재사용(재구현 없음, 노드 파일에서 그대로 import):
  wp_poes_node.py : resample_window(M1), compute_z(M2), compute_bg_stats(M4)
  ap_fsm_node.py  : ApFsmCore.process_tick(M3, RPN+카운터+상태전이 전부)
  ai_tcn_node.py  : ZBuffer(M7/zbuf_update_ms), infer_from_tensor(M8),
                    bench_model_for_input_size(input_size 스윕), load_ensemble_models

입력은 합성 난수가 아니라 실제 데이터를 쓴다 -- omni_p6 실제 카운트/배경/z 로 짧은
리플레이(prepare_real_data)를 한 번 돌려 각 벤치가 순환할 실측 표본을 만든다.
난수는 분기 예측·캐시 거동이 실제와 달라진다(md 지시).

통계는 median/p95/min 만 보고한다. mean 은 스케줄러 이상치에 끌리므로 안 쓴다.

실행 환경 고정(전부 출력에 기록):
  torch.set_num_threads(1), os.sched_setaffinity(0,{N})(가능하면, Windows 는 없음),
  time.perf_counter_ns()(time.time() 아님), 워밍업 제외(항목별 100회, M8 은 20회).
  jetson_clocks/nvpmodel 상태 조회 -- 고정 못 하면(비-Jetson 등) 그 사실을 명시.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wp_poes_node as wp   # noqa: E402
import ap_fsm_node as ap    # noqa: E402
import ai_tcn_node as ai    # noqa: E402

try:
    import resource as _resource   # POSIX 전용 -- Windows 에는 없음
    _HAVE_RESOURCE = True
except ImportError:
    _resource = None
    _HAVE_RESOURCE = False

RESULT_DIR = os.environ.get("RESULT_DIR", None)

# ── 반복 수 / 워밍업 (md 지시 그대로, --smoke 로 축소 가능) ──
REPEATS = {
    "M1_resample_ms": 10_000, "M2_z_eval_ms": 100_000, "M3_fsm_eval_ms": 100_000,
    "M4_bg_update_ms": 10_000, "M7_preproc_ms": 10_000, "M8_infer_ms": 1_000,
    "zbuf_update_ms": 100_000, "serialize_ms": 10_000,
}
WARMUP = {
    "M1_resample_ms": 100, "M2_z_eval_ms": 100, "M3_fsm_eval_ms": 100,
    "M4_bg_update_ms": 100, "M7_preproc_ms": 100, "M8_infer_ms": 20,
    "zbuf_update_ms": 100, "serialize_ms": 100,
}

CHANNEL = "omni_p6"
BG_WINDOW_DAYS = 7
K, ONSET_FLOOR, GATE_N, ALERT_N = 7.0, 0.1, 2, 4
WINDOW = 14


# ══════════════════════════════════════════════════════
# 실행 환경 고정
# ══════════════════════════════════════════════════════
def pin_single_core(core: int = 0) -> dict:
    """가능하면 단일 코어 고정(Linux 만). 반환: {"pinned":bool,"core":int,"reason":str|None}."""
    if hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {core})
            return {"pinned": True, "core": core, "reason": None}
        except Exception as e:
            return {"pinned": False, "core": core, "reason": str(e)}
    return {"pinned": False, "core": core, "reason": "os.sched_setaffinity 없음(비-Linux, 예: Windows)"}


def query_clock_status() -> dict:
    """nvpmodel/jetson_clocks 상태 조회. 없으면(비-Jetson) 그 사실을 명시.
    DVFS 가 살아있으면 p95 가 연산이 아니라 주파수 전환을 재게 되므로 반드시 기록."""
    info = {"nvpmodel_mode": None, "jetson_clocks_status": None,
           "nvpmodel_available": False, "jetson_clocks_available": False}
    if shutil.which("nvpmodel"):
        info["nvpmodel_available"] = True
        try:
            out = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=5)
            info["nvpmodel_mode"] = out.stdout.strip()
        except Exception as e:
            info["nvpmodel_mode"] = f"조회 실패: {e}"
    else:
        info["nvpmodel_mode"] = "nvpmodel 없음(비-Jetson 환경) -- 클럭 고정 여부 확인 불가"
    if shutil.which("jetson_clocks"):
        info["jetson_clocks_available"] = True
        try:
            out = subprocess.run(["jetson_clocks", "--show"], capture_output=True, text=True, timeout=5)
            info["jetson_clocks_status"] = out.stdout.strip()
        except Exception as e:
            info["jetson_clocks_status"] = f"조회 실패: {e}"
    else:
        info["jetson_clocks_status"] = "jetson_clocks 없음(비-Jetson 환경) -- DVFS 고정 못 함, p95 해석 시 주의"
    return info


# ══════════════════════════════════════════════════════
# 벤치 하네스 -- median/p95/min, perf_counter_ns, 워밍업 제외
# ══════════════════════════════════════════════════════
def bench_call(fn, n_iter: int, n_warmup: int) -> dict:
    for _ in range(n_warmup):
        fn()
    times_ns = np.empty(n_iter, dtype=np.int64)
    for i in range(n_iter):
        t0 = perf_counter_ns()
        fn()
        times_ns[i] = perf_counter_ns() - t0
    times_ms = times_ns / 1e6
    return {
        "n_iter": n_iter, "n_warmup_excluded": n_warmup,
        "median_ms": float(np.median(times_ms)), "p95_ms": float(np.percentile(times_ms, 95)),
        "min_ms": float(np.min(times_ms)),
    }


# ══════════════════════════════════════════════════════
# 실 데이터 준비 (합성 난수 금지) -- omni_p6 실측으로 짧은 리플레이
# ══════════════════════════════════════════════════════
def prepare_real_data(cache_dir: Path, n_ticks: int = 3000) -> dict:
    """실제 omni_p6 리플레이(재사용, 재구현 아님)로 각 벤치가 순환할 실측 표본을 만든다.
    n_ticks=3000(~31일)이면 배경 워밍업(7일=672틱) 이후 ~2300틱의 유효(z 유한) 표본이
    남는다."""
    raw = wp.load_raw_channel(cache_dir, CHANNEL)
    schedule = wp.build_tick_schedule(raw)[:n_ticks]
    cadence = pd.Timedelta(seconds=900)

    cs = wp.ChannelState(CHANNEL, bg_window_days=BG_WINDOW_DAYS)
    adt = ap.build_adt([CHANNEL], ap.build_rpn_equation([CHANNEL]), GATE_N, ALERT_N, "RTS_SEP_ALERT")[0]
    ap_core = ap.ApFsmCore(adt, dry_run=False)

    counts, bg_medians, bg_stds, zs, watches, ts_list = [], [], [], [], [], []
    for ts in schedule:
        count = wp.resample_window(raw, ts, cadence)
        cs.push_sample(ts, count)
        cs.maybe_update_bg(ts)
        res = cs.eval_watchpoint(count, ts, K, ONSET_FLOOR, GATE_N, ALERT_N)
        ap_core.process_tick({CHANNEL: res["watch"]}, {CHANNEL: res["count"]}, ts.timestamp())

        counts.append(res["count"] if res["count"] is not None else float("nan"))
        bg_medians.append(res["bg_median"] if res["bg_median"] is not None else float("nan"))
        bg_stds.append(res["bg_std"] if res["bg_std"] is not None else float("nan"))
        zs.append(res["z"] if res["z"] is not None else float("nan"))
        watches.append(res["watch"])
        ts_list.append(ts)

    counts_a = np.array(counts, dtype=float)
    bg_med_a = np.array(bg_medians, dtype=float)
    bg_std_a = np.array(bg_stds, dtype=float)
    z_a = np.array(zs, dtype=float)

    valid = np.isfinite(counts_a) & np.isfinite(bg_med_a) & np.isfinite(bg_std_a)
    n_valid = int(valid.sum())
    if n_valid < 20:
        raise SystemExit(f"[bench] 유효 표본 {n_valid}개뿐 -- n_ticks 를 늘릴 것")

    valid_z = z_a[np.isfinite(z_a)]

    # M4 용 실제 672(=BG_WINDOW_DAYS*96) 표본 트레일링 버퍼 -- 리플레이 종료 시점 버퍼 그대로.
    bg_buf_672 = np.array([v for _, v in cs.buf], dtype=float)
    if len(bg_buf_672) < 2:
        raise SystemExit("[bench] 배경 버퍼 표본 부족 -- n_ticks 를 늘릴 것")

    return {
        "raw": raw, "schedule": schedule, "cadence": cadence,
        "counts_valid": counts_a[valid], "bg_medians_valid": bg_med_a[valid],
        "bg_stds_valid": bg_std_a[valid], "z_valid": valid_z,
        "bg_buf": bg_buf_672,
        "watches": watches, "ts_list": ts_list, "counts_all": counts,
        "n_valid": n_valid,
    }


# ══════════════════════════════════════════════════════
# M1 resample_ms -- 1분 원시 15개 누적 -> 평균
# ══════════════════════════════════════════════════════
def bench_m1(data: dict) -> dict:
    raw, schedule, cadence = data["raw"], data["schedule"], data["cadence"]
    idx_cycle = itertools.cycle(range(len(schedule)))

    def call():
        i = next(idx_cycle)
        return wp.resample_window(raw, schedule[i], cadence)

    return bench_call(call, REPEATS["M1_resample_ms"], WARMUP["M1_resample_ms"])


# ══════════════════════════════════════════════════════
# M2 z_eval_ms -- (c-m)/max(sigma,eps) + clip
# ══════════════════════════════════════════════════════
def bench_m2(data: dict) -> dict:
    c, m, s = data["counts_valid"], data["bg_medians_valid"], data["bg_stds_valid"]
    n = len(c)
    idx_cycle = itertools.cycle(range(n))

    def call():
        i = next(idx_cycle)
        return wp.compute_z(c[i], m[i], s[i])

    return bench_call(call, REPEATS["M2_z_eval_ms"], WARMUP["M2_z_eval_ms"])


# ══════════════════════════════════════════════════════
# M3 fsm_eval_ms -- RPN 결합 + 카운터 + 상태전이 (ApFsmCore.process_tick 그대로)
# ══════════════════════════════════════════════════════
def bench_m3(data: dict) -> dict:
    adt = ap.build_adt([CHANNEL], ap.build_rpn_equation([CHANNEL]), GATE_N, ALERT_N, "RTS_SEP_ALERT")[0]
    core = ap.ApFsmCore(adt, dry_run=False)
    watches, ts_list = data["watches"], data["ts_list"]
    n = len(watches)
    idx_cycle = itertools.cycle(range(n))

    def call():
        i = next(idx_cycle)
        return core.process_tick({CHANNEL: watches[i]}, {CHANNEL: data["counts_all"][i]},
                                 ts_list[i].timestamp())

    return bench_call(call, REPEATS["M3_fsm_eval_ms"], WARMUP["M3_fsm_eval_ms"])


# ══════════════════════════════════════════════════════
# M4 bg_update_ms -- 실제 672표본 median + MAD
# ══════════════════════════════════════════════════════
def bench_m4(data: dict) -> dict:
    vals = data["bg_buf"]

    def call():
        return wp.compute_bg_stats(vals)

    return bench_call(call, REPEATS["M4_bg_update_ms"], WARMUP["M4_bg_update_ms"])


# ══════════════════════════════════════════════════════
# M7 preproc_ms -- 링버퍼 -> (1,1,14) 텐서 조립(단일채널 기준)
# ══════════════════════════════════════════════════════
def bench_m7(data: dict) -> dict:
    z_valid = data["z_valid"]
    zbuf = ai.ZBuffer([CHANNEL], maxlen=WINDOW)
    for v in z_valid[:WINDOW]:
        zbuf.update(CHANNEL, v)

    def call():
        X = zbuf.assemble()
        return torch.tensor(X, dtype=torch.float32)

    return bench_call(call, REPEATS["M7_preproc_ms"], WARMUP["M7_preproc_ms"])


# ══════════════════════════════════════════════════════
# zbuf_update_ms -- 링버퍼 append(상시 비용)
# ══════════════════════════════════════════════════════
def bench_zbuf_update(data: dict) -> dict:
    zbuf = ai.ZBuffer([CHANNEL], maxlen=WINDOW)
    z_valid = data["z_valid"]
    n = len(z_valid)
    idx_cycle = itertools.cycle(range(n))

    def call():
        i = next(idx_cycle)
        zbuf.update(CHANNEL, float(z_valid[i]))

    return bench_call(call, REPEATS["zbuf_update_ms"], WARMUP["zbuf_update_ms"])


# ══════════════════════════════════════════════════════
# serialize_ms -- verdict JSON 직렬화 (M9 구성요소)
# ══════════════════════════════════════════════════════
def bench_serialize(data: dict) -> dict:
    verdict_template = {
        "ts": 0.0, "ai_pub_ts": 0.0, "t_sample_rx": 0.0, "ap_state": "ALERT",
        "status": "OK", "p_event": 0.5, "y_pred": 1, "n_models": 1, "window_ok": True,
    }
    counter = itertools.count()

    def call():
        v = dict(verdict_template)
        v["ts"] = float(next(counter))
        return json.dumps(v)

    return bench_call(call, REPEATS["serialize_ms"], WARMUP["serialize_ms"])


# ══════════════════════════════════════════════════════
# M8 infer_ms -- TCN forward 1회 (실 z 기반 실 1채널 체크포인트)
# ══════════════════════════════════════════════════════
def bench_m8_real(data: dict, ckpt_root: Path) -> dict:
    model, manifest = ai.load_single_model(ckpt_root / "omni_p6_binary" / "checkpoints", fold=0)
    z_valid = data["z_valid"]
    n = len(z_valid)
    if n < WINDOW:
        raise SystemExit(f"[bench] z 유효표본 {n}개 < WINDOW({WINDOW}) -- n_ticks 를 늘릴 것")

    idx_cycle = itertools.cycle(range(n - WINDOW))
    with torch.no_grad():
        def call():
            i = next(idx_cycle)
            xb = torch.tensor(z_valid[i:i + WINDOW].reshape(1, -1).astype(np.float32), dtype=torch.float32)
            return model(xb)

        return bench_call(call, REPEATS["M8_infer_ms"], WARMUP["M8_infer_ms"])


# ══════════════════════════════════════════════════════
# input_size 스윕: {1,3,5} x {single,ensemble} -- 더미 텐서, 모양만 좌우
# ══════════════════════════════════════════════════════
def bench_infer_sweep(ckpt_root: Path, n_iter: int = 1000, n_warmup: int = 20) -> list[dict]:
    """체크포인트 있는 1/3채널은 실가중치, 5채널은 랜덤 초기화("랜덤 가중치, 타이밍
    전용"). 비용은 텐서 모양에만 좌우되므로 더미 입력으로 충분(md 지시)."""
    rows = []
    for input_size in (1, 3, 5):
        base_model, is_random = ai.bench_model_for_input_size(input_size, ckpt_root=ckpt_root)
        dummy = torch.zeros((1, input_size, WINDOW) if input_size > 1 else (1, WINDOW),
                            dtype=torch.float32)

        for folds_mode in ("single", "ensemble"):
            if folds_mode == "single":
                models = [base_model]
            else:
                run_name = ai._BENCH_CKPT_BY_INPUT_SIZE.get(input_size)
                ckpt_dir = (ckpt_root / run_name / "checkpoints") if run_name else None
                if (not is_random) and ckpt_dir is not None and (ckpt_dir / "manifest.json").exists():
                    models, _ = ai.load_ensemble_models(ckpt_dir)
                else:
                    models = [base_model] * 5   # 랜덤 가중치 재사용 -- 타이밍은 모양만 좌우

            with torch.no_grad():
                def call(models=models):
                    for m in models:
                        m(dummy)

                stats = bench_call(call, n_iter, n_warmup)
            stats.update({"input_size": input_size, "folds_mode": folds_mode,
                         "n_models": len(models), "random_weights": is_random})
            rows.append(stats)
            label = "랜덤 가중치, 타이밍 전용" if is_random else "실 체크포인트"
            print(f"[bench] infer_sweep input_size={input_size} folds_mode={folds_mode} "
                 f"n_models={len(models)} ({label}) median={stats['median_ms']:.4f}ms "
                 f"p95={stats['p95_ms']:.4f}ms")
    return rows


# ══════════════════════════════════════════════════════
# 버스트 모드 -- 추론/FSM 1회당 에너지 (tegrastats 없으면 우아하게 스킵)
# ══════════════════════════════════════════════════════
def has_tegrastats() -> bool:
    return shutil.which("tegrastats") is not None


_VDD_IN_RE = re.compile(r"VDD_IN (\d+)mW")


def _read_tegrastats_power(log_path: Path) -> list[float]:
    if not log_path.exists():
        return []
    powers = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = _VDD_IN_RE.search(line)
        if m:
            powers.append(float(m.group(1)))
    return powers


def _cpu_seconds() -> float:
    if not _HAVE_RESOURCE:
        return float("nan")
    u = _resource.getrusage(_resource.RUSAGE_SELF)
    return u.ru_utime + u.ru_stime


def _run_tegra_window(fn, duration_s: float, log_path: Path, interval_ms: int = 1000) -> dict:
    """fn 을 duration_s 초 이상 반복 실행하며 tegrastats 로 전력을 동시에 잰다.
    반환: {"n_calls", "duration_s", "cpu_seconds", "power_samples_mw"}."""
    proc = subprocess.Popen(["tegrastats", "--interval", str(interval_ms), "--logfile", str(log_path)])
    time.sleep(1.0)   # tegrastats 자체 기동 안정화
    cpu0 = _cpu_seconds()
    t0 = time.time()
    n_calls = 0
    while (time.time() - t0) < duration_s:
        fn()
        n_calls += 1
    actual_duration = time.time() - t0
    cpu1 = _cpu_seconds()
    time.sleep(1.0)
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
    powers = _read_tegrastats_power(log_path)
    return {"n_calls": n_calls, "duration_s": actual_duration,
           "cpu_seconds": (cpu1 - cpu0) if _HAVE_RESOURCE else float("nan"),
           "power_samples_mw": powers}


def run_burst(target_fn, label: str, out_dir: Path, pilot_iters: int = 1000,
             min_duration_s: float = 60.0) -> dict | None:
    """① idle 60s -> P_idle. ② 파일럿(pilot_iters)으로 1회 시간 추정.
    ③ n_iter=ceil(min_duration_s/t), 최소 min_duration_s 보장. ④ burst 실행 -> P_burst.
    ⑤ dP=P_burst-P_idle, energy_per_op=dP*median_ms(파일럿). ⑥ 표본 수 전부 기록.
    tegrastats 없으면 None(우아한 스킵)."""
    if not has_tegrastats():
        print(f"[bench] tegrastats 없음 -- 버스트 모드({label}) 스킵")
        return None

    idle_log = out_dir / f"tegrastats_idle_{label}.log"
    print(f"[bench] [{label}] idle {min_duration_s:.0f}s 측정 -> P_idle")
    idle = _run_tegra_window(lambda: time.sleep(0.05), min_duration_s, idle_log)
    p_idle = float(np.median(idle["power_samples_mw"])) if idle["power_samples_mw"] else float("nan")

    print(f"[bench] [{label}] 파일럿 {pilot_iters}회로 1회당 시간 추정")
    t0 = perf_counter_ns()
    for _ in range(pilot_iters):
        target_fn()
    pilot_ms_per_call = ((perf_counter_ns() - t0) / 1e6) / pilot_iters
    n_iter = max(pilot_iters, math.ceil((min_duration_s * 1000.0) / max(pilot_ms_per_call, 1e-6)))
    print(f"[bench] [{label}] pilot 1회={pilot_ms_per_call:.4f}ms -> n_iter={n_iter} "
         f"(지속시간 >= {min_duration_s:.0f}s 보장)")

    # burst 는 duration 이 아니라 n_iter 고정 실행이라 _run_tegra_window(duration 기반)를
    # 못 쓴다 -- 여기서 직접 tegrastats 를 띄우고 n_iter 회 실행한다.
    proc_log = out_dir / f"tegrastats_burst_{label}.log"
    proc = subprocess.Popen(["tegrastats", "--interval", "1000", "--logfile", str(proc_log)])
    time.sleep(1.0)
    cpu0 = _cpu_seconds()
    t0 = time.time()
    for _ in range(n_iter):
        target_fn()
    actual_duration = time.time() - t0
    cpu1 = _cpu_seconds()
    time.sleep(1.0)
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
    powers = _read_tegrastats_power(proc_log)
    p_burst = float(np.median(powers)) if powers else float("nan")

    dp_mw = p_burst - p_idle if (p_burst == p_burst and p_idle == p_idle) else float("nan")
    energy_per_op_mj = (dp_mw * (pilot_ms_per_call / 1000.0)) if dp_mw == dp_mw else float("nan")
    cpu_seconds_per_op = ((cpu1 - cpu0) / n_iter) if _HAVE_RESOURCE and n_iter else float("nan")

    result = {
        "target": label, "P_idle_mW": p_idle, "P_burst_mW": p_burst, "dP_mW": dp_mw,
        "n_iter": n_iter, "duration_s": actual_duration,
        "n_tegra_samples_idle": len(idle["power_samples_mw"]), "n_tegra_samples_burst": len(powers),
        "pilot_ms_per_call": pilot_ms_per_call, "energy_per_op_mJ": energy_per_op_mj,
        "cpu_seconds_total": (cpu1 - cpu0) if _HAVE_RESOURCE else float("nan"),
        "cpu_seconds_per_op": cpu_seconds_per_op,
    }
    print(f"[bench] [{label}] P_idle={p_idle:.1f}mW P_burst={p_burst:.1f}mW dP={dp_mw:.1f}mW "
         f"energy/op={energy_per_op_mj:.4f}mJ (n_iter={n_iter}, {actual_duration:.1f}s, "
         f"tegra표본 idle={len(idle['power_samples_mw'])}/burst={len(powers)}, "
         f"cpu-초 교차검증={cpu_seconds_per_op:.6f}s/op)")
    return result


# ══════════════════════════════════════════════════════
# 출력
# ══════════════════════════════════════════════════════
def write_latency_csv(path: Path, results: dict) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "n_iter", "n_warmup_excluded", "median_ms", "p95_ms", "min_ms"])
        for name, r in results.items():
            w.writerow([name, r["n_iter"], r["n_warmup_excluded"],
                       r["median_ms"], r["p95_ms"], r["min_ms"]])


def write_infer_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["input_size", "folds_mode", "n_models", "random_weights",
                   "n_iter", "n_warmup_excluded", "median_ms", "p95_ms", "min_ms"])
        for r in rows:
            w.writerow([r["input_size"], r["folds_mode"], r["n_models"], r["random_weights"],
                       r["n_iter"], r["n_warmup_excluded"], r["median_ms"], r["p95_ms"], r["min_ms"]])


def write_burst_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = ["target", "P_idle_mW", "P_burst_mW", "dP_mW", "n_iter", "duration_s",
                 "n_tegra_samples_idle", "n_tegra_samples_burst", "pilot_ms_per_call",
                 "energy_per_op_mJ", "cpu_seconds_total", "cpu_seconds_per_op"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            if r is not None:
                w.writerow(r)


def write_meta_json(path: Path, meta: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)


# ══════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════
def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="bench_micro -- 노드 없이 판정 함수 직접 호출, 순수 연산 비용 측정")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--cache", type=str, default=str(
        wp.C_PD / "POES" / "MetOp03_count" / "poes_metop03_cache_parquet"))
    p.add_argument("--ckpt-root", type=str, default=str(ai.CKPT_ROOT))
    p.add_argument("--n-ticks", type=int, default=3000, help="실 데이터 준비용 리플레이 틱 수")
    p.add_argument("--core", type=int, default=0)
    p.add_argument("--burst", action="store_true", help="버스트(에너지) 모드도 실행")
    p.add_argument("--smoke", action="store_true", help="반복 수를 대폭 줄여 로직만 확인")
    return p.parse_known_args(argv)[0]


def main(argv=None):
    args = _parse_args(argv)

    if args.smoke:
        for k in REPEATS:
            REPEATS[k] = min(REPEATS[k], 5)
            WARMUP[k] = min(WARMUP[k], 2)

    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(RESULT_DIR) if RESULT_DIR else
        Path(__file__).resolve().parents[2] / "results" / f"bench_micro_{time.strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(1)
    core_pin = pin_single_core(args.core)
    clocks = query_clock_status()

    print(f"[bench] out_dir={out_dir}")
    print(f"[bench] torch_threads={torch.get_num_threads()} core_pin={core_pin}")
    print(f"[bench] clocks={clocks}")

    ckpt_root = Path(args.ckpt_root)
    data = prepare_real_data(Path(args.cache), n_ticks=args.n_ticks)
    print(f"[bench] 실데이터 준비 완료: 유효표본 {data['n_valid']}개, bg_buf {len(data['bg_buf'])}개")

    latency_results = {}
    latency_results["M1_resample_ms"] = bench_m1(data)
    latency_results["M2_z_eval_ms"] = bench_m2(data)
    latency_results["M3_fsm_eval_ms"] = bench_m3(data)
    latency_results["M4_bg_update_ms"] = bench_m4(data)
    latency_results["M7_preproc_ms"] = bench_m7(data)
    latency_results["zbuf_update_ms"] = bench_zbuf_update(data)
    latency_results["serialize_ms"] = bench_serialize(data)
    latency_results["M8_infer_ms"] = bench_m8_real(data, ckpt_root)

    for name, r in latency_results.items():
        print(f"[bench] {name}: n_iter={r['n_iter']}(워밍업 {r['n_warmup_excluded']}회 제외) "
             f"median={r['median_ms']:.5f}ms p95={r['p95_ms']:.5f}ms min={r['min_ms']:.5f}ms")

    write_latency_csv(out_dir / "bench_latency.csv", latency_results)

    infer_rows = bench_infer_sweep(ckpt_root,
                                   n_iter=(5 if args.smoke else 1000),
                                   n_warmup=(2 if args.smoke else 20))
    write_infer_csv(out_dir / "bench_infer.csv", infer_rows)

    burst_rows = []
    if args.burst:
        model, manifest = ai.load_single_model(ckpt_root / "omni_p6_binary" / "checkpoints", fold=0)
        z_valid = data["z_valid"]
        xb = torch.tensor(np.array(z_valid[:WINDOW], dtype=np.float32).reshape(1, -1), dtype=torch.float32)

        def tcn_call():
            with torch.no_grad():
                model(xb)

        burst_rows.append(run_burst(tcn_call, "tcn_infer", out_dir))

        adt = ap.build_adt([CHANNEL], ap.build_rpn_equation([CHANNEL]), GATE_N, ALERT_N, "RTS_SEP_ALERT")[0]
        core = ap.ApFsmCore(adt, dry_run=False)
        watches, ts_list, counts_all = data["watches"], data["ts_list"], data["counts_all"]
        n = len(watches)
        idx_cycle = itertools.cycle(range(n))

        def fsm_call():
            i = next(idx_cycle)
            wp.compute_z(counts_all[i], data["bg_medians_valid"][0], data["bg_stds_valid"][0])
            core.process_tick({CHANNEL: watches[i]}, {CHANNEL: counts_all[i]}, ts_list[i].timestamp())

        burst_rows.append(run_burst(fsm_call, "fsm_eval", out_dir))

    write_burst_csv(out_dir / "bench_burst.csv", burst_rows)

    meta = {
        "torch_threads": torch.get_num_threads(), "core_pin": core_pin, "clocks": clocks,
        "device": "cpu", "python_version": sys.version, "torch_version": torch.__version__,
        "n_ticks_prepared": args.n_ticks, "n_valid_samples": data["n_valid"],
        "repeats": REPEATS, "warmup_excluded": WARMUP,
        "burst_mode_run": args.burst, "tegrastats_available": has_tegrastats(),
        "smoke": args.smoke, "os_name": os.name,
        "has_resource_module": _HAVE_RESOURCE,
    }
    write_meta_json(out_dir / "bench_meta.json", meta)
    print(f"[bench] 저장 완료 -> {out_dir}")


if __name__ == "__main__":
    main()
