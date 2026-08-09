"""
verify_bench_micro.py
======================
S6 검증 (Windows). 문법·import·로직만 확인한다 -- 수치는 안 낸다(실제 측정은
젯슨에서만 유효). 읽기 전용, C_PD 아래 아무것도 쓰지 않는다.

검증 4종 (md 요청 그대로):
  ① rclpy 없이 import 성공.
  ② 각 측정 대상 함수가 실제로 호출되는지 -- monkeypatch 로 호출 카운터를 심어
     bench_m1~m8/zbuf/serialize 를 작은 n_iter 로 돌리고 카운트를 확인한다.
  ③ 반복 수·워밍업 제외 수가 meta 에 정확히 기록되는지 -- bench_call() 반환값과
     --smoke main() 이 쓴 bench_meta.json 양쪽을 확인.
  ④ tegrastats 없는 환경(이 Windows 머신)에서 버스트 모드가 우아하게 스킵되는지.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wp_poes_node as wp   # noqa: E402
import ap_fsm_node as ap    # noqa: E402
import ai_tcn_node as ai    # noqa: E402
import bench_micro as bm    # noqa: E402  -- rclpy 없이 import 되는지 자체가 검증 1

assert not bm.wp._HAVE_RCLPY, "이 Windows 환경엔 rclpy 가 없어야 하는데 있음(예상과 다름)"
print(f"[verify_bench] 검증 1: rclpy 없이 bench_micro import 성공")

CACHE_DIR = wp.C_PD / "POES" / "MetOp03_count" / "poes_metop03_cache_parquet"
CKPT_ROOT = ai.CKPT_ROOT


def check_call_counters():
    print("=" * 70)
    print("[verify_bench] 검증 2: 각 측정 대상 함수가 실제로 호출되는지 (호출 카운터)")
    print("=" * 70)
    data = bm.prepare_real_data(CACHE_DIR, n_ticks=1200)
    print(f"실데이터 준비: 유효표본 {data['n_valid']}개")

    results = {}

    with mock.patch.object(wp, "resample_window", wraps=wp.resample_window) as m:
        bm.REPEATS["M1_resample_ms"], bm.WARMUP["M1_resample_ms"] = 5, 2
        bm.bench_m1(data)
        results["resample_window(M1)"] = m.call_count == 7   # n_iter+n_warmup

    with mock.patch.object(wp, "compute_z", wraps=wp.compute_z) as m:
        bm.REPEATS["M2_z_eval_ms"], bm.WARMUP["M2_z_eval_ms"] = 5, 2
        bm.bench_m2(data)
        results["compute_z(M2)"] = m.call_count == 7

    with mock.patch.object(ap.ApFsmCore, "process_tick", autospec=True,
                           side_effect=ap.ApFsmCore.process_tick) as m:
        bm.REPEATS["M3_fsm_eval_ms"], bm.WARMUP["M3_fsm_eval_ms"] = 5, 2
        bm.bench_m3(data)
        results["ApFsmCore.process_tick(M3)"] = m.call_count == 7

    with mock.patch.object(wp, "compute_bg_stats", wraps=wp.compute_bg_stats) as m:
        bm.REPEATS["M4_bg_update_ms"], bm.WARMUP["M4_bg_update_ms"] = 5, 2
        bm.bench_m4(data)
        results["compute_bg_stats(M4)"] = m.call_count == 7

    with mock.patch.object(ai.ZBuffer, "assemble", autospec=True,
                           side_effect=ai.ZBuffer.assemble) as m:
        bm.REPEATS["M7_preproc_ms"], bm.WARMUP["M7_preproc_ms"] = 5, 2
        bm.bench_m7(data)
        results["ZBuffer.assemble(M7)"] = m.call_count == 7

    with mock.patch.object(ai.ZBuffer, "update", autospec=True,
                           side_effect=ai.ZBuffer.update) as m:
        bm.REPEATS["zbuf_update_ms"], bm.WARMUP["zbuf_update_ms"] = 5, 2
        bm.bench_zbuf_update(data)
        results["ZBuffer.update(zbuf)"] = m.call_count == 7

    with mock.patch.object(json, "dumps", wraps=json.dumps) as m:
        bm.REPEATS["serialize_ms"], bm.WARMUP["serialize_ms"] = 5, 2
        bm.bench_serialize(data)
        results["json.dumps(serialize)"] = m.call_count == 7

    ckpt_dir = CKPT_ROOT / "omni_p6_binary" / "checkpoints"
    if (ckpt_dir / "manifest.json").exists():
        # nn.Module.__call__ 자체는 내부 디스패치(_call_impl 등)가 복잡해 직접 패치가
        # 안 먹는다 -- 실제로 항상 호출되는 TCNClassifier.forward 를 패치한다.
        from tcn import TCNClassifier
        real_forward = TCNClassifier.forward
        with mock.patch.object(TCNClassifier, "forward", autospec=True,
                               side_effect=real_forward) as m:
            bm.REPEATS["M8_infer_ms"], bm.WARMUP["M8_infer_ms"] = 5, 2
            bm.bench_m8_real(data, CKPT_ROOT)
            results["TCNClassifier.forward(M8)"] = m.call_count == 7
    else:
        print(f"  (omni_p6_binary 체크포인트 없음 -- M8 호출 카운터 확인 스킵)")

    for name, ok in results.items():
        print(f"  {name}: 호출 확인 {'OK' if ok else 'FAIL'}")
    ok_all = all(results.values())
    print(f"검증 2 종합: {ok_all}")
    return ok_all


def check_repeats_and_warmup_recorded():
    print("=" * 70)
    print("[verify_bench] 검증 3: 반복 수/워밍업 제외 수가 정확히 기록되는지")
    print("=" * 70)
    calls = {"n": 0}

    def fn():
        calls["n"] += 1

    r = bm.bench_call(fn, n_iter=37, n_warmup=11)
    ok1 = (r["n_iter"] == 37 and r["n_warmup_excluded"] == 11 and calls["n"] == 48)
    print(f"bench_call(n_iter=37, n_warmup=11) -> n_iter={r['n_iter']} "
         f"n_warmup_excluded={r['n_warmup_excluded']} 실제호출={calls['n']}(기대 48) {'OK' if ok1 else 'FAIL'}")

    # --smoke 전체 main() 실행 -> bench_meta.json 의 repeats/warmup_excluded 확인
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "bench_out"
        argv = ["--smoke", "--out-dir", str(out_dir),
               "--cache", str(CACHE_DIR), "--ckpt-root", str(CKPT_ROOT), "--n-ticks", "1200"]
        bm.main(argv)
        meta = json.loads((out_dir / "bench_meta.json").read_text(encoding="utf-8"))
        ok2 = all(v <= 5 for v in meta["repeats"].values()) and all(v <= 2 for v in meta["warmup_excluded"].values())
        print(f"--smoke bench_meta.json repeats(축소됨, <=5): {meta['repeats']}")
        print(f"--smoke bench_meta.json warmup_excluded(축소됨, <=2): {meta['warmup_excluded']}")
        print(f"기록 정확성: {'OK' if ok2 else 'FAIL'}")

        lat_csv = out_dir / "bench_latency.csv"
        infer_csv = out_dir / "bench_infer.csv"
        ok3 = lat_csv.exists() and infer_csv.exists()
        print(f"bench_latency.csv/bench_infer.csv 생성됨: {ok3}")

    ok = ok1 and ok2 and ok3
    print(f"검증 3 종합: {ok}")
    return ok


def check_burst_graceful_skip():
    print("=" * 70)
    print("[verify_bench] 검증 4: tegrastats 없는 환경에서 버스트 모드가 우아하게 스킵되는지")
    print("=" * 70)
    has_tg = bm.has_tegrastats()
    print(f"has_tegrastats(): {has_tg} (이 Windows 환경에선 False 여야 함)")

    called = {"n": 0}

    def dummy_fn():
        called["n"] += 1

    with tempfile.TemporaryDirectory() as tmpdir:
        result = bm.run_burst(dummy_fn, "smoke_test", Path(tmpdir), pilot_iters=5, min_duration_s=1.0)

    ok = (has_tg is False) and (result is None) and (called["n"] == 0)
    print(f"run_burst() 반환값: {result} (기대: None)")
    print(f"target_fn 이 호출되지 않고 즉시 스킵됐는지(호출 {called['n']}회, 기대 0): {called['n'] == 0}")
    print(f"검증 4 종합: {ok}")
    return ok


def main():
    r2 = check_call_counters()
    r3 = check_repeats_and_warmup_recorded()
    r4 = check_burst_graceful_skip()
    print("=" * 70)
    print(f"[verify_bench] 종합: 함수호출확인={r2} 반복수기록={r3} 버스트스킵={r4}")
    print(f"[verify_bench] 실제 수치(median/p95/energy 등)는 젯슨에서만 유효 -- "
         f"이 결과는 로직·배선 검증일 뿐이다.")
    print(f"[verify_bench] 전체 통과: {all([r2, r3, r4])}")


if __name__ == "__main__":
    main()
