"""
verify_aggregate.py
====================
S7 검증 (Windows). **합성 고정치로 로직만 확인한다 -- 수치는 실측이 아니므로
출력은 전부 "합성 표본" 이라고 라벨링한다.** 읽기 전용, C_PD/실제 results/
아래 아무것도 건드리지 않는다(임시 디렉토리에서만 작업).

검증 4종 (md 요청 그대로):
  ① 입력 파일이 하나도 없어도 크래시 없이 "no runs found" 로 종료.
  ② run_meta.json 없는 run 이 섞여도 그것만 건너뛰고 나머지 집계.
  ③ 활성화 표본 30 미만일 때 p95 가 "n<30" 으로 나오는지.
  ④ 파생 상수(96,15,6.3,0.0684)가 하드코딩이 아니라 이름 붙은 상수로 선언되고
     출력(derived_cost.csv 헤더)에 찍히는지.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aggregate_onboard as agg   # noqa: E402


def _write_run_meta(rdir: Path, scenario: str, window: str | None, rep: int | str,
                    n_bg_warmup_ticks_excluded: int = 0, active_speed: float = 7200.0,
                    instrumented: bool = True) -> None:
    rdir.mkdir(parents=True, exist_ok=True)
    meta = {
        "scenario": scenario, "window": window, "rep": rep,
        "warmup_speed": 7200.0, "active_speed": active_speed, "instrumented": instrumented,
        "n_ticks_total": 100, "n_bg_warmup_ticks_excluded": n_bg_warmup_ticks_excluded,
        "channels": ["omni_p6"],
    }
    (rdir / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")


def _write_log_csv(rdir: Path, n_rows: int, warmup_rows: int = 0) -> None:
    rows = []
    for i in range(warmup_rows + n_rows):
        rows.append({
            "phys_ts": 1000.0 + i, "ap_pub_wall_ts": 2000.0 + i, "n_channels": 1,
            "resample_ms": 0.04 + 0.001 * (i % 5), "z_eval_ms": 0.003, "bg_update_ms": "",
            "dds_transport_ms": 0.5, "fsm_eval_ms": 0.0015, "node_latency_ms": 0.6,
            "from_state": "NOMINAL", "ap_state": "NOMINAL", "rpn_result": "NORMAL",
            "counter": 0, "trigger_channels": "",
        })
    pd.DataFrame(rows).to_csv(rdir / "log.csv", index=False)


def _write_ai_log_csv(rdir: Path, n_rows: int) -> None:
    rows = []
    for i in range(n_rows):
        rows.append({
            "phys_ts": 1000.0 + i, "ai_pub_wall_ts": 3000.0 + i, "ap_state": "ALERT", "status": "OK",
            "alert_transport_ms": 0.3, "preproc_ms": 0.004, "infer_ms": 0.24,
            "activation_total_ms": 0.25 + 0.001 * i, "per_model_ms": "[0.24]", "n_models": 1,
            "p_event": 0.5, "y_pred": 1,
        })
    pd.DataFrame(rows).to_csv(rdir / "ai_log.csv", index=False)


def _write_tegrastats(rdir: Path, n_lines: int = 5) -> None:
    lines = [f"08-09-2026 12:00:{i:02d} RAM {1000+i}/3956MB CPU [10%@1907,12%@1907] "
            f"VDD_IN {5000+i}mW/5000mW" for i in range(n_lines)]
    (rdir / "tegrastats.log").write_text("\n".join(lines), encoding="utf-8")


def check_no_runs_found():
    print("=" * 70)
    print("[verify_agg] 검증 ①: 입력 파일 0개 -- 크래시 없이 'no runs found'")
    print("=" * 70)
    with tempfile.TemporaryDirectory() as tmp:
        empty_root = Path(tmp) / "results_empty"
        empty_root.mkdir()
        try:
            result = agg.run(empty_root)
            ok = result["table1"].empty and result["table2"].empty and result["table3"].empty
            print(f"빈 디렉토리 run() 결과: table1/2/3 전부 비어있음={ok} (크래시 없음)")
        except Exception as e:
            print(f"크래시 발생: {e}")
            ok = False

        # 디렉토리 자체가 없는 경우도 확인
        missing_root = Path(tmp) / "does_not_exist"
        try:
            result2 = agg.run(missing_root)
            ok2 = result2["table1"].empty
            print(f"디렉토리 자체가 없을 때도 크래시 없음: {ok2}")
        except Exception as e:
            print(f"크래시 발생(디렉토리 없음): {e}")
            ok2 = False

    ok_all = ok and ok2
    print(f"검증 ① 종합: {ok_all}")
    return ok_all


def check_skip_missing_meta():
    print("=" * 70)
    print("[verify_agg] 검증 ②: run_meta.json 없는 run 은 건너뛰고 나머지 집계 (합성 표본)")
    print("=" * 70)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "results"
        root.mkdir()

        good_dir = root / "20260101_000000_b_fsm_strong_rep1"
        _write_run_meta(good_dir, "b_fsm", "strong", 1)
        _write_log_csv(good_dir, n_rows=50)
        _write_tegrastats(good_dir)

        bad_dir = root / "20260101_000001_b_fsm_weak_rep1"
        bad_dir.mkdir(parents=True)
        _write_log_csv(bad_dir, n_rows=50)   # run_meta.json 없음(의도적)

        runs, warnings = agg.find_runs(root)
        ok_count = (len(runs) == 1)
        ok_warn = any("run_meta.json 없음" in w for w in warnings)
        print(f"로드된 run 수: {len(runs)}(기대 1, 유효한 것만) {'OK' if ok_count else 'FAIL'}")
        print(f"경고 목록: {warnings}")
        print(f"경고에 누락 사실이 남았는지: {ok_warn}")

        result = agg.run(root)
        t1 = result["table1"]
        ok_table = (len(t1) == 1 and t1.iloc[0]["scenario"] == "b") if len(t1) else False
        print(f"table1(합성): \n{t1.to_string() if len(t1) else '(비어있음)'}")
        print(f"table1 에 유효한 b 시나리오 1행만 있는지: {ok_table}")

    ok = ok_count and ok_warn and ok_table
    print(f"검증 ② 종합: {ok}")
    return ok


def check_p95_below_30():
    print("=" * 70)
    print("[verify_agg] 검증 ③: 활성화 표본 30 미만 -> p95 = 'n<30' (합성 표본)")
    print("=" * 70)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "results"
        root.mkdir()

        rdir = root / "20260101_000000_c_tcn_1ch_strong_rep1"
        _write_run_meta(rdir, "c_tcn_1ch", "strong", 1)
        _write_log_csv(rdir, n_rows=50)
        _write_ai_log_csv(rdir, n_rows=12)   # 30 미만(합성)
        _write_tegrastats(rdir)

        result = agg.run(root)
        t1 = result["table1"]
        row = t1[t1["scenario"] == "c1ch"].iloc[0] if len(t1) else None
        print(f"table1(합성) c1ch 행: M9_n={row['M9_n'] if row is not None else '?'}, "
             f"M9_p95_ms={row['M9_p95_ms'] if row is not None else '?'}")
        ok = (row is not None and row["M9_n"] == 12 and row["M9_p95_ms"] == "n<30")

        # 30 이상이면 숫자여야 함(대조군, 합성)
        rdir2 = root / "20260101_000001_c_tcn_1ch_weak_rep1"
        _write_run_meta(rdir2, "c_tcn_1ch", "weak", 1)
        _write_log_csv(rdir2, n_rows=50)
        _write_ai_log_csv(rdir2, n_rows=40)
        _write_tegrastats(rdir2)
        result2 = agg.run(root)
        # weak 은 시나리오 매핑상 c1ch 로 합쳐지므로(둘 다 c1ch) 표본이 12+40=52 -> p95 숫자여야 함
        t1b = result2["table1"]
        row2 = t1b[t1b["scenario"] == "c1ch"].iloc[0]
        ok2 = isinstance(row2["M9_p95_ms"], float) and row2["M9_n"] == 52
        print(f"표본 52개(합성)로 늘리면 M9_p95_ms 가 숫자로 나오는지: {row2['M9_p95_ms']} (n={row2['M9_n']}) {'OK' if ok2 else 'FAIL'}")

    ok_all = ok and ok2
    print(f"검증 ③ 종합: {ok_all}")
    return ok_all


def check_named_constants_in_header():
    print("=" * 70)
    print("[verify_agg] 검증 ④: 파생 상수가 이름 붙은 상수 + 출력 헤더에 기록되는지")
    print("=" * 70)
    ok_named = all(hasattr(agg, n) for n in
                  ("TICKS_PER_DAY", "SAMPLES_PER_TICK", "GATE_OPENINGS_PER_DAY", "DUTY_CYCLE"))
    print(f"모듈 상단에 이름 붙은 상수로 존재: TICKS_PER_DAY={agg.TICKS_PER_DAY}, "
         f"SAMPLES_PER_TICK={agg.SAMPLES_PER_TICK}, GATE_OPENINGS_PER_DAY={agg.GATE_OPENINGS_PER_DAY}, "
         f"DUTY_CYCLE={agg.DUTY_CYCLE}  {'OK' if ok_named else 'FAIL'}")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "results"
        root.mkdir()
        rdir = root / "20260101_000000_b_fsm_strong_rep1"
        _write_run_meta(rdir, "b_fsm", "strong", 1)
        _write_log_csv(rdir, n_rows=50)
        _write_tegrastats(rdir)

        result = agg.run(root)
        agg.write_outputs(root, result)
        header_line = (root / "derived_cost.csv").read_text(encoding="utf-8").splitlines()[0]
        print(f"derived_cost.csv 첫 줄(헤더 주석): {header_line}")
        ok_header = all(str(v) in header_line for v in
                       (agg.TICKS_PER_DAY, agg.SAMPLES_PER_TICK, agg.GATE_OPENINGS_PER_DAY, agg.DUTY_CYCLE))
        print(f"헤더에 네 상수 값이 전부 찍혀있는지: {ok_header}")

    ok = ok_named and ok_header
    print(f"검증 ④ 종합: {ok}")
    return ok


def main():
    r1 = check_no_runs_found()
    r2 = check_skip_missing_meta()
    r3 = check_p95_below_30()
    r4 = check_named_constants_in_header()
    print("=" * 70)
    print(f"[verify_agg] 종합: no_runs={r1} skip_missing_meta={r2} p95_n<30={r3} named_constants={r4}")
    print("[verify_agg] 위 수치는 전부 합성 표본이다 -- 실제 논문 표는 젯슨 S5/S6 산출물로만 유효.")
    print(f"[verify_agg] 전체 통과: {all([r1, r2, r3, r4])}")


if __name__ == "__main__":
    main()
