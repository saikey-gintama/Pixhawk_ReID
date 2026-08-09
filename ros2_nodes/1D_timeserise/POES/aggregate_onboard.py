"""
aggregate_onboard.py
=====================
S7 -- S5(노드 리플레이)와 S6(마이크로벤치)의 산출물을 읽어 논문 4.3절 표 4개를
만든다. **새 측정은 하지 않는다.** 집계와 파생 계산만.

입력:
  results/*/log.csv, ai_log.csv, zbuf_log.csv, event_log.csv, run_meta.json, tegrastats.log
  results/*/speed_sanity.csv
  results/bench_micro_*/bench_latency.csv, bench_infer.csv, bench_burst.csv, bench_meta.json

run_meta.json 이 없는 run 은 집계하지 않고 경고로만 남긴다 -- 측정 조건을 모르는
수치는 논문에 못 쓴다(md 원칙).

warm-up 제외: run_meta.json 의 n_bg_warmup_ticks_excluded 를 그대로 쓴다. log.csv
앞부분에서 그만큼 잘라낸다. KSEM aggregate_resource.py 의 "타이머 간격 급변"
휴리스틱은 재발명하지 않는다 -- 이미 wp_poes_node.py 가 실측해 run_meta 에 넣어뒀다.

산출 4개:
  results/onboard_cost.csv      표1 -- 시나리오별(a/a'/b/c1ch/d3ch/e) M1~M10 + 전력/CPU/RAM
  results/bench_summary.csv     표2 -- 벤치(순수 연산) vs 노드(DDS 포함) 나란히, ratio
  results/derived_cost.csv      표3 -- 하루 비용/에너지 파생(핵심 표, 계산 근거 열 포함)
  results/measurement_note.md   표4 -- 논문 4.3절 인용용 측정 조건 markdown

파생 상수(하드코딩 아님, 아래 이름 붙여 선언 + 출력 헤더에 기록):
  TICKS_PER_DAY=96(15분 격자 하루 틱 수), SAMPLES_PER_TICK=15(15분당 1분 샘플 수),
  GATE_OPENINGS_PER_DAY=6.3, DUTY_CYCLE=0.0684
    -- 이 둘은 S0.5(gate_persistence_sweep.py, N=2 확정 게이트 행)의 236,848틱
       전 구간 통계에서 온 값이다. 젯슨 리플레이는 17일 구간 3개뿐이라 여기서
       "하루 몇 번" 을 재추정하면 안 된다 -- 그래서 상수로 고정한다.
       (젯슨 리플레이=1회 비용 실측, S0.5=그 비용이 하루에 몇 번 나는지)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════
# 파생 상수 (재계산 금지 -- 출처를 주석에 명시)
# ══════════════════════════════════════════════════════
TICKS_PER_DAY = 96              # 15분 격자 하루 틱 수 (24*60/15)
SAMPLES_PER_TICK = 15           # 15분당 1분 원시 샘플 수 (M1 리샘플 입력 크기)
GATE_OPENINGS_PER_DAY = 6.3     # S0.5 gate_persistence_sweep.csv N=2 행 n_invocations_day. 재계산 금지.
DUTY_CYCLE = 0.0684             # S0.5 gate_persistence_sweep.csv N=2 행 duty_cycle(6.84%). 재계산 금지.
MIN_SAMPLES_FOR_P95 = 30        # 활성화 단위 표본이 이보다 적으면 p95 대신 "n<30"

SCENARIO_LABELS = {
    "a_idle": "a", "a_prime_idle_ros": "a'", "b_fsm": "b",
    "c_tcn_1ch": "c1ch", "d_tcn_3ch": "d3ch", "e_offboard": "e",
}
SCENARIO_ORDER = ["a", "a'", "b", "c1ch", "d3ch", "e"]


# ══════════════════════════════════════════════════════
# run 로딩 -- run_meta.json 없으면 경고만 남기고 제외
# ══════════════════════════════════════════════════════
def find_runs(results_root: Path) -> tuple[list[dict], list[str]]:
    runs, warnings = [], []
    if not results_root.exists():
        return runs, warnings
    for rdir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        if rdir.name.startswith("bench_micro_") or rdir.name == "paper":
            continue
        meta_path = rdir / "run_meta.json"
        if not meta_path.exists():
            warnings.append(f"{rdir.name}: run_meta.json 없음 -- 집계에서 제외")
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            warnings.append(f"{rdir.name}: run_meta.json 파싱 실패({e}) -- 집계에서 제외")
            continue
        runs.append({"rdir": rdir, "meta": meta})
    return runs, warnings


def is_speedsanity(meta: dict) -> bool:
    return str(meta.get("rep", "")).startswith("speedsanity")


def is_nolog(meta: dict) -> bool:
    return "nolog" in str(meta.get("rep", "")) or meta.get("instrumented") is False


def canonical_scenario(meta: dict) -> str | None:
    return SCENARIO_LABELS.get(meta.get("scenario"))


# ══════════════════════════════════════════════════════
# warm-up 트림 (run_meta 의 실측값 그대로 사용, 휴리스틱 재발명 금지)
# ══════════════════════════════════════════════════════
def load_csv_trimmed(rdir: Path, filename: str, meta: dict, ts_col: str = "phys_ts") -> pd.DataFrame:
    path = rdir / filename
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if df.empty or ts_col not in df.columns:
        return df
    df = df.sort_values(ts_col).reset_index(drop=True)
    n_warmup = meta.get("n_bg_warmup_ticks_excluded")
    if n_warmup is None or filename == "event_log.csv":
        # event_log.csv 는 phys_ts 가 아니라 timestamp(벽시계) 라 틱 워밍업 트림 대상이 아님
        return df
    return df.iloc[int(n_warmup):].reset_index(drop=True)


# ══════════════════════════════════════════════════════
# tegrastats 파서 (power_mW, ram_MB, cpu_pct). run_meta 의 prestart 초를 라인 단위로 스킵.
# ══════════════════════════════════════════════════════
_RE_POWER = re.compile(r"VDD_IN (\d+)mW")
_RE_RAM = re.compile(r"RAM (\d+)/(\d+)MB")
_RE_CPU = re.compile(r"CPU \[([^\]]+)\]")


def parse_tegrastats(rdir: Path, meta: dict) -> pd.DataFrame:
    path = rdir / "tegrastats.log"
    if not path.exists():
        return pd.DataFrame(columns=["power_mW", "ram_MB", "cpu_pct"])
    interval_ms = meta.get("tegra_interval_ms") or 1000.0
    prestart_s = meta.get("tegra_prestart_sec_excluded") or 0.0
    n_skip = int(np.ceil((prestart_s * 1000.0) / max(interval_ms, 1.0)))

    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        p, r, c = _RE_POWER.search(line), _RE_RAM.search(line), _RE_CPU.search(line)
        cpu_pct = np.nan
        if c:
            vals = re.findall(r"(\d+)%", c.group(1))
            if vals:
                cpu_pct = float(np.mean([float(v) for v in vals]))
        rows.append({
            "power_mW": float(p.group(1)) if p else np.nan,
            "ram_MB": float(r.group(1)) if r else np.nan,
            "cpu_pct": cpu_pct,
        })
    df = pd.DataFrame(rows)
    return df.iloc[n_skip:].reset_index(drop=True) if len(df) > n_skip else df.iloc[0:0]


# ══════════════════════════════════════════════════════
# 통계 헬퍼 -- median/p95(표본 30 미만이면 "n<30")
# ══════════════════════════════════════════════════════
def median_p95(values) -> dict:
    arr = np.asarray(pd.to_numeric(pd.Series(values), errors="coerce").dropna(), dtype=float)
    n = len(arr)
    if n == 0:
        return {"median": np.nan, "p95": np.nan, "n": 0}
    med = float(np.median(arr))
    p95 = float(np.percentile(arr, 95)) if n >= MIN_SAMPLES_FOR_P95 else "n<30"
    return {"median": med, "p95": p95, "n": n}


# ══════════════════════════════════════════════════════
# 표 1 -- 시나리오별 onboard_cost.csv
# ══════════════════════════════════════════════════════
def build_table1(runs: list[dict]) -> pd.DataFrame:
    main_runs = [r for r in runs if not is_speedsanity(r["meta"]) and not is_nolog(r["meta"])]
    by_scn: dict[str, list[dict]] = {}
    for r in main_runs:
        scn = canonical_scenario(r["meta"])
        if scn is None:
            continue
        by_scn.setdefault(scn, []).append(r)

    power_means: dict[str, float] = {}
    rows = []
    for scn in SCENARIO_ORDER:
        runs_here = by_scn.get(scn, [])
        if not runs_here:
            continue
        log_frames, ai_frames, ev_frames, tegra_frames = [], [], [], []
        n_ticks_active = 0
        for r in runs_here:
            rdir, meta = r["rdir"], r["meta"]
            log_df = load_csv_trimmed(rdir, "log.csv", meta)
            ai_df = load_csv_trimmed(rdir, "ai_log.csv", meta)
            ev_df = load_csv_trimmed(rdir, "event_log.csv", meta)
            tg_df = parse_tegrastats(rdir, meta)
            n_ticks_active += len(log_df)
            if len(log_df):
                log_frames.append(log_df)
            if len(ai_df):
                ai_frames.append(ai_df)
            if len(ev_df):
                ev_frames.append(ev_df)
            if len(tg_df):
                tegra_frames.append(tg_df)

        log_all = pd.concat(log_frames, ignore_index=True) if log_frames else pd.DataFrame()
        ai_all = pd.concat(ai_frames, ignore_index=True) if ai_frames else pd.DataFrame()
        ev_all = pd.concat(ev_frames, ignore_index=True) if ev_frames else pd.DataFrame()
        tegra_all = pd.concat(tegra_frames, ignore_index=True) if tegra_frames else pd.DataFrame()

        row = {"scenario": scn, "n_runs": len(runs_here), "n_ticks_active": n_ticks_active,
              "n_activations": len(ai_all)}

        for label, col in (("M1", "resample_ms"), ("M2", "z_eval_ms"), ("M3", "fsm_eval_ms"),
                          ("M4", "bg_update_ms"), ("M5", "dds_transport_ms")):
            stat = median_p95(log_all[col]) if col in log_all.columns else {"median": np.nan, "p95": np.nan, "n": 0}
            row[f"{label}_median_ms"] = stat["median"]
            row[f"{label}_p95_ms"] = stat["p95"]

        for label, col in (("M6", "alert_transport_ms"), ("M7", "preproc_ms"),
                          ("M8", "infer_ms"), ("M9", "activation_total_ms")):
            stat = median_p95(ai_all[col]) if col in ai_all.columns else {"median": np.nan, "p95": np.nan, "n": 0}
            row[f"{label}_median_ms"] = stat["median"]
            row[f"{label}_p95_ms"] = stat["p95"]
            row[f"{label}_n"] = stat["n"]

        m10 = median_p95(ev_all["e2e_ms"]) if "e2e_ms" in ev_all.columns else {"median": np.nan, "p95": np.nan, "n": 0}
        row["M10_median_ms"] = m10["median"]
        row["M10_p95_ms"] = m10["p95"]
        row["M10_n"] = m10["n"]

        row["power_mW_mean"] = float(tegra_all["power_mW"].mean()) if len(tegra_all) else np.nan
        row["cpu_pct_mean"] = float(tegra_all["cpu_pct"].mean()) if len(tegra_all) else np.nan
        row["ram_MB_mean"] = float(tegra_all["ram_MB"].mean()) if len(tegra_all) else np.nan
        power_means[scn] = row["power_mW_mean"]

        speeds = {r["meta"].get("active_speed") for r in runs_here}
        row["speed_factor"] = speeds.pop() if len(speeds) == 1 else f"mixed{sorted(speeds)}"

        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    p_a = power_means.get("a", np.nan)
    p_aprime = power_means.get("a'", np.nan)
    df["dP_vs_idle_mW"] = df["power_mW_mean"] - p_a
    df["dP_vs_idle_ros_mW"] = df["power_mW_mean"] - p_aprime
    return df


# ══════════════════════════════════════════════════════
# 표 2 -- bench_summary.csv (S6 벤치 vs S5 노드 나란히, ratio)
# ══════════════════════════════════════════════════════
def load_bench(results_root: Path) -> dict | None:
    """가장 최근 bench_micro_* 디렉토리 하나를 읽는다."""
    cands = sorted(results_root.glob("bench_micro_*"))
    if not cands:
        return None
    bdir = cands[-1]
    out = {"bdir": bdir}
    for name in ("bench_latency.csv", "bench_infer.csv", "bench_burst.csv"):
        p = bdir / name
        out[name] = pd.read_csv(p) if p.exists() else pd.DataFrame()
    meta_p = bdir / "bench_meta.json"
    out["bench_meta.json"] = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    return out


def build_table2(table1: pd.DataFrame, bench: dict | None) -> pd.DataFrame:
    if bench is None or table1.empty:
        return pd.DataFrame()
    lat = bench["bench_latency.csv"]
    if lat.empty:
        return pd.DataFrame()
    lat_idx = lat.set_index("metric")

    node_row = table1[table1["scenario"] == "b"]
    node_row = node_row.iloc[0] if len(node_row) else None

    pairs = [("M1", "M1_resample_ms"), ("M2", "M2_z_eval_ms"), ("M3", "M3_fsm_eval_ms"),
            ("M4", "M4_bg_update_ms"), ("M7", "M7_preproc_ms"), ("M8", "M8_infer_ms")]
    rows = []
    for label, bench_metric in pairs:
        bench_median = float(lat_idx.loc[bench_metric, "median_ms"]) if bench_metric in lat_idx.index else np.nan
        node_col = f"{label}_median_ms"
        node_median = float(node_row[node_col]) if (node_row is not None and node_col in node_row and
                                                     pd.notna(node_row[node_col])) else np.nan
        ratio = (node_median / bench_median) if (bench_median and bench_median == bench_median
                                                 and node_median == node_median) else np.nan
        rows.append({"metric": label, "bench_pure_ms": bench_median, "node_with_dds_ms": node_median,
                    "ratio_node_over_bench": ratio})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════
# 표 3 -- derived_cost.csv (핵심 표)
# ══════════════════════════════════════════════════════
def build_table3(table1: pd.DataFrame, bench: dict | None) -> pd.DataFrame:
    rows = []
    b_row = table1[table1["scenario"] == "b"]
    if b_row.empty:
        return pd.DataFrame()
    b_row = b_row.iloc[0]
    m1, m2, m3, m4 = (b_row.get(f"{k}_median_ms") for k in ("M1", "M2", "M3", "M4"))
    if any(v is None or (isinstance(v, float) and math_isnan(v)) for v in (m1, m2, m3, m4)):
        fsm_daily_ms = np.nan
    else:
        fsm_daily_ms = m4 * 1 + TICKS_PER_DAY * (SAMPLES_PER_TICK * m1 + m2 + m3)

    # ── S5 실측 M9 (있으면, 참고/교차검증용) ──
    m9_real = {}
    for scn, input_size in (("c1ch", 1), ("d3ch", 3)):
        r = table1[table1["scenario"] == scn]
        if len(r) and pd.notna(r.iloc[0].get("M9_median_ms")):
            m9_real[(input_size, "single")] = float(r.iloc[0]["M9_median_ms"])

    # ── S6 벤치로 구성한 M9 (모든 input_size x folds_mode 조합에 대해) ──
    m9_bench = {}
    if bench is not None:
        lat = bench["bench_latency.csv"]
        infer = bench["bench_infer.csv"]
        if not lat.empty and not infer.empty:
            lat_idx = lat.set_index("metric")
            m7 = float(lat_idx.loc["M7_preproc_ms", "median_ms"]) if "M7_preproc_ms" in lat_idx.index else np.nan
            ser = float(lat_idx.loc["serialize_ms", "median_ms"]) if "serialize_ms" in lat_idx.index else np.nan
            for _, r in infer.iterrows():
                key = (int(r["input_size"]), r["folds_mode"])
                m9_bench[key] = m7 + float(r["median_ms"]) + ser   # infer median 은 이미 앙상블 전체 모델 합산 시간

    for input_size in (1, 3, 5):
        for folds_mode, mult_label in (("single", "1x"), ("ensemble", "5x")):
            key = (input_size, folds_mode)
            m9 = m9_bench.get(key, np.nan)
            m9_source = "S6 벤치 구성값(M7+infer+serialize)"
            m9_r = m9_real.get(key)
            if m9_r is not None:
                m9_source += f"; S5 실측 M9={m9_r:.4f}ms(교차검증용, 참고)"
                if m9 == m9 and abs(m9 - m9_r) / max(m9_r, 1e-9) > 0.2:
                    m9_source += " [!] 벤치-구성값과 20% 이상 차이"

            tcn_daily_ms = GATE_OPENINGS_PER_DAY * m9 if m9 == m9 else np.nan
            tcn_always_on_daily_ms = TICKS_PER_DAY * m9 if m9 == m9 else np.nan
            cost_ratio = (tcn_daily_ms / fsm_daily_ms) if (fsm_daily_ms == fsm_daily_ms and fsm_daily_ms
                                                           and tcn_daily_ms == tcn_daily_ms) else np.nan
            rows.append({
                "input_size": input_size, "folds_mode": folds_mode, "mult": mult_label,
                "fsm_daily_ms": fsm_daily_ms, "M9_median_ms": m9, "m9_source": m9_source,
                "tcn_daily_ms": tcn_daily_ms, "cost_ratio_tcn_over_fsm": cost_ratio,
                "duty_cycle": DUTY_CYCLE, "gate_saving_vs_always_on": 1.0 / DUTY_CYCLE,
                "tcn_always_on_daily_ms": tcn_always_on_daily_ms,
                "ticks_per_day": TICKS_PER_DAY, "samples_per_tick": SAMPLES_PER_TICK,
                "gate_openings_per_day": GATE_OPENINGS_PER_DAY,
            })

    # ── 에너지 파생 (S6 버스트) ──
    if bench is not None:
        burst = bench["bench_burst.csv"]
        if not burst.empty:
            burst_idx = burst.set_index("target")
            e_tcn = float(burst_idx.loc["tcn_infer", "energy_per_op_mJ"]) if "tcn_infer" in burst_idx.index else np.nan
            e_fsm = float(burst_idx.loc["fsm_eval", "energy_per_op_mJ"]) if "fsm_eval" in burst_idx.index else np.nan
            cpu_tcn = float(burst_idx.loc["tcn_infer", "cpu_seconds_per_op"]) if "tcn_infer" in burst_idx.index else np.nan
            cpu_fsm = float(burst_idx.loc["fsm_eval", "cpu_seconds_per_op"]) if "fsm_eval" in burst_idx.index else np.nan

            daily_energy_mJ = (TICKS_PER_DAY * e_fsm + GATE_OPENINGS_PER_DAY * e_tcn) \
                if (e_fsm == e_fsm and e_tcn == e_tcn) else np.nan
            daily_cpu_seconds = (TICKS_PER_DAY * cpu_fsm + GATE_OPENINGS_PER_DAY * cpu_tcn) \
                if (cpu_fsm == cpu_fsm and cpu_tcn == cpu_tcn) else np.nan

            note = ""
            if daily_energy_mJ == daily_energy_mJ and daily_cpu_seconds == daily_cpu_seconds:
                # 대략적 일관성 확인용(단위가 달라 직접 비교 불가 -- 상대적 경향만): TCN/FSM 비중이 두 추정에서 비슷한 방향인지
                mj_ratio = (GATE_OPENINGS_PER_DAY * e_tcn) / max(daily_energy_mJ, 1e-9)
                cpu_ratio = (GATE_OPENINGS_PER_DAY * cpu_tcn) / max(daily_cpu_seconds, 1e-9)
                if abs(mj_ratio - cpu_ratio) > 0.2:
                    note = (f"[!] 에너지 기준 TCN 비중({mj_ratio:.2f})과 CPU-초 기준 TCN 비중({cpu_ratio:.2f})이 "
                           f"20%p 이상 어긋남 -- 두 추정을 하나로 합치지 말고 나란히 보고할 것")
            rows.append({
                "input_size": "-", "folds_mode": "-", "mult": "energy(burst)",
                "fsm_daily_ms": np.nan, "M9_median_ms": np.nan,
                "m9_source": f"energy_per_inference_mJ={e_tcn}, energy_per_fsm_tick_mJ={e_fsm}",
                "tcn_daily_ms": np.nan, "cost_ratio_tcn_over_fsm": np.nan,
                "duty_cycle": DUTY_CYCLE, "gate_saving_vs_always_on": 1.0 / DUTY_CYCLE,
                "tcn_always_on_daily_ms": np.nan,
                "ticks_per_day": TICKS_PER_DAY, "samples_per_tick": SAMPLES_PER_TICK,
                "gate_openings_per_day": GATE_OPENINGS_PER_DAY,
                "daily_energy_mJ": daily_energy_mJ, "daily_cpu_seconds_independent_estimate": daily_cpu_seconds,
                "cross_check_note": note,
            })

    return pd.DataFrame(rows)


def math_isnan(v) -> bool:
    try:
        return v != v
    except Exception:
        return False


# ══════════════════════════════════════════════════════
# rep 분산 (같은 시나리오·같은 구간 반복 run 간 표준편차/변동계수)
# ══════════════════════════════════════════════════════
def build_rep_variance(runs: list[dict]) -> pd.DataFrame:
    main_runs = [r for r in runs if not is_speedsanity(r["meta"]) and not is_nolog(r["meta"])]
    groups: dict[tuple, list[dict]] = {}
    for r in main_runs:
        scn = canonical_scenario(r["meta"])
        window = r["meta"].get("window")
        if scn not in ("b", "c1ch"):
            continue
        groups.setdefault((scn, window), []).append(r)

    rows = []
    for (scn, window), rs in groups.items():
        if len(rs) < 2:
            continue
        for label, col in (("M1", "resample_ms"), ("M2", "z_eval_ms"), ("M3", "fsm_eval_ms"),
                          ("M9", "activation_total_ms")):
            per_run_medians = []
            for r in rs:
                fname = "ai_log.csv" if label == "M9" else "log.csv"
                df = load_csv_trimmed(r["rdir"], fname, r["meta"])
                if col in df.columns and len(df):
                    per_run_medians.append(float(pd.to_numeric(df[col], errors="coerce").median()))
            if len(per_run_medians) < 2:
                continue
            arr = np.array(per_run_medians)
            mean = float(arr.mean())
            std = float(arr.std(ddof=1))
            cv = (std / mean) if mean else np.nan
            rows.append({
                "scenario": scn, "window": window, "metric": label, "n_reps": len(per_run_medians),
                "mean_of_medians_ms": mean, "std_ms": std, "cv": cv,
                "flag_high_cv": (cv == cv and cv > 0.10),
            })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════
# 표 4 -- measurement_note.md
# ══════════════════════════════════════════════════════
def build_measurement_note(runs: list[dict], bench: dict | None, warnings: list[str],
                           table1: pd.DataFrame, rep_var: pd.DataFrame) -> str:
    lines = []
    lines.append("# 측정 조건 노트 (논문 4.3절 인용용)\n")
    lines.append("**경고: 전력·CPU 는 가속 리플레이 값이며 비행 조건이 아니다. "
                 "연산 1회당 에너지는 버스트 모드(S6)에서만 유효하다.**\n")
    lines.append("**하루 비용(daily_ms/daily_energy)은 해석적으로 계산된 값이다 -- "
                 "젯슨은 1회 비용만 실측했고(S5/S6), 하루 몇 번 발생하는지는 "
                 "S0.5 전 구간(236,848틱) 통계의 상수(6.3회/일, duty_cycle 6.84%)를 그대로 썼다.**\n")

    bmeta = (bench or {}).get("bench_meta.json", {})
    lines.append("## 플랫폼/환경")
    lines.append(f"- torch_threads: {bmeta.get('torch_threads', '(bench 없음)')}")
    lines.append(f"- core_pin: {bmeta.get('core_pin', '(bench 없음)')}")
    lines.append(f"- clocks(nvpmodel/jetson_clocks): {bmeta.get('clocks', '(bench 없음)')}")
    lines.append(f"- device: {bmeta.get('device', '(bench 없음)')}")
    lines.append(f"- os: {bmeta.get('os_name', '(bench 없음)')}\n")

    lines.append("## 리플레이 구간 3개 (event_windows.json)")
    ew_windows = set()
    for r in runs:
        w = r["meta"].get("event_window_detail")
        if w:
            ew_windows.add(json.dumps(w, sort_keys=True))
    for w_json in sorted(ew_windows):
        w = json.loads(w_json)
        lines.append(f"- event_id={w.get('event_id', w.get('episode_id'))}, "
                     f"{w.get('start')}~{w.get('end')}, "
                     f"peak_count={w.get('peak_count', w.get('n_events_in_window'))}, "
                     f"기준: {w.get('criterion')}")
    lines.append("")

    lines.append("## 배속")
    speeds = {(r["meta"].get("warmup_speed"), r["meta"].get("active_speed")) for r in runs
             if not is_speedsanity(r["meta"])}
    lines.append(f"- warm-up/active 배속 조합: {sorted(str(s) for s in speeds)}")
    ss_note = "speed_sanity.csv 없음(아직 실행 안 됨)"
    for r in runs:
        p = r["rdir"] / "speed_sanity.csv"
        if p.exists():
            try:
                ss = pd.read_csv(p)
                all_pass = bool(ss["pass"].all()) if "pass" in ss.columns else None
                ss_note = f"{p.parent.name}/speed_sanity.csv: {'PASS' if all_pass else 'FAIL'} " \
                         f"(행별 rel_diff: {ss[['metric','rel_diff','pass']].to_dict('records') if len(ss) else []})"
            except Exception as e:
                ss_note = f"speed_sanity.csv 파싱 실패: {e}"
            break
    lines.append(f"- speed_sanity(1800x vs 7200x): {ss_note}\n")

    lines.append("## warm-up 제외")
    wu = {r["meta"].get("n_bg_warmup_ticks_excluded") for r in runs}
    lines.append(f"- 배경 워밍업 제외 틱 수(run_meta 실측값, run별): {sorted(v for v in wu if v is not None)}")
    lines.append("- 기준: bg_median/bg_std 가 전 채널에서 처음 유효해지는 시각 이전 전부 "
                 "(wp_poes_node.py 실측, 휴리스틱 아님)\n")

    lines.append("## 반복 수")
    if not table1.empty:
        for _, row in table1.iterrows():
            lines.append(f"- 시나리오 {row['scenario']}: n_runs={row['n_runs']}, "
                         f"n_ticks_active={row['n_ticks_active']}, n_activations={row['n_activations']}")
    if bmeta:
        lines.append(f"- 벤치 반복 수: {bmeta.get('repeats')}")
        lines.append(f"- 벤치 워밍업 제외 수: {bmeta.get('warmup_excluded')}")
    lines.append("")

    lines.append("## 버스트 모드")
    if bench is not None and not bench["bench_burst.csv"].empty:
        for _, r in bench["bench_burst.csv"].iterrows():
            lines.append(f"- {r['target']}: n_iter={r['n_iter']}, duration_s={r['duration_s']:.1f}, "
                         f"tegra표본 idle={r['n_tegra_samples_idle']}/burst={r['n_tegra_samples_burst']}, "
                         f"energy/op={r['energy_per_op_mJ']:.4f}mJ, "
                         f"cpu-초/op(독립추정)={r['cpu_seconds_per_op']:.6f}s")
    else:
        lines.append("- 버스트 결과 없음(--burst 없이 실행됐거나 tegrastats 미가용)")
    lines.append("")

    lines.append("## 계측 오버헤드 ((b) 로깅 on/off 차이)")
    nolog_runs = [r for r in runs if is_nolog(r["meta"]) and canonical_scenario(r["meta"]) == "b"]
    withlog_runs = [r for r in runs if not is_nolog(r["meta"]) and not is_speedsanity(r["meta"])
                    and canonical_scenario(r["meta"]) == "b"
                    and r["meta"].get("window") == (nolog_runs[0]["meta"].get("window") if nolog_runs else None)]
    if nolog_runs and withlog_runs:
        def _m3_median(r):
            df = load_csv_trimmed(r["rdir"], "log.csv", r["meta"])
            return float(pd.to_numeric(df["fsm_eval_ms"], errors="coerce").median()) if "fsm_eval_ms" in df.columns and len(df) else np.nan
        m_with = _m3_median(withlog_runs[0])
        m_without = _m3_median(nolog_runs[0])
        lines.append(f"- 같은 구간({nolog_runs[0]['meta'].get('window')}) M3 median: "
                     f"로깅 on={m_with:.5f}ms, 로깅 off={m_without:.5f}ms, "
                     f"차이(계측 비용)={m_with - m_without:.5f}ms" if (m_with == m_with and m_without == m_without)
                     else "- (log.csv 표본 부족으로 비교 불가)")
    else:
        lines.append("- --no-log 대조 run 없음(run_fsm_resource.sh 아직 안 돌림)")
    lines.append("")

    if not rep_var.empty:
        lines.append("## rep 분산 (cv > 10% 항목만 표시)")
        high_cv = rep_var[rep_var["flag_high_cv"] == True]
        if len(high_cv):
            for _, r in high_cv.iterrows():
                lines.append(f"- {r['scenario']}/{r['window']}/{r['metric']}: "
                             f"cv={r['cv']*100:.1f}% (n_reps={r['n_reps']}) -- median 만 싣기 어려움")
        else:
            lines.append("- 모든 항목 cv <= 10%")
        lines.append("")

    if warnings:
        lines.append("## 집계 제외된 run (run_meta.json 없음/파싱 실패)")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    lines.append("## 파생 상수")
    lines.append(f"- TICKS_PER_DAY={TICKS_PER_DAY}, SAMPLES_PER_TICK={SAMPLES_PER_TICK}, "
                 f"GATE_OPENINGS_PER_DAY={GATE_OPENINGS_PER_DAY}(S0.5 확정), "
                 f"DUTY_CYCLE={DUTY_CYCLE}(S0.5 확정)")

    return "\n".join(lines) + "\n"


# ══════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════
def run(results_root: Path) -> dict:
    """전체 집계 실행. 반환: {"table1":df,"table2":df,"table3":df,"note":str,"warnings":list,
    "rep_variance":df} -- 파일 쓰기는 write_outputs() 가 별도로 한다(테스트에서 분리 호출 가능)."""
    runs, warnings = find_runs(results_root)
    if not runs:
        print("[aggregate] no runs found")
        return {"table1": pd.DataFrame(), "table2": pd.DataFrame(), "table3": pd.DataFrame(),
               "note": "# 측정 조건 노트\n\nno runs found\n", "warnings": warnings,
               "rep_variance": pd.DataFrame()}

    for w in warnings:
        print(f"[aggregate] WARNING: {w}")

    bench = load_bench(results_root)
    if bench is None:
        print("[aggregate] WARNING: bench_micro_* 산출물 없음 -- 표2/표3 일부 비어있음")

    table1 = build_table1(runs)
    table2 = build_table2(table1, bench)
    table3 = build_table3(table1, bench)
    rep_var = build_rep_variance(runs)
    note = build_measurement_note(runs, bench, warnings, table1, rep_var)

    return {"table1": table1, "table2": table2, "table3": table3, "note": note,
           "warnings": warnings, "rep_variance": rep_var}


def write_outputs(results_root: Path, result: dict) -> None:
    result["table1"].to_csv(results_root / "onboard_cost.csv", index=False)
    result["table2"].to_csv(results_root / "bench_summary.csv", index=False)

    header = (f"# TICKS_PER_DAY={TICKS_PER_DAY} SAMPLES_PER_TICK={SAMPLES_PER_TICK} "
             f"GATE_OPENINGS_PER_DAY={GATE_OPENINGS_PER_DAY}(S0.5 확정, 재계산 금지) "
             f"DUTY_CYCLE={DUTY_CYCLE}(S0.5 확정, 재계산 금지)\n")
    path3 = results_root / "derived_cost.csv"
    with open(path3, "w", encoding="utf-8", newline="") as f:
        f.write(header)
    result["table3"].to_csv(path3, mode="a", index=False)

    (results_root / "measurement_note.md").write_text(result["note"], encoding="utf-8")
    if not result["rep_variance"].empty:
        result["rep_variance"].to_csv(results_root / "rep_variance.csv", index=False)

    print(f"[aggregate] 저장 완료:")
    print(f"  {results_root / 'onboard_cost.csv'}")
    print(f"  {results_root / 'bench_summary.csv'}")
    print(f"  {results_root / 'derived_cost.csv'}")
    print(f"  {results_root / 'measurement_note.md'}")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="aggregate_onboard -- S5/S6 산출물 집계, 새 측정 없음")
    p.add_argument("results_root", nargs="?", default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    results_root = Path(args.results_root) if args.results_root else (
        Path(__file__).resolve().parents[2] / "results")
    print(f"[aggregate] results_root={results_root}")
    result = run(results_root)
    if result["table1"].empty and not result["warnings"]:
        print("[aggregate] no runs found")
        return
    write_outputs(results_root, result)


if __name__ == "__main__":
    main()
