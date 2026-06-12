#!/usr/bin/env python3
"""
aggregate_resource.py  —  14개 run 폴더를 5.1절용 요약 표로 집계
============================================================================
각 run 폴더의 log.csv(fsm HK) + event_log.csv(offboard action) 를 읽어
설정(W30/W7)별로 다음을 산출:

  latency (per-tick, active 구간만):
    wp_eval_ms, dds_transport_ms, fsm_infer_ms, node_latency_ms, action_ms
    → median / mean / p95
  resource (tegrastats, active 구간만):
    power_mW(VDD_IN), gpu_%, cpu_%, ram_MB → mean / p95
  rep 분산: REP 윈도우(W3) 의 rep 간 표준편차

active 구간만 쓰는 이유: warm-up(x7200) 은 타이머가 너무 빨라 측정 의미가
없고, 초록·5.1 의 "continuous triggering" 주장은 정상상태(active) 비용이다.
warm-up/active 경계는 ksem 로그의 배속 전환 시점 = 첫 이벤트 이후.
여기서는 보수적으로 각 run 의 뒤쪽 active 구간을 분리하기 위해
fsm log.csv 의 timestamp(wall-clock) 간격이 급변하는 지점을 경계로 잡는다.
(warm-up 은 tick 간격 900/7200=0.125s, active 는 900/1800=0.5s)
"""
from __future__ import annotations
import sys
import os
import glob
import re
import numpy as np
import pandas as pd

LAT_COLS = ["wp_eval_ms", "dds_transport_ms", "fsm_infer_ms", "node_latency_ms"]
RES_COLS = {"power_mW": "power_mW", "gpu_%": "gpu_%", "cpu_%": "cpu_%", "ram_MB": "ram_MB"}


def _split_active(df: pd.DataFrame) -> pd.DataFrame:
    """tick 간 wall-clock 간격으로 warm-up/active 분리. active(느린쪽)만 반환."""
    if "timestamp" not in df.columns or len(df) < 10:
        return df
    dt = df["timestamp"].diff()
    # active 는 간격이 큰 쪽(0.5s 부근). warm-up 은 0.125s 부근.
    # 중앙값 기준 위/아래로 가르되, 전환점 이후를 active 로.
    thr = dt.median()
    # 전환점: dt 가 thr 보다 확실히 커지기 시작하는 첫 지점
    active_mask = dt > (thr * 2.0)
    if active_mask.sum() < 5:
        # 분리 실패(단일 배속이었거나 데이터 적음) → 전체 사용
        return df
    first_active = active_mask.idxmax()
    return df.loc[first_active:]


def _stats(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return {"n": 0, "median": np.nan, "mean": np.nan, "p95": np.nan}
    return {
        "n": len(s),
        "median": float(np.median(s)),
        "mean": float(np.mean(s)),
        "p95": float(np.percentile(s, 95)),
    }


def parse_run(folder: str) -> dict | None:
    log = os.path.join(folder, "log.csv")
    if not os.path.exists(log):
        return None
    name = os.path.basename(folder.rstrip("/"))
    m = re.search(r"(W\d+_[\d-]+)_W(\d+)_rep(\d+)", name)
    label = m.group(1) if m else name
    wd = int(m.group(2)) if m else -1
    rep = int(m.group(3)) if m else 1

    df = pd.read_csv(log)
    df = _split_active(df)

    rec = {"folder": name, "label": label, "window_d": wd, "rep": rep,
           "n_active": len(df)}
    for c in LAT_COLS:
        if c in df.columns:
            st = _stats(df[c])
            rec[f"{c}_median"] = st["median"]
            rec[f"{c}_mean"] = st["mean"]
            rec[f"{c}_p95"] = st["p95"]
    for c in RES_COLS:
        if c in df.columns:
            st = _stats(df[c])
            rec[f"{c}_mean"] = st["mean"]
            rec[f"{c}_p95"] = st["p95"]

    # action_ms (offboard) — 상태전이 시에만 기록되므로 평균만
    ev = os.path.join(folder, "event_log.csv")
    if os.path.exists(ev):
        edf = pd.read_csv(ev)
        if "action_ms" in edf.columns:
            st = _stats(edf["action_ms"])
            rec["action_ms_median"] = st["median"]
            rec["action_ms_mean"] = st["mean"]
            rec["action_ms_n"] = st["n"]
    return rec


def main(results_root: str):
    folders = sorted(glob.glob(os.path.join(results_root, "*_W*_rep*")))
    if not folders:
        # 폴더 명명이 다를 수 있으니 log.csv 가진 모든 하위폴더
        folders = sorted({os.path.dirname(p) for p in
                          glob.glob(os.path.join(results_root, "*", "log.csv"))})
    rows = [r for f in folders if (r := parse_run(f)) is not None]
    if not rows:
        print(f"No runs found under {results_root}", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)
    out_dir = results_root
    raw_path = os.path.join(out_dir, "resource_runs_raw.csv")
    df.to_csv(raw_path, index=False)
    print(f"[written] {raw_path}  ({len(df)} runs)")
    print()

    # ── 설정별(W30/W7) 요약: run 평균의 평균 ──
    print("=" * 72)
    print("Section 5.1 요약 — 설정별 (active 구간, run 평균)")
    print("=" * 72)
    summary_rows = []
    for wd in sorted(df["window_d"].unique()):
        sub = df[df["window_d"] == wd]
        row = {"window_d": wd, "n_runs": len(sub)}
        for c in LAT_COLS:
            col = f"{c}_median"
            if col in sub:
                row[f"{c}_median(med of runs)"] = round(sub[col].mean(), 4)
            col95 = f"{c}_p95"
            if col95 in sub:
                row[f"{c}_p95(mean of runs)"] = round(sub[col95].mean(), 4)
        for c in RES_COLS:
            col = f"{c}_mean"
            if col in sub:
                row[f"{c}_mean"] = round(sub[col].mean(), 2)
        summary_rows.append(row)
    summ = pd.DataFrame(summary_rows)
    summ_path = os.path.join(out_dir, "resource_summary_by_config.csv")
    summ.to_csv(summ_path, index=False)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(summ.to_string(index=False))
    print(f"\n[written] {summ_path}")
    print()

    # ── W30 vs W7 핵심 대비 (wp_eval = rolling MAD 비용) ──
    print("=" * 72)
    print("W30 vs W7 핵심 대비 (5.1: 윈도우 축소 시 비용 변화)")
    print("=" * 72)
    for c in ["wp_eval_ms_median", "fsm_infer_ms_median", "dds_transport_ms_median"]:
        if c in df.columns:
            g = df.groupby("window_d")[c].mean()
            if 30 in g.index and 7 in g.index and g[30] > 0:
                ratio = g[7] / g[30]
                print(f"  {c:28s}  W30={g[30]:.4f}  W7={g[7]:.4f}  (W7/W30={ratio:.2f})")

    # ── rep 분산 (W3) ──
    print()
    print("=" * 72)
    print("rep-to-rep 분산 (REP 윈도우, power/latency 재현성)")
    print("=" * 72)
    rep_lbl = df["label"].value_counts().idxmax()
    repdf = df[df["label"] == rep_lbl]
    for wd in sorted(repdf["window_d"].unique()):
        sub = repdf[repdf["window_d"] == wd]
        if len(sub) < 2:
            continue
        print(f"  {rep_lbl} W{wd} ({len(sub)} reps):")
        for c in ["wp_eval_ms_median", "fsm_infer_ms_median", "power_mW_mean"]:
            if c in sub.columns:
                vals = pd.to_numeric(sub[c], errors="coerce").dropna()
                if len(vals) >= 2:
                    print(f"     {c:24s} mean={vals.mean():.4f}  std={vals.std(ddof=1):.4f}  "
                          f"cv={100*vals.std(ddof=1)/vals.mean():.1f}%")
    print()
    print("done.")


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    main(root)
