"""
_speed_sanity_compare.py
=========================
S5.1 배속 타당성 대조. scenario (b) 를 seg_strong 에서 1800x/7200x 로 각 1회
돌린 두 log.csv(M1~M5 per-tick) 를 받아 M1/M2/M3/M5 median 을 비교해
speed_sanity.csv 로 낸다.

근거: 배속 전환은 타이머 주기만 바꾸고 판정·측정 입력이 아니므로(md 원칙),
지연(M1~M10)은 배속과 무관하게 유효해야 한다. 두 배속의 median 이 상대오차
--tol(기본 20%) 이내면 PASS -- 7200x 채택 근거. 아니면 FAIL -- 7200x 를 버리고
1800x 로 되돌릴 근거(run_fsm_resource.sh/_run_common.sh 의 ACTIVE_SPEED 수정).

사용: python3 _speed_sanity_compare.py <rdir_1800x> <rdir_7200x> [--out path] [--tol 0.2]
"""
from __future__ import annotations

import argparse
import csv as csvmod
from pathlib import Path

import pandas as pd

METRICS = [("resample_ms", "M1"), ("z_eval_ms", "M2"),
          ("fsm_eval_ms", "M3"), ("dds_transport_ms", "M5")]


def median_metrics(log_csv: Path) -> dict:
    if not log_csv.exists():
        return {}
    df = pd.read_csv(log_csv)
    out = {}
    for col, label in METRICS:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            out[label] = float(vals.median()) if len(vals) else float("nan")
        else:
            out[label] = float("nan")
    return out


def main():
    ap = argparse.ArgumentParser(description="S5.1 배속 타당성 대조(1800x vs 7200x)")
    ap.add_argument("rdir_1800")
    ap.add_argument("rdir_7200")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tol", type=float, default=0.20, help="상대오차 허용치(기본 20%%)")
    args = ap.parse_args()

    rdir_1800, rdir_7200 = Path(args.rdir_1800), Path(args.rdir_7200)
    m1800 = median_metrics(rdir_1800 / "log.csv")
    m7200 = median_metrics(rdir_7200 / "log.csv")

    out_path = Path(args.out) if args.out else rdir_7200.parent / "speed_sanity.csv"
    rows = []
    all_pass = True
    for _, metric in METRICS:
        v1800 = m1800.get(metric, float("nan"))
        v7200 = m7200.get(metric, float("nan"))
        if v1800 == v1800 and v1800 != 0:
            rel_diff = abs(v7200 - v1800) / abs(v1800)
        else:
            rel_diff = float("nan")
        passed = (rel_diff == rel_diff) and (rel_diff <= args.tol)
        all_pass = all_pass and passed
        rows.append({"metric": metric, "median_1800x_ms": v1800, "median_7200x_ms": v7200,
                    "rel_diff": rel_diff, "tol": args.tol, "pass": passed,
                    "rdir_1800": str(rdir_1800), "rdir_7200": str(rdir_7200)})

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csvmod.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[speed_sanity] 저장 -> {out_path}")
    for r in rows:
        print(f"  {r['metric']}: 1800x={r['median_1800x_ms']:.4f}ms  7200x={r['median_7200x_ms']:.4f}ms  "
              f"rel_diff={r['rel_diff']:.3f}(tol={args.tol})  {'PASS' if r['pass'] else 'FAIL'}")
    verdict = "PASS -- 7200x 채택 근거(배속이 지연 측정을 왜곡하지 않음)" if all_pass \
        else "FAIL -- 7200x 를 버리고 1800x 로 되돌릴 근거"
    print(f"[speed_sanity] 종합: {verdict}")


if __name__ == "__main__":
    main()
