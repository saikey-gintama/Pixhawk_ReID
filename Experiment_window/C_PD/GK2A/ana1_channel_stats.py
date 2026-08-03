"""
ana1_channel_stats_ksem.py  (축 1 — 채널별 원본 통계)
======================================================
KSEM count 캐시(JSON / Parquet)에서 각 채널의 *원본 그 자체* 통계를 낸다.
카탈로그 비의존 — count 캐시만 사용. 전 기간 통으로 집계.

채널 식별자: (pd_key, side, logic) → 레이블 "PD1_A_O" 형식
  pd_key : PD1 / PD2 / PD3
  side   : A / B
  logic  : O / OU / CR / OUT / F / FT / FTU / FTUO / TRASH

산출:
  channel_stats.csv          채널 × 지표 (median/MAD/robustσ/std/std_mad/p1/p50/p99/
                             max/high_frac/zero_frac/nan_frac/diurnal_amp/n)
  hist_<chan>.png            채널별 분포 히스토그램 (log-x, log-count)
  diurnal_<chan>.png         채널별 UTC-hour median 프로파일
  overview_<chan>.png        채널별 일별 median 시계열 (전 기간 overview)

사용:
  python ana1_channel_stats_ksem.py --cache <json_or_parquet_dir> --out <out_dir>
  python ana1_channel_stats_ksem.py --cache ksem_cache.json --out ./out_ana1
  python ana1_channel_stats_ksem.py --cache ksem_cache_parquet/ --out ./out_ana1 --no-plots
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ksem_io 를 sys.path 에서 찾기 위해 캐시 경로 부모들을 추가하는 헬퍼
def _import_io(io_arg: str, cache_path: str):
    """io 모듈명(또는 슬래시 경로)으로 io 임포트. 캐시 경로 부모들을 sys.path 에
    추가해, 상위 폴더에서 실행해도 하위 폴더의 io 를 찾게 한다. io_arg 미지정 시
    기본 'ksem_io'."""
    name = PureWindowsPath(io_arg).name if (io_arg and ('/' in io_arg or '\\' in io_arg)) else (io_arg or "ksem_io")
    cache = Path(cache_path).resolve()
    for cand in (cache if cache.is_dir() else cache.parent,
                 cache.parent, cache.parent.parent,
                 Path.cwd(), Path(__file__).resolve().parent):
        sp = str(cand)
        if sp not in sys.path:
            sys.path.insert(0, sp)
    import importlib
    return importlib.import_module(name)


def col_to_label(col: tuple) -> str:
    """(pd_key, side, logic) → 'PD1_A_O' 채널 라벨."""
    return "_".join(col)


# ─────────────────────────────────────────────────────────────────
# 통계 연산 (POES ana1 과 동일)
# ─────────────────────────────────────────────────────────────────
def robust_sigma(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def channel_stats(s: pd.Series) -> dict:
    """단일 채널 Series → 기술통계 dict."""
    v = s.to_numpy(dtype=float)
    n_total = len(v)
    finite = v[np.isfinite(v)]
    n = len(finite)
    nan_frac = 1.0 - n / n_total if n_total else np.nan
    if n == 0:
        return {"n": 0, "nan_frac": nan_frac}
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    rsig = 1.4826 * mad
    std = float(np.std(finite))
    # diurnal 진폭: UTC hour 별 median 의 (max-min)
    by_hour = s.dropna().groupby(s.dropna().index.hour).median()
    diurnal_amp = float(by_hour.max() - by_hour.min()) if len(by_hour) else np.nan
    return {
        "n": n,
        "median": round(med, 4),
        "MAD": round(mad, 4),
        "robust_sigma": round(rsig, 4),
        "std": round(std, 4),
        "std_over_mad": round(std / rsig, 3) if rsig > 0 else np.nan,
        "p1": round(float(np.percentile(finite, 1)), 4),
        "p50": round(med, 4),
        "p99": round(float(np.percentile(finite, 99)), 4),
        "max": round(float(np.max(finite)), 4),
        "high_frac": round(float(np.mean(finite > 0)), 4),
        "zero_frac": round(float(np.mean(finite == 0)), 4),
        "nan_frac": round(nan_frac, 4),
        "diurnal_amp": round(diurnal_amp, 4) if np.isfinite(diurnal_amp) else np.nan,
    }


# ─────────────────────────────────────────────────────────────────
# 그림 (POES ana1 과 동일 구성)
# ─────────────────────────────────────────────────────────────────
def plot_hist(s: pd.Series, name: str, out: Path):
    v = s.dropna().to_numpy()
    v = v[v > 0]   # log-x 위해 양수만
    if len(v) < 10:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(v, bins=np.logspace(np.log10(v.min() + 1e-9), np.log10(v.max() + 1e-9), 60),
            color="#2980b9", alpha=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("count rate (#/s)"); ax.set_ylabel("frequency")
    ax.set_title(f"{name} — distribution (log-log)", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out / f"hist_{name}.png", dpi=110); plt.close(fig)


def plot_diurnal(s: pd.Series, name: str, out: Path):
    ss = s.dropna()
    if ss.empty:
        return
    by_hour = ss.groupby(ss.index.hour).median()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(by_hour.index, by_hour.values, "o-", color="#c0392b")
    ax.set_xlabel("UTC hour"); ax.set_ylabel("median count rate")
    ax.set_title(f"{name} — diurnal median profile", fontsize=9)
    ax.set_xticks(range(0, 24, 3)); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out / f"diurnal_{name}.png", dpi=110); plt.close(fig)


def plot_overview(s: pd.Series, name: str, out: Path):
    ss = s.dropna()
    if ss.empty:
        return
    daily = ss.resample("1D").median()
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(daily.index, daily.values, lw=0.7, color="#27ae60")
    ax.set_yscale("symlog", linthresh=1e-2)
    ax.set_ylabel("daily median"); ax.set_title(f"{name} — full-period overview", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out / f"overview_{name}.png", dpi=110); plt.close(fig)


# ─────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="축1 채널별 원본 통계 (KSEM)")
    p.add_argument("--cache", required=True,
                   help="KSEM 캐시 경로 (JSON 파일 또는 Parquet 디렉터리)")
    p.add_argument("--io", default="ksem_io",
                   help="io 모듈명 또는 경로 (기본 ksem_io). 예: KSEM_count\\ksem_io")
    p.add_argument("--out",   required=True, help="출력 디렉터리")
    p.add_argument("--no-plots", action="store_true", help="그림 생략, CSV만")
    args = p.parse_args()

    io  = _import_io(args.io, args.cache)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    df, meta = io.load(args.cache)
    if df.empty:
        print("[ERROR] 빈 캐시"); return
    print(f"[ana1] {df.shape[1]} 채널, {len(df):,} rows")

    rows = []
    for col in df.columns:
        pd_key, side, logic = col          # KSEM 3-tuple 언패킹
        name = col_to_label(col)           # "PD1_A_O"
        s = df[col]
        st = channel_stats(s)
        st = {
            "channel": name,
            "pd_key":  pd_key,
            "side":    side,
            "logic":   logic,
            **st,
        }
        rows.append(st)
        print(f"  {name:16s} median={st.get('median','?')} "
              f"std/MAD={st.get('std_over_mad','?')} high_frac={st.get('high_frac','?')}")
        if not args.no_plots:
            plot_hist(s, name, out)
            plot_diurnal(s, name, out)
            plot_overview(s, name, out)

    stats = pd.DataFrame(rows)
    csv_path = out / "channel_stats.csv"
    stats.to_csv(csv_path, index=False)
    print(f"[ana1] CSV 저장: {csv_path}  ({len(stats)} 채널)")
    if not args.no_plots:
        print(f"[ana1] 그림: hist_*/diurnal_*/overview_* → {out}")


if __name__ == "__main__":
    main()
