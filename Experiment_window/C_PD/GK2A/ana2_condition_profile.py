"""
ana2_condition_profile_ksem.py  (축 2 — 조건별 count 프로파일)
=============================================================
KSEM count 캐시에서 채널 count 를 조건(condition)별로 분류해 분포를 비교한다.
*매칭이 아니라* 구간 통계다.

POES ana2 와의 차이:
  - geo 조건(quiet / saa / highlat) 제거 — KSEM 은 정지궤도, geo 없음
  - 이벤트 조건(spe / espe) 및 마스크 로직은 완전히 동일
  - io 고정(ksem_io), 채널 라벨 col_to_label() 사용

조건:
  background : 비이벤트 전 구간 (spe/espe 모두 False)   ← quiet 대응
  spe        : NOAA SPE 활성구간 (begin_time ~ max_time, 상승구간만)
  espe       : SWPC ESPE 활성구간 (begin_time ~ max_time)

핵심 지표 (POES ana2 와 동일한 의미):
  spe_over_background   : SPE median / background median
  espe_over_background  : ESPE median / background median
  spe_over_espe         : SPE median / ESPE median   (← saa_over_spe 대응)

산출:
  condition_profile.csv   채널 × 조건 median/p90/p99/n + 위 지표
  cond_box_<chan>.png     채널별 조건간 분포 박스플롯(log-y)
  (--no-plots 로 그림 생략)

사용:
  python ana2_condition_profile_ksem.py \
      --cache <json_or_parquet> --out <out_dir> \
      --spe <noaa_spe_parquet_dir> --espe <swpc_espe_parquet_dir> [--no-plots]
"""
from __future__ import annotations
import argparse, importlib, sys
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────
# io 임포트 헬퍼
# ─────────────────────────────────────────────────────────────────
def _import_io(io_arg: str, cache_path: str):
    """io 모듈명(또는 슬래시 경로)으로 io 임포트. 기본 'ksem_io'."""
    name = PureWindowsPath(io_arg).name if (io_arg and ('/' in io_arg or '\\' in io_arg)) else (io_arg or "ksem_io")
    cache = Path(cache_path).resolve()
    for cand in (cache if cache.is_dir() else cache.parent,
                 cache.parent, cache.parent.parent,
                 Path.cwd(), Path(__file__).resolve().parent):
        sp = str(cand)
        if sp not in sys.path:
            sys.path.insert(0, sp)
    return importlib.import_module(name)


def _import_module(mod_arg: str, *extra_dirs):
    """이벤트 io 모듈 임포트 (POES ana2 와 동일)."""
    name = PureWindowsPath(mod_arg).name if ('/' in mod_arg or '\\' in mod_arg) else mod_arg
    for cand in extra_dirs:
        sp = str(Path(cand).resolve())
        if sp not in sys.path:
            sys.path.insert(0, sp)
    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))
    return importlib.import_module(name)


def col_to_label(col: tuple) -> str:
    """(pd_key, side, logic) → 'PD1_A_O'."""
    return "_".join(col)


# ─────────────────────────────────────────────────────────────────
# 이벤트 마스크 — POES ana2 와 완전 동일
# ─────────────────────────────────────────────────────────────────
def _event_mask(index: pd.DatetimeIndex, ev_df: pd.DataFrame) -> pd.Series:
    """이벤트 df(begin_time 인덱스 + max_time 컬럼)의 begin~max 상승구간 합집합 마스크."""
    # 인덱스 tz 정규화: naive면 UTC 부여, aware면 UTC 변환 (KSEM 캐시는 naive일 수 있음)
    idx = pd.DatetimeIndex(index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    mask = pd.Series(False, index=idx)
    if ev_df is None or ev_df.empty:
        mask.index = index
        return mask
    for begin, row in ev_df.iterrows():
        mx = row.get("max_time")
        if pd.isna(begin) or pd.isna(mx):
            continue
        b = pd.Timestamp(begin); e = pd.Timestamp(mx)
        if b.tz is None: b = b.tz_localize("UTC")
        else: b = b.tz_convert("UTC")
        if e.tz is None: e = e.tz_localize("UTC")
        else: e = e.tz_convert("UTC")
        if e < b:
            continue
        mask |= (idx >= b) & (idx <= e)
    mask.index = index   # 원래 인덱스로 복귀(호출부의 reindex 와 일치)
    return mask


# ─────────────────────────────────────────────────────────────────
# 통계 — POES ana2 와 완전 동일
# ─────────────────────────────────────────────────────────────────
def _stats(s: pd.Series) -> dict:
    v = s.dropna().to_numpy()
    if len(v) == 0:
        return {"med": np.nan, "p90": np.nan, "p99": np.nan, "n": 0}
    return {"med": float(np.median(v)),
            "p90": float(np.percentile(v, 90)),
            "p99": float(np.percentile(v, 99)),
            "n": int(len(v))}


# ─────────────────────────────────────────────────────────────────
# 프로파일 빌드
# ─────────────────────────────────────────────────────────────────
def build_profile(df: pd.DataFrame, spe_mask: pd.Series, espe_mask: pd.Series) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        pd_key, side, logic = col
        name = col_to_label(col)
        s = df[col].dropna()
        if s.empty:
            continue

        spe  = spe_mask.reindex(s.index).fillna(False)
        espe = espe_mask.reindex(s.index).fillna(False)
        bg   = (~spe) & (~espe)          # background = 비이벤트 전 구간

        st = {c: _stats(s[m]) for c, m in
              [("background", bg), ("spe", spe), ("espe", espe)]}

        q = st["background"]["med"]      # background = POES quiet 대응
        row = {
            "channel": name,
            "pd_key":  pd_key,
            "side":    side,
            "logic":   logic,
        }
        for c in ("background", "spe", "espe"):
            row[f"{c}_med"] = round(st[c]["med"], 4) if np.isfinite(st[c]["med"]) else np.nan
            row[f"{c}_p99"] = round(st[c]["p99"], 4) if np.isfinite(st[c]["p99"]) else np.nan
            row[f"{c}_n"]   = st[c]["n"]

        def ratio(a):
            return round(a / q, 2) if (np.isfinite(a) and np.isfinite(q) and q > 0) else np.nan

        row["spe_over_background"]  = ratio(st["spe"]["med"])
        row["espe_over_background"] = ratio(st["espe"]["med"])
        sp = st["spe"]["med"]
        row["spe_over_espe"] = round(st["spe"]["med"] / st["espe"]["med"], 2) \
            if (np.isfinite(sp) and np.isfinite(st["espe"]["med"])
                and st["espe"]["med"] > 0) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
# 박스플롯 — POES ana2 와 동일 구성
# ─────────────────────────────────────────────────────────────────
def plot_box(df: pd.DataFrame, spe_mask: pd.Series, espe_mask: pd.Series, out: Path):
    for col in df.columns:
        name = col_to_label(col)
        s = df[col].dropna()
        if s.empty:
            continue

        spe  = spe_mask.reindex(s.index).fillna(False)
        espe = espe_mask.reindex(s.index).fillna(False)
        bg   = (~spe) & (~espe)

        data, labels = [], []
        for c, m in [("background", bg), ("spe", spe), ("espe", espe)]:
            vals = s[m].to_numpy()
            vals = vals[vals > 0]
            if len(vals) > 0:
                data.append(vals); labels.append(f"{c}\n(n={len(vals)})")
        if len(data) < 2:
            continue

        fig, ax = plt.subplots(figsize=(7, 4))
        bp = ax.boxplot(data, showfliers=False)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_yscale("log")
        ax.set_ylabel("count rate (#/s, >0)")
        ax.set_title(f"{name} — count by condition", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(out / f"cond_box_{name}.png", dpi=110); plt.close(fig)


# ─────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="축2 조건별 count 프로파일 (KSEM)")
    p.add_argument("--cache",    required=True,
                   help="KSEM 캐시 경로 (JSON 파일 또는 Parquet 디렉터리)")
    p.add_argument("--io",       default="ksem_io",
                   help="io 모듈명 또는 경로 (기본 ksem_io). 예: KSEM_count\\ksem_io")
    p.add_argument("--out",      required=True, help="출력 디렉터리")
    p.add_argument("--spe",      help="NOAA SPE parquet 디렉터리")
    p.add_argument("--espe",     help="SWPC ESPE parquet 디렉터리")
    p.add_argument("--spe-io",   default="noaa_goes_spe_io")
    p.add_argument("--espe-io",  default="swpc_alert_espe_io")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()

    io  = _import_io(args.io, args.cache)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    df, meta = io.load(args.cache)
    if df.empty:
        print("[ERROR] 빈 캐시"); return
    print(f"[ana2] {df.shape[1]} 채널, {len(df):,} rows")

    # 이벤트 카탈로그 로드 → 상승구간 마스크 (POES ana2 와 동일)
    spe_mask  = pd.Series(False, index=df.index)
    espe_mask = pd.Series(False, index=df.index)
    if args.spe:
        spe_io  = _import_module(args.spe_io,
                                 Path(args.spe).parent, Path(args.spe).parent.parent)
        spe_df, _ = spe_io.load(args.spe)
        spe_mask  = _event_mask(df.index, spe_df)
        print(f"[ana2] SPE 이벤트 {len(spe_df)}개 → 활성 {int(spe_mask.sum()):,} min")
    if args.espe:
        espe_io = _import_module(args.espe_io,
                                 Path(args.espe).parent, Path(args.espe).parent.parent)
        espe_df, _ = espe_io.load(args.espe)
        espe_mask  = _event_mask(df.index, espe_df)
        print(f"[ana2] ESPE 이벤트 {len(espe_df)}개 → 활성 {int(espe_mask.sum()):,} min")

    prof = build_profile(df, spe_mask, espe_mask)
    prof.to_csv(out / "condition_profile.csv", index=False)
    print(f"[ana2] condition_profile.csv ({len(prof)} 채널)")

    show = ["channel", "background_med", "spe_med", "espe_med",
            "spe_over_background", "spe_over_espe"]
    show = [c for c in show if c in prof.columns]
    print(prof[show].to_string(index=False))

    if not args.no_plots:
        plot_box(df, spe_mask, espe_mask, out)
        print(f"[ana2] cond_box_* 그림 → {out}")


if __name__ == "__main__":
    main()