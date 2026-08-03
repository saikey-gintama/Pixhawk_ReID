"""
ana3_coord_dependence_ksem.py  (축 3 — 채널 계층 의존성 + 스파이크 원인 규명)
==============================================================================
KSEM 은 정지궤도 계측기이므로 위성 궤도 geo(Bmag/maglat/lat/lon) 가 없다.
POES ana3 의 역할 — "극단값이 어느 조건에서 나오는가" — 을 KSEM 의
채널 분류 계층으로 대응시킨다:

  POES geo 축         →  KSEM 채널 계층 축
  ──────────────         ────────────────────────────────────────────────
  SAA(|B|<25000)      →  TRASH logic  (노이즈·관통 입자 오염 채널)
  high_lat(>60°)      →  FT/FTU/FTUO  (전자 채널 그룹 — 고에너지 민감)
  quiet (나머지)      →  O/OU/CR/OUT  (양성자 채널 그룹 — 비교 기준)

"스파이크" 기준:  max > (p99 × --spike-factor, 기본 10)

산출:
  spike_origin.csv     채널별 상위 극단값(top-K) 의 logic 그룹 집계 +
                       max 한 점 상세 (시각, 값, logic_group)
  logic_profile.csv    채널 × logic 그룹 median count (maglat_profile 대응)
  group_summary.csv    채널별: trash_med / electron_med / proton_med /
                       trash_over_proton / electron_over_proton 등
                       (coord_summary 대응)
  logic_profile_<chan>.png   채널별 logic 그룹 막대 프로파일  (maglat_profile 대응)
  heatmap_pd_logic.png       PD × logic 2D 평균 count 히트맵  (lat-lon heatmap 대응)
  (--no-plots 로 그림 생략)

사용:
  python ana3_coord_dependence_ksem.py --cache <json_or_parquet> --out <out_dir>
  python ana3_coord_dependence_ksem.py --cache ksem_cache.json --out ./out_ana3 \\
      --topk 50 --spike-factor 10
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────
# ksem_io 동적 임포트
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
    import importlib
    return importlib.import_module(name)


def col_to_label(col: tuple) -> str:
    """(pd_key, side, logic) → 'PD1_A_O'."""
    return "_".join(col)


# ─────────────────────────────────────────────────────────────────
# KSEM logic 그룹 정의  (POES geo 태그에 대응)
# ─────────────────────────────────────────────────────────────────
#   POES: SAA         ←→  KSEM: TRASH  (오염/관통 이벤트 기원 가능성)
#   POES: high_lat    ←→  KSEM: electron  (F/FT/FTU/FTUO)
#   POES: quiet       ←→  KSEM: proton    (O/OU/CR/OUT)
TRASH_LOGICS    = {"TRASH"}
ELECTRON_LOGICS = {"F", "FT", "FTU", "FTUO"}
PROTON_LOGICS   = {"O", "OU", "CR", "OUT"}

def logic_group(logic: str) -> str:
    if logic in TRASH_LOGICS:    return "trash"
    if logic in ELECTRON_LOGICS: return "electron"
    if logic in PROTON_LOGICS:   return "proton"
    return "other"


# ─────────────────────────────────────────────────────────────────
# 분석 함수
# ─────────────────────────────────────────────────────────────────
def spike_origin(df: pd.DataFrame, topk: int, spike_factor: float):
    """
    채널별 상위 topk 극단값의 logic_group 집계 + max 한 점 상세.
    → POES spike_origin.csv 와 동일 구조.
    """
    rows = []
    for col in df.columns:
        pd_key, side, logic = col
        name  = col_to_label(col)
        grp   = logic_group(logic)
        s     = df[col].dropna()
        if s.empty:
            continue

        # 스파이크 임계: p99 × spike_factor
        p99   = float(np.percentile(s.values, 99))
        spike_thr = p99 * spike_factor

        top   = s.nlargest(topk)
        # logic_group 은 채널 자신에 속하므로 "같은 그룹의 채널에서 왔는가"
        # 대신 이 채널의 top-K 가 스파이크 임계를 넘는지 집계
        in_trash    = int(grp == "trash")      * topk  # 채널 자체가 trash 이면 전부
        in_electron = int(grp == "electron")   * topk
        in_proton   = int(grp == "proton")     * topk
        # 임계 초과 개수 (그룹 무관, 절댓값 스파이크 식별)
        n_spike = int((top > spike_thr).sum()) if spike_thr > 0 else 0

        mx_val  = float(s.max())
        mx_time = s.idxmax()

        rows.append({
            "channel":       name,
            "pd_key":        pd_key,
            "side":          side,
            "logic":         logic,
            "logic_group":   grp,
            "max":           round(mx_val, 2),
            "max_time":      mx_time,
            "p99":           round(p99, 4),
            "spike_thr":     round(spike_thr, 4),
            "max_is_spike":  bool(mx_val > spike_thr),
            f"top{topk}_in_trash":    in_trash    if grp == "trash"    else
                                      int((top > spike_thr).sum()),  # 다른 그룹은 임계 초과 수
            f"top{topk}_in_electron": in_electron if grp == "electron" else
                                      int((top > spike_thr).sum()),
            f"top{topk}_in_proton":   in_proton   if grp == "proton"   else
                                      int((top > spike_thr).sum()),
            f"top{topk}_n_spike":     n_spike,
        })
    return pd.DataFrame(rows)


def logic_profile(df: pd.DataFrame):
    """
    채널 × logic-group median count.
    → POES maglat_profile.csv 대응 (행: logic_group, 열: 채널)
    """
    groups = ["proton", "electron", "trash", "other"]
    out = {}
    for col in df.columns:
        name = col_to_label(col)
        grp  = logic_group(col[2])
        s    = df[col].dropna()
        # 자기 채널의 median 을 소속 그룹 행에 기록
        med_by_grp = {g: np.nan for g in groups}
        if not s.empty:
            med_by_grp[grp] = float(s.median())
        out[name] = pd.Series(med_by_grp)
    prof = pd.DataFrame(out)
    prof.index.name = "logic_group"
    return prof


def group_summary(df: pd.DataFrame):
    """
    채널별: proton / electron / trash 그룹 median 및 상대 비율.
    → POES coord_summary.csv 대응.
    """
    # 그룹별로 채널들의 median 평균을 내어 대표값으로 사용
    group_dfs = {"proton": [], "electron": [], "trash": []}
    for col in df.columns:
        grp = logic_group(col[2])
        if grp in group_dfs:
            s = df[col].dropna()
            if not s.empty:
                group_dfs[grp].append(s)

    def grp_median(series_list):
        if not series_list:
            return np.nan
        combined = pd.concat(series_list)
        return float(combined.median())

    p_med = grp_median(group_dfs["proton"])
    e_med = grp_median(group_dfs["electron"])
    t_med = grp_median(group_dfs["trash"])

    rows = []
    for col in df.columns:
        pd_key, side, logic = col
        name = col_to_label(col)
        grp  = logic_group(logic)
        s    = df[col].dropna()
        ch_med = float(s.median()) if not s.empty else np.nan

        rows.append({
            "channel":               name,
            "pd_key":                pd_key,
            "side":                  side,
            "logic":                 logic,
            "logic_group":           grp,
            "channel_med":           round(ch_med, 4) if np.isfinite(ch_med) else np.nan,
            "proton_grp_med":        round(p_med, 4)  if np.isfinite(p_med)  else np.nan,
            "electron_grp_med":      round(e_med, 4)  if np.isfinite(e_med)  else np.nan,
            "trash_grp_med":         round(t_med, 4)  if np.isfinite(t_med)  else np.nan,
            # 비율 (proton 을 기준 'quiet' 로 사용)
            "electron_over_proton":  round(e_med / p_med, 2)
                                     if (np.isfinite(e_med) and np.isfinite(p_med) and p_med > 0)
                                     else np.nan,
            "trash_over_proton":     round(t_med / p_med, 2)
                                     if (np.isfinite(t_med) and np.isfinite(p_med) and p_med > 0)
                                     else np.nan,
            "n_channel":             int(s.notna().sum()),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
# 그림
# ─────────────────────────────────────────────────────────────────
def plot_logic_profile(prof: pd.DataFrame, name: str, out: Path):
    """채널별 logic-group 막대 프로파일 (maglat_<chan>.png 대응)."""
    if name not in prof.columns:
        return
    col = prof[name].dropna()
    if col.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(col.index, col.values, color="#8e44ad", alpha=0.8)
    ax.set_xlabel("logic group"); ax.set_ylabel("median count rate")
    ax.set_title(f"{name} — logic group profile", fontsize=9)
    # 값 레이블
    for bar, val in zip(bars, col.values):
        if np.isfinite(val):
            ax.text(bar.get_x() + bar.get_width() / 2, val,
                    f"{val:.3g}", ha="center", va="bottom", fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / f"maglat_{name}.png", dpi=110); plt.close(fig)


def plot_heatmap_pd_logic(df: pd.DataFrame, out: Path):
    """
    PD(행) × logic(열) 2D 평균 count 히트맵.
    → POES heatmap_<chan>.png (lat-lon) 대응 — 채널당 1장 대신 전체 1장.
    """
    from ksem_io import PD_KEYS, LOGICS  # noqa: 상수 재사용
    grid = np.full((len(PD_KEYS), len(LOGICS)), np.nan)
    for ci, pd_key in enumerate(PD_KEYS):
        for li, logic in enumerate(LOGICS):
            # A/B 양쪽 평균
            vals = []
            for side in ("A", "B"):
                col = (pd_key, side, logic)
                if col in df.columns:
                    s = df[col].dropna()
                    if not s.empty:
                        vals.append(float(s.mean()))
            if vals:
                grid[ci, li] = float(np.mean(vals))

    fig, ax = plt.subplots(figsize=(10, 3.5))
    # 0 / nan 혼재하므로 symlog norm 사용
    vmin = np.nanmin(grid[grid > 0]) if np.any(grid > 0) else 1e-3
    vmax = np.nanmax(grid)           if np.any(np.isfinite(grid)) else 1.0
    norm = matplotlib.colors.SymLogNorm(linthresh=max(vmin, 1e-3),
                                        vmin=0, vmax=vmax)
    im = ax.pcolormesh(np.arange(len(LOGICS) + 1),
                       np.arange(len(PD_KEYS) + 1),
                       grid, shading="flat", norm=norm, cmap="viridis")
    ax.set_xticks(np.arange(len(LOGICS)) + 0.5); ax.set_xticklabels(LOGICS, fontsize=8)
    ax.set_yticks(np.arange(len(PD_KEYS)) + 0.5); ax.set_yticklabels(PD_KEYS, fontsize=8)
    ax.set_xlabel("logic"); ax.set_ylabel("pd_key")
    ax.set_title("mean count by PD × logic  (A+B average)", fontsize=9)
    fig.colorbar(im, ax=ax, label="mean count rate")
    fig.tight_layout()
    # 채널당 heatmap 파일명 규칙("heatmap_<chan>.png")을 유지하되
    # KSEM 은 전체 1장으로 대체 → heatmap_pd_logic.png
    fig.savefig(out / "heatmap_pd_logic.png", dpi=110); plt.close(fig)

    # 채널별 개별 heatmap (POES 규칙 완전 유지 — A vs B 만 있어 1×2)
    for col in df.columns:
        name   = col_to_label(col)
        pd_key, side, logic = col
        s      = df[col].dropna()
        if s.empty:
            continue
        _plot_single_channel_heatmap(df, pd_key, logic, name, out)


def _plot_single_channel_heatmap(df: pd.DataFrame, pd_key: str, logic: str,
                                  name: str, out: Path):
    """
    단일 채널(pd_key × logic)에 대한 side(A/B) × 시간-bin 히트맵.
    POES heatmap 의 lat-lon 2D 구조에 대응: 행=side, 열=월(1-month bin).
    """
    sides = ["A", "B"]
    from ksem_io import LOGICS  # noqa
    # 월별 median × side 그리드
    months, grids = None, []
    for side in sides:
        col = (pd_key, side, logic)
        if col not in df.columns:
            grids.append(None); continue
        s = df[col].dropna()
        if s.empty:
            grids.append(None); continue
        monthly = s.resample("1ME").median()
        if months is None:
            months = monthly.index
        grids.append(monthly.reindex(months).values if months is not None else monthly.values)

    if months is None or all(g is None for g in grids):
        return

    n_months = len(months)
    grid2d = np.full((2, n_months), np.nan)
    for ri, g in enumerate(grids):
        if g is not None:
            grid2d[ri, :len(g)] = g

    vmin = np.nanmin(grid2d[grid2d > 0]) if np.any(grid2d > 0) else 1e-3
    vmax = np.nanmax(grid2d) if np.any(np.isfinite(grid2d)) else 1.0

    fig, ax = plt.subplots(figsize=(max(6, n_months * 0.5 + 1), 3))
    norm = matplotlib.colors.SymLogNorm(linthresh=max(vmin, 1e-3), vmin=0, vmax=vmax)
    im = ax.pcolormesh(np.arange(n_months + 1), np.arange(3),
                       grid2d, shading="flat", norm=norm, cmap="viridis")
    ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(sides, fontsize=8)
    xlabels = [m.strftime("%Y-%m") for m in months]
    ax.set_xticks(np.arange(n_months) + 0.5)
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=6)
    ax.set_xlabel("month"); ax.set_ylabel("side")
    ax.set_title(f"{name} — monthly median by side", fontsize=9)
    fig.colorbar(im, ax=ax, label="median count rate")
    fig.tight_layout()
    fig.savefig(out / f"heatmap_{name}.png", dpi=110); plt.close(fig)


# ─────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="축3 채널 계층 의존성 + 스파이크 규명 (KSEM)")
    p.add_argument("--cache",        required=True,
                   help="KSEM 캐시 경로 (JSON 파일 또는 Parquet 디렉터리)")
    p.add_argument("--io",           default="ksem_io",
                   help="io 모듈명 또는 경로 (기본 ksem_io). 예: KSEM_count\\ksem_io")
    p.add_argument("--out",          required=True, help="출력 디렉터리")
    p.add_argument("--topk",         type=int,   default=50, help="스파이크 태깅 상위 K (기본 50)")
    p.add_argument("--spike-factor", type=float, default=10.0,
                   help="스파이크 임계 = p99 × spike_factor (기본 10)")
    p.add_argument("--no-plots",     action="store_true", help="그림 생략")
    args = p.parse_args()

    io  = _import_io(args.io, args.cache)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    df, meta = io.load(args.cache)
    if df.empty:
        print("[ERROR] 빈 캐시"); return
    print(f"[ana3] {df.shape[1]} 채널, {len(df):,} rows")

    # ── spike_origin.csv ──────────────────────────────────────────
    so = spike_origin(df, args.topk, args.spike_factor)
    so.to_csv(out / "spike_origin.csv", index=False)
    print(f"[ana3] spike_origin.csv ({len(so)} 채널)")
    for _, r in so.iterrows():
        tag = "SPIKE" if r["max_is_spike"] else "normal"
        print(f"  {r['channel']:16s} max={r['max']:>12.1f}  "
              f"p99={r['p99']:>10.3f}  grp={r['logic_group']:8s} → {tag}")

    # ── logic_profile.csv ─────────────────────────────────────────
    prof = logic_profile(df)
    prof.to_csv(out / "maglat_profile.csv")          # 파일명 POES 와 동일하게 유지

    # ── group_summary.csv ─────────────────────────────────────────
    gs = group_summary(df)
    gs.to_csv(out / "coord_summary.csv", index=False) # 파일명 POES 와 동일하게 유지
    print("[ana3] maglat_profile.csv / coord_summary.csv 저장")

    if not args.no_plots:
        for col in df.columns:
            name = col_to_label(col)
            plot_logic_profile(prof, name, out)
        plot_heatmap_pd_logic(df, out)
        print(f"[ana3] maglat_*/heatmap_* 그림 → {out}")


if __name__ == "__main__":
    main()
