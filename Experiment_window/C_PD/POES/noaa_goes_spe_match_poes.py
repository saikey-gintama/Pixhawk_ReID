"""
noaa_goes_spe_match_poes.py
===========================
POES FSM 검출 이벤트 ↔ NOAA SPE 카탈로그 매칭/평가 (POES 판).

KSEM noaa_goes_spe_match.py 의 POES 이식판:
  - 매칭 로직(match_events) / sweep_table 은 KSEM 과 동일 (channel·onset_time·
    k·onset_floor·peak_floor 컬럼만 읽으므로 POES FSM 출력 CSV 그대로 동작).
  - POES 전용 변경:
      · 이벤트 io 인자화(--spe-io, 기본 noaa_goes_spe_io) — ../ 등에서 로드
      · POD/FAR scatter 그룹화: KSEM logic → POES species(pro/omni/ele)
      · 저장: 위성 count 폴더 안 (--out)
      · in_saa 컬럼 활용 → SAA발 false alarm 집계 (n_fa_saa, n_fa_saa_frac)
      · onset 지연 분포(mean/std/median), best-channel 요약 콘솔 강조

출력 (outdir = <--out>/<runtag>/):
  noaa_match_summary_all_<runtag>.csv   전 (channel×k×onset×peak) 조합 POD/FAR
  noaa_match_<runtag>.csv               FINAL 조합(--k/--onset/--peak) 채널별, POD 내림차순
  fig_noaa_scatter_<runtag>.png         POD-FAR 산점도 (species 색상)

사용 (POES 루트에서):
  python noaa_goes_spe_match_poes.py \
    --events MetOp03_count/fsm_output/<runtag>/fsm_event_<runtag>.csv \
    --catalog ../NOAA_GOES/noaa_goes_spe_cache_parquet \
    --spe-io ../NOAA_GOES/noaa_goes_spe_io \
    --out MetOp03_count/noaa_match_output \
    --k 10 --onset 0.5 --peak 2
"""
from __future__ import annotations
import argparse, sys, importlib, re
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

MATCH_TOL_H = 24.0
ERA         = ("2019-01-01", "2025-12-31")   # POES 관측 기간
PFU_BINS    = [10, 100, 1000, np.inf]
PFU_LABELS  = ["10-100", "100-1k", "1k+"]

CATALOG_LABEL = "NOAA SPE"
DEFAULT_IO    = "noaa_goes_spe_io"
OUT_PREFIX    = "noaa"


def _import_event_io(io_arg: str, catalog_dir: str):
    """이벤트 io 임포트. 카탈로그 경로 부모들을 sys.path 에 추가."""
    name = PureWindowsPath(io_arg).name if (io_arg and ('/' in io_arg or '\\' in io_arg)) else io_arg
    cat = Path(catalog_dir).resolve()
    for cand in (cat if cat.is_dir() else cat.parent, cat.parent, cat.parent.parent,
                 Path.cwd(), Path(__file__).resolve().parent):
        sp = str(cand)
        if sp not in sys.path:
            sys.path.insert(0, sp)
    return importlib.import_module(name)


def _name_stem_from_events(events_path) -> str:
    """fsm_event_<tag>.csv → <tag>."""
    stem = Path(events_path).stem
    m = re.match(r"fsm_event_(.+)", stem)
    return m.group(1) if m else stem


# ── 매칭 (검출 엔진과 마찬가지로 KSEM 과 동일 의미) ─────────────────
def match_events(det: pd.DataFrame, cat: pd.DataFrame, tol_h: float = MATCH_TOL_H) -> dict:
    """검출 이벤트(det) ↔ 카탈로그(cat: begin_time index, max_time/max_pfu) 매칭."""
    on = pd.to_datetime(det["onset_time"]).values
    pk = pd.to_datetime(det["peak_time"]).values
    cb = cat.index.values
    cm = pd.to_datetime(cat["max_time"]).values
    cpfu = cat["max_pfu"].values

    cat_hit = np.zeros(len(cat), dtype=bool)
    onset_diff, peak_diff, matched_pfu = [], [], []
    for i, b in enumerate(cb):
        dh = (on - b) / np.timedelta64(1, "h")
        j = np.where(np.abs(dh) <= tol_h)[0]
        if len(j):
            cat_hit[i] = True
            jc = j[np.argmin(np.abs(dh[j]))]
            onset_diff.append((on[jc] - b) / np.timedelta64(1, "h"))
            if pd.notna(cm[i]):
                peak_diff.append((pk[jc] - cm[i]) / np.timedelta64(1, "h"))
            matched_pfu.append(cpfu[i])

    det_matched = np.zeros(len(det), dtype=bool)
    for i, o in enumerate(on):
        dh = (cb - o) / np.timedelta64(1, "h")
        if np.any(np.abs(dh) <= tol_h):
            det_matched[i] = True

    n_cat, n_det = len(cat), len(det)
    pod = cat_hit.sum() / n_cat if n_cat else np.nan
    far = (~det_matched).sum() / n_det if n_det else np.nan

    pfu_pod = {}
    binned = pd.cut(cpfu, PFU_BINS, labels=PFU_LABELS, right=False)
    for lab in PFU_LABELS:
        m = (binned == lab)
        pfu_pod[lab] = (int(cat_hit[m].sum()), int(m.sum()))

    # POES 부가: SAA발 false alarm
    n_fa = int((~det_matched).sum())
    n_fa_saa = np.nan
    if "in_saa" in det.columns:
        fa = det[~det_matched]
        n_fa_saa = int((fa["in_saa"] == True).sum())

    return {"n_cat": n_cat, "n_det": n_det, "pod": pod, "far": far,
            "n_hit": int(cat_hit.sum()), "n_fa": n_fa, "n_fa_saa": n_fa_saa,
            "pfu_pod": pfu_pod,
            "onset_diff_h": np.array(onset_diff),
            "peak_diff_h": np.array(peak_diff),
            "matched_pfu": np.array(matched_pfu)}


def sweep_table(events_csv: Path, cat: pd.DataFrame, tol_h: float = MATCH_TOL_H) -> pd.DataFrame:
    """모든 (channel×k×onset×peak) 조합 POD/FAR 표."""
    ev = pd.read_csv(events_csv)
    rows = []
    keys = ["channel", "k", "onset_floor", "peak_floor"]
    for (ch, k, onf, pkf), grp in ev.groupby(keys):
        r = match_events(grp, cat, tol_h)
        od = r["onset_diff_h"]
        n_fa = r["n_fa"]; n_fa_saa = r["n_fa_saa"]
        row = {"channel": ch, "k": k, "onset_floor": onf, "peak_floor": pkf,
               "n_det": r["n_det"], "n_hit": r["n_hit"], "n_fa": n_fa,
               "n_fa_saa": n_fa_saa,
               "n_fa_saa_frac": round(n_fa_saa / n_fa, 3)
                   if (np.isfinite(n_fa_saa) and n_fa > 0) else np.nan,
               "POD": round(r["pod"], 3), "FAR": round(r["far"], 3),
               "onset_diff_med_h": round(float(np.median(od)), 2) if len(od) else np.nan,
               "onset_diff_mean_h": round(float(np.mean(od)), 2) if len(od) else np.nan,
               "onset_diff_std_h": round(float(np.std(od)), 2) if len(od) else np.nan,
               "peak_diff_med_h": round(float(np.median(r["peak_diff_h"])), 2)
                   if len(r["peak_diff_h"]) else np.nan}
        row.update({f"POD_{lab}": f"{h}/{n}" for lab, (h, n) in r["pfu_pod"].items()})
        rows.append(row)
    return pd.DataFrame(rows)


def _load_count_channel(io, cache_dir: str, channel: str) -> pd.Series:
    """POES 캐시에서 채널 1개의 count Series 로드 (15min 리샘플, UTC)."""
    # 채널명(str)을 튜플로 변환해 부분 로드 시도, 안 되면 전체 로드 후 선택
    df = None
    if hasattr(io, "fname_to_tuple") and _io_supports_channels(io):
        try:
            tpl = io.fname_to_tuple(channel)
            df, _ = io.load(cache_dir, channels=[tpl])
        except Exception:
            df = None
    if df is None:
        df, _ = io.load(cache_dir)
    col = None
    for c in df.columns:
        if io.tuple_to_fname(tuple(c)) == channel:
            col = c; break
    if col is None:
        return pd.Series(dtype=float)
    cnt = df[col].dropna()
    if cnt.index.tz is None:
        cnt.index = cnt.index.tz_localize("UTC")
    return cnt.resample("15min").mean().dropna()


def _io_supports_channels(io) -> bool:
    """io.load 가 channels 인자를 받는지 (부분 로드 지원 여부)."""
    import inspect
    try:
        return "channels" in inspect.signature(io.load).parameters
    except (ValueError, TypeError):
        return False


def plot_overlay(cnt: pd.Series, cat: pd.DataFrame, det: pd.DataFrame,
                 title: str, out_path: Path, tol_h: float = MATCH_TOL_H):
    """POES count(좌축 선) + 카탈로그 pfu 동그라미(우축 log).
    매칭된 카탈로그=채운 원, 놓친 것=빈 원. 검출 onset=세로 점선
    (SAA발 onset 은 빨강, 그 외는 주황 — POES 전용 구분)."""
    on = pd.to_datetime(det["onset_time"]).values
    in_saa = det["in_saa"].values if "in_saa" in det.columns else np.full(len(on), False)

    fig, axL = plt.subplots(figsize=(16, 4.5))
    axL.plot(cnt.index, cnt.values, color="#2c3e50", lw=0.6, zorder=1)
    axL.set_ylabel("POES count [15-min]", fontsize=9)
    axL.set_ylim(bottom=0)

    axR = axL.twinx(); axR.set_yscale("log")
    axR.set_ylabel("catalog max_pfu", fontsize=9)

    n_hit = n_miss = 0
    for b, mp in zip(cat.index, cat["max_pfu"].values):
        if pd.isna(mp):
            continue
        dh = (on - b.to_datetime64()) / np.timedelta64(1, "h")
        hit = np.any(np.abs(dh) <= tol_h)
        axR.scatter(b, mp, s=80, facecolor=("#e74c3c" if hit else "none"),
                    edgecolor="#e74c3c", linewidth=1.5, zorder=3)
        n_hit += hit; n_miss += (not hit)

    n_saa = 0
    for o, s in zip(on, in_saa):
        c = "#c0392b" if s is True else "orange"
        axL.axvline(pd.Timestamp(o), ls=":", color=c, lw=0.8,
                    alpha=0.65 if s is True else 0.5, zorder=2)
        n_saa += (s is True)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="#2c3e50", lw=1.0, label="POES count"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#e74c3c",
               markeredgecolor="#e74c3c", markersize=9, label=f"catalog detected ({n_hit})"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor="#e74c3c", markeredgewidth=1.5, markersize=9,
               label=f"catalog missed ({n_miss})"),
        Line2D([0], [0], ls=":", color="orange", lw=1.0, label="onset (non-SAA)"),
        Line2D([0], [0], ls=":", color="#c0392b", lw=1.0, label=f"onset (SAA, {n_saa})"),
    ]
    axL.legend(handles=handles, fontsize=8, loc="upper left", framealpha=0.9)
    axL.set_title(title, fontsize=10)
    axL.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axL.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[match] overlay saved → {out_path}")


def plot_pod_far_scatter(tbl: pd.DataFrame, title: str, out_path: Path):
    """POD-FAR 산점도, species 색상 (pro/omni/ele)."""
    df = tbl.copy()
    if df.empty:
        return
    df["species"] = df["channel"].str.split("_").str[0]
    color = {"pro": "#c0392b", "omni": "#e67e22", "ele": "#2980b9"}
    fig, ax = plt.subplots(figsize=(7, 6))
    for sp, g in df.groupby("species"):
        ax.scatter(g["FAR"], g["POD"], s=70, alpha=0.85,
                   c=color.get(sp, "#888"), label=sp, edgecolors="k", linewidths=0.5)
    for _, r in df.iterrows():
        ax.annotate(r["channel"], (r["FAR"], r["POD"]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("FAR (false alarm rate)")
    ax.set_ylabel("POD (probability of detection)")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_title(title, fontsize=10); ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=f"POES FSM ↔ {CATALOG_LABEL} 매칭/평가")
    ap.add_argument("--events", required=True, help="POES FSM event CSV")
    ap.add_argument("--catalog", required=True, help=f"{CATALOG_LABEL} 캐시 parquet 디렉터리")
    ap.add_argument("--spe-io", default=DEFAULT_IO, help="이벤트 io 모듈명/경로")
    ap.add_argument("--tol", type=float, default=MATCH_TOL_H)
    ap.add_argument("--out", default=f"{OUT_PREFIX}_match_output",
                    help="출력 폴더 (위성 count 폴더 안 권장)")
    ap.add_argument("--k", type=float, default=10)
    ap.add_argument("--onset", type=float, default=0.5)
    ap.add_argument("--peak", type=float, default=2.0)
    ap.add_argument("--count-cache", default=None,
                    help="overlay 용 POES count 캐시 (지정 시 채널별 시계열 겹쳐그리기)")
    ap.add_argument("--count-io", default=None,
                    help="overlay 용 io 모듈명/경로 (poes_metop03_io 등; 미지정 시 --count-cache 부모에서 추론)")
    ap.add_argument("--overlay-channels", default=None,
                    help="overlay 그릴 채널 콤마구분 (예: pro_tel0_p5,omni_p6). 'all'=전체. 미지정 시 best 1개")
    args = ap.parse_args()

    spe_io = _import_event_io(args.spe_io, args.catalog)
    cat_all, _ = spe_io.load(args.catalog)
    cat = spe_io.filter_by_date(cat_all, *ERA)
    print(f"[match] {CATALOG_LABEL} {len(cat)}개 이벤트 ({ERA[0]}~{ERA[1]})")

    name = _name_stem_from_events(args.events)
    outdir = Path(args.out) / name
    outdir.mkdir(parents=True, exist_ok=True)

    tbl = sweep_table(Path(args.events), cat, args.tol)
    f_all = outdir / f"{OUT_PREFIX}_match_summary_all_{name}.csv"
    tbl.to_csv(f_all, index=False)
    print(f"[match] summary saved → {f_all} ({len(tbl)} rows)")

    fin = tbl[(tbl.k == args.k) & (tbl.onset_floor == args.onset)
              & (tbl.peak_floor == args.peak)].sort_values("POD", ascending=False)
    f_fin = outdir / f"{OUT_PREFIX}_match_{name}.csv"
    fin.to_csv(f_fin, index=False)
    print(f"[match] FINAL saved → {f_fin}")

    show = ["channel", "n_det", "n_hit", "n_fa", "n_fa_saa", "n_fa_saa_frac",
            "POD", "FAR", "onset_diff_med_h"]
    show = [c for c in show if c in fin.columns]
    print(fin[show].to_string(index=False))

    # best channel (낮은 FAR 우선, 동률 시 높은 POD) 콘솔 강조
    if not fin.empty:
        ranked = fin.sort_values(["FAR", "POD"], ascending=[True, False])
        best = ranked.iloc[0]
        print(f"\n[match] ▶ best trigger channel: {best['channel']}  "
              f"POD={best['POD']} FAR={best['FAR']} "
              f"n_fa_saa={best.get('n_fa_saa','?')}/{best.get('n_fa','?')}")

    plot_pod_far_scatter(fin, f"{CATALOG_LABEL} match  {name}",
                         outdir / f"fig_{OUT_PREFIX}_scatter_{name}.png")
    print(f"[match] scatter → {outdir}/fig_{OUT_PREFIX}_scatter_{name}.png")

    # ── overlay (채널별 count 시계열 + 카탈로그 hit/miss + onset) ──
    if args.count_cache:
        cnt_io = _import_event_io(args.count_io or "poes_metop03_io", args.count_cache)
        ev = pd.read_csv(args.events)
        sel = ev[(ev.k == args.k) & (ev.onset_floor == args.onset)
                 & (ev.peak_floor == args.peak)]
        if args.overlay_channels == "all":
            channels = sorted(sel["channel"].unique())
        elif args.overlay_channels:
            channels = [c.strip() for c in args.overlay_channels.split(",")]
        else:
            channels = [fin.sort_values(["FAR", "POD"], ascending=[True, False])
                        .iloc[0]["channel"]] if not fin.empty else []
        for ch in channels:
            det = sel[sel["channel"] == ch]
            if det.empty:
                print(f"[match] overlay skip {ch}: 검출 없음"); continue
            cnt = _load_count_channel(cnt_io, args.count_cache, ch)
            if cnt.empty:
                print(f"[match] overlay skip {ch}: count 없음"); continue
            plot_overlay(cnt, cat, det, f"{ch}  {name}",
                         outdir / f"overlay_{ch}_{name}.png", args.tol)


if __name__ == "__main__":
    main()