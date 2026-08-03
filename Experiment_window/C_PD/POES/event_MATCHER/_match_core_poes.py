"""
_match_core_poes.py
===================
POES 매처(NOAA / SWPC) 공통 유틸리티.
직접 실행하지 않음 — noaa_goes_spe_match_poes / swpc_alert_espe_match_poes 에서 import.

공유: match_events(SAA 추적 + 이벤트 단위 클러스터링), det_matched_mask(검출별
      TP/FP 판정, match_events과 공유), sweep_table(onset CSV 지원), _final_table,
      plot_overlay, plot_pod_far_scatter(onset_diff boxplot),
      _name_stem_from_events, _import_event_io,
      _load_count_channel, _io_supports_channels, run_matcher

이벤트 단위 지표 (n_events_fa / event_FAR):
  검출 하나하나를 셀 때(FAR=n_fa/n_det)는 같은 실제 이벤트를 여러 번 재검출하면
  n_det만 부풀어 FAR이 실제보다 낮아 보이는 착시가 생긴다. match_events()가
  onset_time을 24h(cluster_gap_h, 기본 tol_h와 동일) 간격으로 묶어 "고유 검출
  사건" 단위로도 함께 집계한다 -- 클러스터 안에 카탈로그 매칭 검출이 하나라도
  있으면 TP 클러스터, 없으면 FP(오탐) 클러스터. n_hit(카탈로그측 적중 수, POD
  분자)은 클러스터링과 무관하게 불변이므로 event_FAR = n_events_fa /
  (n_hit + n_events_fa) 로 정의(고유 알람 중 가짜 비율, FDR류 지표).
"""
from __future__ import annotations
import argparse
import importlib
import re
import sys
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

MATCH_TOL_H = 24.0
ERA         = ("2019-01-01", "2025-12-31")
PFU_BINS    = [10, 100, 1000, np.inf]
PFU_LABELS  = ["10-100", "100-1k", "1k+"]

SPECIES_ORDER = ["pro", "omni", "ele"]
SPECIES_COLOR = {"pro": "#c0392b", "omni": "#e67e22", "ele": "#2980b9"}


def _import_event_io(io_arg: str, catalog_dir: str):
    """이벤트 io 임포트. 카탈로그 경로 부모들을 sys.path 에 추가."""
    name = (PureWindowsPath(io_arg).name
            if (io_arg and ('/' in io_arg or '\\' in io_arg)) else io_arg)
    cat = Path(catalog_dir).resolve()
    for cand in (cat if cat.is_dir() else cat.parent, cat.parent, cat.parent.parent,
                 Path.cwd(), Path(__file__).resolve().parent):
        sp = str(cand)
        if sp not in sys.path:
            sys.path.insert(0, sp)
    return importlib.import_module(name)


def _name_stem_from_events(events_path) -> str:
    """fsm_event_<tag>.csv / fsm_onset_<tag>.csv → <tag>."""
    stem = Path(events_path).stem
    m = re.match(r"fsm_(?:event|onset)_(.+)", stem)
    return m.group(1) if m else stem


def det_matched_mask(det: pd.DataFrame, cat: pd.DataFrame,
                     tol_h: float = MATCH_TOL_H) -> np.ndarray:
    """검출(det)별 TP/FP 판정. True=카탈로그 begin_time과 tol_h 이내 매칭(TP), False=FP.
    match_events()의 det_matched 계산을 추출한 것 -- 여기를 고치면 match_events()도
    함께 바뀐다(정의는 하나). 재사용처: diag_rolling_threshold_poes.py 판정 표시."""
    on = pd.to_datetime(det["onset_time"]).values
    cb = cat.index.values
    det_matched = np.zeros(len(det), dtype=bool)
    for i, o in enumerate(on):
        dh = (cb - o) / np.timedelta64(1, "h")
        if np.any(np.abs(dh) <= tol_h):
            det_matched[i] = True
    return det_matched


def _cluster_indices(onset_times: np.ndarray, gap_h: float) -> list[np.ndarray]:
    """onset_times를 시간순 정렬 후 gap>gap_h[h]일 때마다 새 클러스터를 여는 방식으로
    '고유 검출 사건' 단위로 묶는다(apply_refractory()류 sequential-gap 클러스터링과
    동일 개념). 각 클러스터에 속하는 원본(정렬 전) 인덱스 배열의 리스트를 반환 --
    _cluster_events()가 집계(카운트)만 필요할 때 이 함수를 감싸 쓰고,
    diag_rolling_threshold_poes.py의 --cluster(실제 그룹 멤버십 필요)도 재구현 없이
    이 함수를 그대로 재사용(det_matched_mask를 diag가 재사용하는 것과 동일 패턴)."""
    n = len(onset_times)
    if n == 0:
        return []
    order = np.argsort(onset_times)
    ot = np.asarray(onset_times)[order]
    groups: list[list[int]] = []
    cur = [int(order[0])]
    for i in range(1, n):
        if (ot[i] - ot[i - 1]) / np.timedelta64(1, "h") > gap_h:
            groups.append(cur)
            cur = []
        cur.append(int(order[i]))
    groups.append(cur)
    return [np.array(g, dtype=int) for g in groups]


def _cluster_events(onset_times: np.ndarray, matched: np.ndarray, gap_h: float) -> tuple[int, int, int]:
    """_cluster_indices()로 묶은 뒤, matched[i]==True(카탈로그 매칭)인 검출이 클러스터
    안에 하나라도 있으면 그 클러스터는 TP, 없으면 FP(오탐) 클러스터로 집계.
    반환: (n_events_total, n_events_tp, n_events_fa)."""
    groups = _cluster_indices(onset_times, gap_h)
    matched = np.asarray(matched)
    n_total = len(groups)
    n_tp = sum(1 for g in groups if np.any(matched[g]))
    n_fa = n_total - n_tp
    return n_total, n_tp, n_fa


def match_events(det: pd.DataFrame, cat: pd.DataFrame,
                 tol_h: float = MATCH_TOL_H, cluster_gap_h: float | None = None) -> dict:
    """검출 이벤트(det) ↔ 카탈로그(cat) 매칭. in_saa 컬럼이 있으면 SAA 기원 FA 집계.
    cluster_gap_h(기본 None -> tol_h와 동일): 이벤트 단위 지표(n_events_*/event_far)
    클러스터링 간격 -- 모듈 docstring 참고."""
    if cluster_gap_h is None:
        cluster_gap_h = tol_h
    on   = pd.to_datetime(det["onset_time"]).values
    pk   = pd.to_datetime(det["peak_time"]).values
    cb   = cat.index.values
    cm   = pd.to_datetime(cat["max_time"]).values
    cpfu = cat["max_pfu"].values

    cat_hit = np.zeros(len(cat), dtype=bool)
    onset_diff, peak_diff, matched_pfu = [], [], []
    for i, b in enumerate(cb):
        dh = (on - b) / np.timedelta64(1, "h")
        j  = np.where(np.abs(dh) <= tol_h)[0]
        if len(j):
            cat_hit[i] = True
            jc = j[np.argmin(np.abs(dh[j]))]
            onset_diff.append((on[jc] - b) / np.timedelta64(1, "h"))
            if pd.notna(cm[i]):
                peak_diff.append((pk[jc] - cm[i]) / np.timedelta64(1, "h"))
            matched_pfu.append(cpfu[i])

    det_matched = det_matched_mask(det, cat, tol_h)

    n_cat, n_det = len(cat), len(det)
    n_fa = int((~det_matched).sum())
    n_fa_saa = np.nan
    if "in_saa" in det.columns:
        fa = det[~det_matched]
        n_fa_saa = int((fa["in_saa"] == True).sum())

    # FA/TP geo 분포 (onset_maglat/onset_Bmag) -- POES 전용, GK2A onset CSV엔
    # 이 컬럼들이 없어 항상 빈 배열(=집계 시 NaN)로 통과한다.
    fa_maglat = tp_maglat = fa_bmag = np.array([], dtype=float)
    if "onset_maglat" in det.columns:
        fa_maglat = det.loc[~det_matched, "onset_maglat"].dropna().to_numpy(dtype=float)
        tp_maglat = det.loc[det_matched,  "onset_maglat"].dropna().to_numpy(dtype=float)
    if "onset_Bmag" in det.columns:
        fa_bmag = det.loc[~det_matched, "onset_Bmag"].dropna().to_numpy(dtype=float)

    pfu_pod = {}
    binned = pd.cut(cpfu, PFU_BINS, labels=PFU_LABELS, right=False)
    for lab in PFU_LABELS:
        mask = (binned == lab)
        pfu_pod[lab] = (int(cat_hit[mask].sum()), int(mask.sum()))

    n_hit_val = int(cat_hit.sum())
    n_events_total, n_events_tp, n_events_fa = _cluster_events(on, det_matched, cluster_gap_h)
    event_far = (n_events_fa / (n_hit_val + n_events_fa)
                 if (n_hit_val + n_events_fa) > 0 else np.nan)

    # ML 표준 명명(event 단위) -- TP/FP/FN은 각각 n_hit/n_events_fa/n_cat-n_hit의 별칭,
    # precision=1-event_far, recall=pod(둘 다 이미 위에서 정의된 값과 항등) -- 새로
    # 정의하는 값이 아니라 ML 용어로 다시 이름 붙인 것뿐. POD/FAR/event_far 등
    # 기존 키는 전부 무변경으로 병기.
    tp_val = n_hit_val
    fp_val = n_events_fa
    fn_val = n_cat - n_hit_val
    precision = (tp_val / (tp_val + fp_val)) if (tp_val + fp_val) > 0 else np.nan
    recall = (tp_val / n_cat) if n_cat else np.nan
    f1 = (2 * precision * recall / (precision + recall)
          if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
          else np.nan)

    return {
        "n_cat": n_cat, "n_det": n_det,
        "pod": cat_hit.sum() / n_cat if n_cat else np.nan,
        "far": (~det_matched).sum() / n_det if n_det else np.nan,
        "n_hit": n_hit_val, "n_fa": n_fa, "n_fa_saa": n_fa_saa,
        "fa_maglat": fa_maglat, "tp_maglat": tp_maglat, "fa_bmag": fa_bmag,
        "pfu_pod": pfu_pod,
        "onset_diff_h": np.array(onset_diff),
        "peak_diff_h":  np.array(peak_diff),
        "matched_pfu":  np.array(matched_pfu),
        "n_events_total": n_events_total, "n_events_tp": n_events_tp,
        "n_events_fa": n_events_fa, "event_far": event_far,
        "TP": tp_val, "FP": fp_val, "FN": fn_val,
        "precision": precision, "recall": recall, "f1": f1,
    }


def sweep_table(events_csv: Path, cat: pd.DataFrame,
                tol_h: float = MATCH_TOL_H) -> pd.DataFrame:
    """모든 (channel×k×onset×peak) 조합 POD/FAR 표. onset CSV(peak_floor 없음) 지원."""
    ev = pd.read_csv(events_csv)
    has_peak = "peak_floor" in ev.columns
    if not has_peak:
        ev = ev.copy()
        ev["peak_floor"] = np.nan

    rows = []
    keys = ["channel", "k", "onset_floor"] + (["peak_floor"] if has_peak else [])
    for gk, grp in ev.groupby(keys):
        if has_peak:
            ch, k, onf, pkf = gk
        else:
            ch, k, onf = gk; pkf = np.nan
        r  = match_events(grp, cat, tol_h)
        od = r["onset_diff_h"]
        n_fa = r["n_fa"]; n_fa_saa = r["n_fa_saa"]
        fa_maglat = r["fa_maglat"]; tp_maglat = r["tp_maglat"]; fa_bmag = r["fa_bmag"]
        row = {
            "channel": ch, "k": k, "onset_floor": onf, "peak_floor": pkf,
            "n_det": r["n_det"], "n_hit": r["n_hit"], "n_fa": n_fa,
            "n_fa_saa": n_fa_saa,
            "n_fa_saa_frac": round(n_fa_saa / n_fa, 3)
                if (np.isfinite(n_fa_saa) and n_fa > 0) else np.nan,
            "fa_maglat_median": round(float(np.median(fa_maglat)), 2) if fa_maglat.size else np.nan,
            "fa_maglat_p10":    round(float(np.percentile(fa_maglat, 10)), 2) if fa_maglat.size else np.nan,
            "fa_maglat_p90":    round(float(np.percentile(fa_maglat, 90)), 2) if fa_maglat.size else np.nan,
            "tp_maglat_median": round(float(np.median(tp_maglat)), 2) if tp_maglat.size else np.nan,
            "fa_bmag_median":   round(float(np.median(fa_bmag)), 2) if fa_bmag.size else np.nan,
            "POD": round(r["pod"], 3), "FAR": round(r["far"], 3),
            "n_events_total": r["n_events_total"], "n_events_tp": r["n_events_tp"],
            "n_events_fa": r["n_events_fa"],
            "event_FAR": round(r["event_far"], 3) if np.isfinite(r["event_far"]) else np.nan,
            # ML 표준 명명(event 단위) -- TP=n_hit/FP=n_events_fa/FN=n_cat-n_hit,
            # precision=1-event_FAR, recall=POD의 별칭. 기존 POD/FAR/event_FAR는 무변경 병기.
            "TP": r["TP"], "FP": r["FP"], "FN": r["FN"],
            "precision": round(r["precision"], 3) if np.isfinite(r["precision"]) else np.nan,
            "recall": round(r["recall"], 3) if np.isfinite(r["recall"]) else np.nan,
            "f1": round(r["f1"], 3) if np.isfinite(r["f1"]) else np.nan,
            "onset_diff_med_h":  round(float(np.median(od)), 2) if len(od) else np.nan,
            "onset_diff_mean_h": round(float(np.mean(od)),   2) if len(od) else np.nan,
            "onset_diff_std_h":  round(float(np.std(od)),    2) if len(od) else np.nan,
            "peak_diff_med_h":   round(float(np.median(r["peak_diff_h"])), 2)
                if len(r["peak_diff_h"]) else np.nan,
        }
        row.update({f"POD_{lab}": f"{h}/{n}" for lab, (h, n) in r["pfu_pod"].items()})
        rows.append(row)
    return pd.DataFrame(rows)


def _final_table(tbl: pd.DataFrame, no_peak: bool, args) -> pd.DataFrame:
    """단일 파라미터 조합이면 자동 선택, sweep CSV이면 --k/--onset[/--peak]로 선택."""
    param_cols = ["k", "onset_floor"] + ([] if no_peak else ["peak_floor"])
    n_combo = tbl[param_cols].drop_duplicates().shape[0]
    if n_combo <= 1:
        return tbl.sort_values("POD", ascending=False)
    if args.k is None or args.onset is None or (not no_peak and args.peak is None):
        raise SystemExit(
            f"[match] 이 CSV엔 파라미터 조합이 {n_combo}개 있습니다(sweep). "
            "--k/--onset" + ("" if no_peak else "/--peak") +
            " 로 FINAL 조합을 지정하세요.")
    sel = (tbl.k == args.k) & (tbl.onset_floor == args.onset)
    if not no_peak:
        sel = sel & (tbl.peak_floor == args.peak)
    return tbl[sel].sort_values("POD", ascending=False)


def _io_supports_channels(io) -> bool:
    import inspect
    try:
        return "channels" in inspect.signature(io.load).parameters
    except (ValueError, TypeError):
        return False


def _load_count_channel(io, cache_dir: str, channel: str) -> pd.Series:
    """POES 캐시에서 채널 1개 count Series 로드 (15min 리샘플, UTC)."""
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


def plot_overlay(cnt: pd.Series, cat: pd.DataFrame, det: pd.DataFrame,
                 title: str, out_path: Path, tol_h: float = MATCH_TOL_H):
    """POES count(좌축) + 카탈로그 pfu 동그라미(우축 log) + onset 세로선.
    SAA 발 onset은 빨강, 그 외는 주황."""
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
        dh  = (on - b.to_datetime64()) / np.timedelta64(1, "h")
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
        Line2D([0], [0], ls=":", color="orange",   lw=1.0, label="onset (non-SAA)"),
        Line2D([0], [0], ls=":", color="#c0392b",  lw=1.0, label=f"onset (SAA, {n_saa})"),
    ]
    axL.legend(handles=handles, fontsize=8, loc="upper left", framealpha=0.9)
    axL.set_title(title, fontsize=10)
    axL.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axL.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[match] overlay saved → {out_path}")


def plot_pod_far_scatter(tbl: pd.DataFrame, title: str, out_path: Path,
                         onset_diff_by_channel: dict | None = None):
    """좌: POD-FAR 산점도 (species 색상). 우: species별 onset_diff boxplot."""
    df = tbl.copy()
    if df.empty:
        return
    df["species"] = df["channel"].str.split("_").str[0]

    draw_box = bool(onset_diff_by_channel)
    if draw_box:
        fig, (ax, axb) = plt.subplots(
            1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2.2, 1.0]})
    else:
        fig, ax = plt.subplots(figsize=(7, 6))

    ax.text(0.02, 0.97, "ideal", color="green", fontsize=10,
            transform=ax.transAxes, va="top")
    for sp, g in df.groupby("species"):
        ax.scatter(g["FAR"], g["POD"], s=70, alpha=0.85,
                   c=SPECIES_COLOR.get(sp, "#888"), label=sp,
                   edgecolors="k", linewidths=0.5)
    for _, r in df.iterrows():
        ax.annotate(r["channel"], (r["FAR"], r["POD"]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")
    ax.axhline(0.5, ls=":", color="gray", lw=0.8)
    ax.axvline(0.5, ls=":", color="gray", lw=0.8)
    ax.set_xlabel("FAR (false alarm rate)")
    ax.set_ylabel("POD (probability of detection)")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.08)
    ax.set_title(title, fontsize=10); ax.grid(True, alpha=0.3); ax.legend()

    if draw_box:
        present = [sp for sp in SPECIES_ORDER if sp in set(df["species"])]
        data, labels, box_colors = [], [], []
        for sp in present:
            chans = df[df["species"] == sp]["channel"].tolist()
            vals  = np.concatenate(
                [onset_diff_by_channel.get(ch, np.array([])) for ch in chans]
            ) if chans else np.array([])
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            data.append(vals); labels.append(sp)
            box_colors.append(SPECIES_COLOR.get(sp, "#888"))
        if data:
            bp = axb.boxplot(data, labels=labels, patch_artist=True,
                             showfliers=False, widths=0.6)
            for patch, c in zip(bp["boxes"], box_colors):
                patch.set_facecolor(c); patch.set_alpha(0.55)
            for med in bp["medians"]:
                med.set_color("orange"); med.set_linewidth(1.5)
        axb.axhline(0.0, ls="--", color="gray", lw=1.0)
        axb.set_ylabel("onset_diff [h]\n(neg = POES leads)", fontsize=9)
        axb.set_xlabel("species")
        axb.set_title("onset_diff", fontsize=10)
        axb.grid(True, axis="y", alpha=0.3)

    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)
    print(f"[match] scatter → {out_path}")


def run_matcher(catalog_label: str, default_io: str, out_prefix: str,
                default_catalog: str | None = None):
    """POES 매처 공통 main 몸체. catalog_label/default_io/out_prefix 만 다르다."""
    ap = argparse.ArgumentParser(
        description=f"POES FSM ↔ {catalog_label} 매칭/평가")
    ap.add_argument("--events",  required=True, help="POES FSM event/onset CSV")
    if default_catalog:
        ap.add_argument("--catalog", default=default_catalog,
                        help=f"{catalog_label} 캐시 parquet 디렉터리")
    else:
        ap.add_argument("--catalog", required=True,
                        help=f"{catalog_label} 캐시 parquet 디렉터리")
    ap.add_argument("--spe-io",  default=default_io, help="카탈로그 io 모듈명/경로")
    ap.add_argument("--tol",     type=float, default=MATCH_TOL_H)
    ap.add_argument("--out",     default=f"{out_prefix}_match_output",
                    help="출력 폴더 (위성 count 폴더 안 권장)")
    ap.add_argument("--k",     type=float, default=10)
    ap.add_argument("--onset", type=float, default=0.5)
    ap.add_argument("--peak",  type=float, default=2.0)
    ap.add_argument("--count-cache", default=None,
                    help="overlay 용 POES count 캐시")
    ap.add_argument("--count-io", default=None,
                    help="overlay 용 io 모듈명/경로 (미지정 시 poes_metop03_io)")
    ap.add_argument("--overlay-channels", default=None,
                    help="overlay 그릴 채널 콤마구분. 'all'=전체, 미지정=best 1개")
    args = ap.parse_args()

    spe_io = _import_event_io(args.spe_io, args.catalog)
    cat_all, _ = spe_io.load(args.catalog)
    cat = spe_io.filter_by_date(cat_all, *ERA)
    print(f"[match] {catalog_label} {len(cat)}개 이벤트 ({ERA[0]}~{ERA[1]})")

    name = _name_stem_from_events(args.events)
    _ev_cols = pd.read_csv(Path(args.events), nrows=0).columns
    no_peak = "peak_floor" not in _ev_cols
    if no_peak:
        name = f"{name}_onset"
    outdir = Path(args.out) / name
    outdir.mkdir(parents=True, exist_ok=True)

    tbl = sweep_table(Path(args.events), cat, args.tol)
    f_all = outdir / f"{out_prefix}_match_summary_all_{name}.csv"
    tbl.to_csv(f_all, index=False)
    print(f"[match] summary saved → {f_all} ({len(tbl)} rows)")

    fin = _final_table(tbl, no_peak, args)
    f_fin = outdir / f"{out_prefix}_match_{name}.csv"
    fin.to_csv(f_fin, index=False)
    print(f"[match] FINAL saved → {f_fin}")

    show = ["channel", "n_det", "n_hit", "n_fa", "n_fa_saa", "n_fa_saa_frac",
            "POD", "FAR", "n_events_fa", "event_FAR", "onset_diff_med_h"]
    show = [c for c in show if c in fin.columns]
    print(fin[show].to_string(index=False))

    if not fin.empty:
        ranked = fin.sort_values(["FAR", "POD"], ascending=[True, False])
        best   = ranked.iloc[0]
        print(f"\n[match] ▶ best trigger channel: {best['channel']}  "
              f"POD={best['POD']} FAR={best['FAR']} "
              f"n_fa_saa={best.get('n_fa_saa','?')}/{best.get('n_fa','?')}")

    # boxplot용 onset_diff 분포 수집 (FINAL 조합)
    onset_diff_by_channel: dict = {}
    ev_fin = pd.read_csv(Path(args.events))
    _sf = (ev_fin.k == args.k) & (ev_fin.onset_floor == args.onset)
    if not no_peak:
        _sf = _sf & (ev_fin.peak_floor == args.peak)
    for ch, grp in ev_fin[_sf].groupby("channel"):
        r = match_events(grp, cat, args.tol)
        onset_diff_by_channel[ch] = r["onset_diff_h"]

    plot_pod_far_scatter(fin, f"{catalog_label} match  {name}",
                         outdir / f"fig_{out_prefix}_scatter_{name}.png",
                         onset_diff_by_channel=onset_diff_by_channel)

    # ── overlay ──
    if args.count_cache:
        cnt_io = _import_event_io(args.count_io or "poes_metop03_io", args.count_cache)
        ev     = pd.read_csv(args.events)
        _sm    = (ev.k == args.k) & (ev.onset_floor == args.onset)
        if not no_peak:
            _sm = _sm & (ev.peak_floor == args.peak)
        sel = ev[_sm]
        if args.overlay_channels == "all":
            channels = sorted(sel["channel"].unique())
        elif args.overlay_channels:
            channels = [c.strip() for c in args.overlay_channels.split(",")]
        else:
            channels = ([fin.sort_values(["FAR", "POD"], ascending=[True, False])
                         .iloc[0]["channel"]] if not fin.empty else [])
        for ch in channels:
            det = sel[sel["channel"] == ch]
            if det.empty:
                print(f"[match] overlay skip {ch}: 검출 없음"); continue
            cnt = _load_count_channel(cnt_io, args.count_cache, ch)
            if cnt.empty:
                print(f"[match] overlay skip {ch}: count 없음"); continue
            plot_overlay(cnt, cat, det, f"{ch}  {name}",
                         outdir / f"overlay_{ch}_{name}.png", args.tol)
