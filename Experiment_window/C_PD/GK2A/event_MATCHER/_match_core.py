"""
_match_core.py
==============
GK2A 매처(NOAA / SWPC) 공통 유틸리티.
직접 실행하지 않음 — noaa_goes_spe_match / swpc_alert_espe_match 에서 import.

공유 함수: match_events, sweep_table, _final_table,
           _name_stem_from_events, _discover, _channel_to_parquet, _load_count
"""
from __future__ import annotations
import re
from pathlib import Path

import numpy as np
import pandas as pd

MATCH_TOL_H = 24.0
KSEM_ERA    = ("2019-01-01", "2024-12-31")
PFU_BINS    = [10, 100, 1000, np.inf]
PFU_LABELS  = ["10-100", "100-1k", "1k+"]


def _name_stem_from_events(events_path) -> str:
    """fsm_event_<tag>.csv / fsm_onset_<tag>.csv → <tag>."""
    stem = Path(events_path).stem
    m = re.match(r"fsm_(?:event|onset)_(.+)", stem)
    return m.group(1) if m else stem


def match_events(det: pd.DataFrame, cat: pd.DataFrame,
                 tol_h: float = MATCH_TOL_H) -> dict:
    """
    검출 이벤트(det) ↔ 카탈로그(cat: begin_time index, max_time/max_pfu) 매칭.
    반환: POD/FAR, pfu 구간별 POD, 매칭 쌍의 시간차 배열.
    """
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
        mask = (binned == lab)
        pfu_pod[lab] = (int(cat_hit[mask].sum()), int(mask.sum()))

    return {
        "n_cat": n_cat, "n_det": n_det,
        "pod": pod, "far": far,
        "n_hit": int(cat_hit.sum()), "n_fa": int((~det_matched).sum()),
        "pfu_pod": pfu_pod,
        "onset_diff_h": np.array(onset_diff),
        "peak_diff_h":  np.array(peak_diff),
        "matched_pfu":  np.array(matched_pfu),
    }


def sweep_table(events_csv: Path, cat: pd.DataFrame,
                tol_h: float = MATCH_TOL_H) -> pd.DataFrame:
    """
    events CSV 의 모든 (channel × k × onset × peak) 조합에 대해 POD/FAR 표 산출.
    onset CSV (peak_floor 컬럼 없음) 도 처리: peak_floor=NaN 으로 채워 동작.
    """
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
        r = match_events(grp, cat, tol_h)
        rows.append({
            "channel": ch, "k": k, "onset_floor": onf, "peak_floor": pkf,
            "n_det": r["n_det"], "n_hit": r["n_hit"], "n_fa": r["n_fa"],
            "POD": round(r["pod"], 3), "FAR": round(r["far"], 3),
            "onset_diff_med_h": round(float(np.median(r["onset_diff_h"])), 2)
                if len(r["onset_diff_h"]) else np.nan,
            "peak_diff_med_h":  round(float(np.median(r["peak_diff_h"])), 2)
                if len(r["peak_diff_h"]) else np.nan,
            **{f"POD_{lab}": f"{h}/{n}" for lab, (h, n) in r["pfu_pod"].items()},
        })
    return pd.DataFrame(rows)


def _final_table(tbl: pd.DataFrame, no_peak: bool, args) -> pd.DataFrame:
    """FINAL 채널별 표 선택.
    단일 파라미터 조합이면 그대로 POD 내림차순 반환(--k/--onset 불필요).
    여러 조합 든 sweep CSV 일 때만 selector 로 걸러낸다."""
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


def _discover(events_dir: Path, kind: str) -> list:
    """fsm*_output 폴더 트리에서 fsm_{kind}_*.csv 전부 찾기."""
    return sorted(Path(events_dir).rglob(f"fsm_{kind}_*.csv"))


def _channel_to_parquet(channel: str) -> str:
    """채널명 'PD3A-OU' → parquet 파일명 'PD3_A_OU.parquet'."""
    pd_side, logic = channel.split("-")
    pd_key, side = pd_side[:-1], pd_side[-1]
    return f"{pd_key}_{side}_{logic}.parquet"


def _load_count(parquet_path: Path) -> pd.Series:
    cnt = pd.read_parquet(parquet_path)
    if isinstance(cnt, pd.DataFrame):
        cnt = cnt.iloc[:, 0]
    cnt.index = pd.to_datetime(cnt.index, utc=True)
    return cnt.resample("15min").mean().dropna()
