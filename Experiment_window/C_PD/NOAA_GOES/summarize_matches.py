"""
consolidate_matches.py
======================
여러 배경추정 방법(quietoff_mad / blc1_lowe / blc1_fixed_const / ...)의
NOAA 매칭 결과 CSV를 한 표로 통합·정렬해 비교하는 도구.

매번 noaa_match_*.csv 를 일일이 열지 않아도 되도록:
  - 방법(method)·채널별 행을 한 프레임으로 합치고
  - J' = POD - FAR (Youden's J 계열) 를 계산해 정렬
  - logic(OU/OUT/...)·group(A/B/C)별로도 묶어 비교
  - 결과를 CSV 여러 장 + xlsx 멀티시트로 저장하고 콘솔에 랭킹 출력

── summary vs FINAL CSV ────────────────────────────────────────────
noaa_match_<name>.csv (FINAL) 와 noaa_match_summary_all_<name>.csv 는
단일조합 파일에선 **내용 동일, 정렬만 다름**(FINAL=POD 내림차순, summary=채널순).
따라서 이 도구는 방법당 하나만(FINAL 우선) 읽고 summary_all 은 건너뛴다.
(--use-summary 로 강제 포함 가능.)

── J' 정의 (바꾸려면 _jprime 한 곳만 수정) ─────────────────────────
J' = POD - FAR.
  POD = n_hit / n_noaa            (재현율, TPR)
  FAR = n_fa  / n_det            (검출 중 오경보 비율; 고전 FPR=FP/(FP+TN)와
                                  달리 TN 정의가 없어 FAR를 비용항으로 씀)
KSEM 논문의 J' 정의가 다르면 _jprime() 만 교체하면 전 표에 일괄 반영된다.

사용:
  # 폴더 안 noaa_match_*.csv 전부 통합 (summary_all 자동 제외)
  python consolidate_matches.py --dir noaa_match_onset_output --out match_consolidated

  # 특정 파일들만
  python consolidate_matches.py --inputs a.csv b.csv c.csv --out cmp

  # 상위 N개 채널/방법
  python consolidate_matches.py --dir noaa_match_onset_output --topk 5
"""

from __future__ import annotations
import re
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# 채널 logic → group (다른 코드의 _group과 동일 규칙)
GROUP_MAP_A = {"OU", "OUT"}
GROUP_MAP_B = {"FTU", "FTUO"}
# 통계 집계에서 제외할 문자열(=POD 구간) 컬럼
BIN_COLS = ["POD_10-100", "POD_100-1k", "POD_1k+"]


# ══════════════════════════════════════════════════════════════════
# J' (이 한 곳만 바꾸면 전 표 일괄 반영)
# ══════════════════════════════════════════════════════════════════
def _jprime(df: pd.DataFrame) -> pd.Series:
    """J' = POD - FAR. (Youden's J 계열, FAR를 FPR 대용 비용항으로)"""
    return df["POD"] - df["FAR"]


def _group(logic: str) -> str:
    if logic in GROUP_MAP_A:
        return "A"          # OU/OUT : quiet bg, 탐지 적합
    if logic in GROUP_MAP_B:
        return "B"          # FTU/FTUO : 중간
    return "C"              # O/F/FT/CR : active bg, 부적합


def parse_method(path: Path) -> str:
    """파일명에서 방법 식별자 추출.
    noaa_match_blc1_lowe_w5_m1.25_on0.5_pk2_onset.csv -> blc1_lowe_w5_m1.25_on0.5_pk2
    summary_all 접두/_onset·_nopk·_event 접미 제거."""
    stem = path.stem
    stem = re.sub(r"^noaa_match_", "", stem)
    stem = re.sub(r"^summary_all_", "", stem)
    stem = re.sub(r"_(onset|nopk|event)$", "", stem)
    return stem


def method_family(method: str) -> str:
    """파라미터(window/onset/...)를 떼고 방법 계열만.
    quietoff_mad_w30_k10_on0.5_pk2 -> quietoff_mad
    blc1_fixed_const_on0.5_pk2     -> blc1_fixed_const"""
    return re.split(r"_w\d|_on", method)[0]


def split_channel(channel: str):
    """'PD3A-OU' -> (pd_key='PD3', side='A', logic='OU')."""
    try:
        pd_side, logic = channel.split("-", 1)
        return pd_side[:-1], pd_side[-1], logic
    except ValueError:
        return np.nan, np.nan, np.nan


# ══════════════════════════════════════════════════════════════════
def discover(dir_path: Path, use_summary: bool) -> list[Path]:
    """폴더 트리에서 noaa_match_*.csv 수집. summary_all 은 기본 제외(중복)."""
    files = sorted(dir_path.rglob("noaa_match_*.csv"))
    if not use_summary:
        files = [f for f in files if "summary_all" not in f.name]
    return files


def load_all(files: list[Path], use_summary: bool) -> pd.DataFrame:
    """파일들을 읽어 method/channel 메타 컬럼 + J' 붙여 한 프레임으로."""
    seen, frames = {}, []
    for f in files:
        method = parse_method(f)
        # FINAL 우선: 같은 method가 이미 (summary로) 들어왔으면 FINAL로 교체
        is_summary = "summary_all" in f.name
        if method in seen:
            if seen[method] == "summary" and not is_summary:
                frames = [fr for fr in frames if fr["method"].iloc[0] != method]
            else:
                continue
        seen[method] = "summary" if is_summary else "final"

        df = pd.read_csv(f)
        if df.empty or "channel" not in df.columns:
            print(f"[skip] {f.name}: 빈 결과(검출 0건) — 통합에서 제외")
            continue
        df["method"] = method
        df["method_family"] = method_family(method)
        pds = df["channel"].apply(split_channel)
        df["pd_key"] = [p[0] for p in pds]
        df["side"]   = [p[1] for p in pds]
        df["logic"]  = [p[2] for p in pds]
        df["group"]  = df["logic"].map(_group)
        df["Jprime"] = _jprime(df)
        df["source_file"] = f.name
        frames.append(df)

    if not frames:
        raise SystemExit("[consolidate] 읽을 매칭 CSV가 없습니다.")
    return pd.concat(frames, ignore_index=True)


# ══════════════════════════════════════════════════════════════════
# 뷰 생성
# ══════════════════════════════════════════════════════════════════
LEAD_COLS = ["method_family", "method", "channel", "pd_key", "side",
             "logic", "group", "POD", "FAR", "Jprime",
             "n_det", "n_hit", "n_fa", "onset_diff_med_h", "peak_diff_med_h",
             *BIN_COLS]


def view_all(comb: pd.DataFrame) -> pd.DataFrame:
    """전체 (method × channel), J' 내림차순(방법계열 내)."""
    cols = [c for c in LEAD_COLS if c in comb.columns]
    out = comb[cols].sort_values(
        ["method_family", "method", "Jprime"],
        ascending=[True, True, False]).reset_index(drop=True)
    return out


def view_best_per_method(comb: pd.DataFrame, topk: int) -> pd.DataFrame:
    """방법별 J' 상위 topk 채널."""
    cols = ["method", "channel", "logic", "group", "POD", "FAR", "Jprime",
            "n_det", "n_hit", "n_fa"]
    cols = [c for c in cols if c in comb.columns]
    return (comb.sort_values("Jprime", ascending=False)
                .groupby("method", sort=False)
                .head(topk)[cols]
                .sort_values(["method", "Jprime"], ascending=[True, False])
                .reset_index(drop=True))


def _agg_by(comb: pd.DataFrame, key: str) -> pd.DataFrame:
    """(method × key) 집계. key='logic' 또는 'group'.
    채널 평균(POD/FAR/J')과 최고 J' 채널을 함께 보여준다.
    주의: 여러 채널의 OR 풀링(voting) POD가 아니라 '채널 평균'이다 — 진짜
    OR-pooled 검출력은 이벤트 단위 데이터가 필요(여기 summary로는 불가)."""
    rows = []
    for (method, kv), g in comb.groupby(["method", key]):
        gg = g.dropna(subset=["Jprime"])
        best = gg.loc[gg["Jprime"].idxmax()] if len(gg) else g.iloc[0]
        rows.append({
            "method": method, key: kv, "n_ch": len(g),
            "POD_mean":    round(g["POD"].mean(), 3),
            "FAR_mean":    round(g["FAR"].mean(), 3),
            "Jprime_mean": round(g["Jprime"].mean(), 3),
            "Jprime_best": round(float(best["Jprime"]), 3) if len(gg) else np.nan,
            "best_channel": best["channel"] if len(gg) else "",
            "POD_best":    round(float(best["POD"]), 3) if len(gg) else np.nan,
            "FAR_best":    round(float(best["FAR"]), 3) if len(gg) else np.nan,
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["method", "Jprime_mean"],
                           ascending=[True, False]).reset_index(drop=True)


def view_by_logic(comb): return _agg_by(comb, "logic")
def view_by_group(comb): return _agg_by(comb, "group")


def view_method_rank(comb: pd.DataFrame) -> pd.DataFrame:
    """방법 전체 랭킹. 탐지적합군(group A=OU/OUT) 기준 평균 J' 와
    전 채널 최고 J' 둘 다 제시. A 평균 내림차순 정렬."""
    rows = []
    for method, g in comb.groupby("method"):
        a = g[g["group"] == "A"].dropna(subset=["Jprime"])
        gg = g.dropna(subset=["Jprime"])
        best = gg.loc[gg["Jprime"].idxmax()] if len(gg) else None
        rows.append({
            "method": method,
            "family": g["method_family"].iloc[0],
            "JprimeA_mean": round(a["Jprime"].mean(), 3) if len(a) else np.nan,
            "PODA_mean":    round(a["POD"].mean(), 3) if len(a) else np.nan,
            "FARA_mean":    round(a["FAR"].mean(), 3) if len(a) else np.nan,
            "n_chA": len(a),
            "Jprime_best_any": round(float(best["Jprime"]), 3) if best is not None else np.nan,
            "best_channel_any": best["channel"] if best is not None else "",
        })
    return pd.DataFrame(rows).sort_values(
        "JprimeA_mean", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════
def write_xlsx(views: dict, path: Path) -> bool:
    """뷰들을 xlsx 멀티시트로. openpyxl 없으면 False 반환(CSV만)."""
    try:
        from openpyxl.styles import Font, Alignment, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError:
        return False
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for sheet, df in views.items():
            df.to_excel(xw, sheet_name=sheet[:31], index=False)
        wb = xw.book
        head_fill = PatternFill("solid", start_color="1F4E78")
        head_font = Font(bold=True, color="FFFFFF", name="Arial")
        for sheet, df in views.items():
            ws = wb[sheet[:31]]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for ci, col in enumerate(df.columns, 1):
                cell = ws.cell(row=1, column=ci)
                cell.fill = head_fill; cell.font = head_font
                cell.alignment = Alignment(horizontal="center")
                width = max(len(str(col)),
                            *(len(str(v)) for v in df[col].astype(str).head(50)),
                            6)
                ws.column_dimensions[get_column_letter(ci)].width = min(width + 2, 40)
    return True


def main():
    ap = argparse.ArgumentParser(
        description="NOAA 매칭 결과 통합·비교 (J' 정렬, logic/group 집계)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dir", help="noaa_match_*.csv 들이 든 폴더(트리 재귀)")
    src.add_argument("--inputs", nargs="+", help="CSV 파일들 직접 지정")
    ap.add_argument("--out", default="match_consolidated", help="출력 폴더")
    ap.add_argument("--topk", type=int, default=5, help="방법별 상위 채널 수")
    ap.add_argument("--use-summary", action="store_true",
                    help="summary_all CSV도 포함(기본은 FINAL만)")
    args = ap.parse_args()

    if args.dir:
        files = discover(Path(args.dir), args.use_summary)
    else:
        files = [Path(p) for p in args.inputs]
    if not files:
        raise SystemExit("[consolidate] 입력 CSV를 찾지 못했습니다.")
    print(f"[consolidate] {len(files)}개 파일 통합:")
    for f in files:
        print(f"   {f}")

    comb = load_all(files, args.use_summary)
    n_methods = comb["method"].nunique()
    print(f"[consolidate] 방법 {n_methods}개 × 채널 → 총 {len(comb)}행\n")

    views = {
        "all_channels":    view_all(comb),
        "best_per_method": view_best_per_method(comb, args.topk),
        "by_logic":        view_by_logic(comb),
        "by_group":        view_by_group(comb),
        "method_rank":     view_method_rank(comb),
    }

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    for name, df in views.items():
        df.to_csv(outdir / f"{name}.csv", index=False)
    xlsx_ok = write_xlsx(views, outdir / "match_comparison.xlsx")

    print("[consolidate] 저장:")
    for name in views:
        print(f"   {outdir / (name + '.csv')}")
    if xlsx_ok:
        print(f"   {outdir / 'match_comparison.xlsx'}  (멀티시트)")
    else:
        print("   (openpyxl 없음 → xlsx 생략, CSV만)")

    # 콘솔 요약
    pd.set_option("display.width", 160, "display.max_columns", 30)
    print("\n================ 방법 랭킹 (group A=OU/OUT 평균 J') ================")
    print(views["method_rank"].to_string(index=False))
    print("\n================ 방법별 최고 채널 (J' top) ================")
    print(views["best_per_method"].groupby("method", sort=False).head(3).to_string(index=False))
    print("\n================ logic별 (방법×logic 평균 J') 일부 ================")
    bl = views["by_logic"]
    print(bl[bl["logic"].isin(["OU", "OUT"])].to_string(index=False))


if __name__ == "__main__":
    main()
