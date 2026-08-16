"""
appendix_tables.py
===================
IAA 논문 부록 A/B 표 생성. 읽기 전용, C_PD 아래 아무것도 쓰지 않는다.

부록 A: 손라벨 완결(onset/peak/end 3종 모두 식별된) 이벤트, N=53(peak_count<2
       인 ambiguous 8건 포함).
부록 B: NOAA GOES SPE 카탈로그(2019-2025), N=42. 부록 A와 tol=24h 대응.

재사용 (재구현 없음 -- import만):
  predict_v0/1_check_labels.py : load_manual_labels / report_completeness /
      build_reconciled_events 그대로 import. 숫자로 시작하는 파일명이라
      importlib.import_module 사용(gate_persistence_sweep.py와 동일 방식).
  predict_v0/2_build_dataset.py : build_event_spans 그대로 import -- 완결 이벤트
      필터 + ambiguous_peak 플래그. gate_persistence_sweep.py:load_manual_catalog()
      가 쓰는 것과 동일한 재사용 경로(그 함수 자체는 event_id/end_time을 버려서
      부록 A용으로는 못 쓰지만, 내부에서 부르는 build_reconciled_events/
      build_event_spans 조합은 그대로 재현).
  gate_persistence_sweep.py : verify_preproc(vp) 임포트 방식을 그대로 재사용
      (import gate_persistence_sweep as gps; vp = gps.vp) -- vp.noaa_goes_spe_io.load
      + vp.noaa_goes_spe_io.filter_by_date(*vp.CATALOG_ERA) 로 카탈로그를 읽는다.
      gate_persistence_sweep.py의 main()이 카탈로그를 읽는 것과 완전히 같은 경로.
  vp.matcher(=_match_core_poes).match_events : tol_h=24로 부록 A <-> NOAA 카탈로그
      대응을 구한다. 카탈로그를 한 행씩 단일행 cat으로 넣어 match_events를 그대로
      호출하고, 그 함수가 반환하는 onset_diff_h(그 함수 자신의 계산값)를 역산해
      어떤 부록 A event_id가 매칭됐는지 찾는다 -- 매칭(허용오차 tol_h 판정) 로직은
      전혀 재구현하지 않고 match_events의 출력만 그대로 소비한다.

출력: 같은 디렉터리에 appendix_A_manual_events.{csv,md}, appendix_B_noaa_catalog.{csv,md}.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd

# 이 스크립트는 window_result/ 로 옮겨져 있어 gate_persistence_sweep.py 는 한 단계
# 위(POES/)에 있다 -- 출력은 그대로 이 스크립트가 있는 디렉터리(window_result/)에 쓴다.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gate_persistence_sweep as gps  # noqa: E402  -- vp(verify_preproc)/matcher 재사용
vp = gps.vp  # verify_preproc 모듈. gps가 이미 import해둔 것을 그대로 재사용(재-import 아님).

sys.path.insert(0, str(vp.C_PD / "predict_v0"))
_check = importlib.import_module("1_check_labels")     # numeric prefix -> importlib (gps.py와 동일 방식)
_dataset = importlib.import_module("2_build_dataset")  # numeric prefix -> importlib

DETECTOR, CHANNEL = "metop03", "omni_p6"
TOL_H = 24.0


# ──────────────────────────────────────────────────────────────────
# 부록 A -- 손라벨 완결 이벤트
# ──────────────────────────────────────────────────────────────────
def build_appendix_a() -> pd.DataFrame:
    """1_check_labels/2_build_dataset을 그대로 호출해 완결 이벤트 + ambiguous_peak을 얻고,
    표시에 필요한 peak_count만 원본 events에서 붙인다(단순 컬럼 결합, 매칭/판정 로직 없음)."""
    labels_path = vp.C_PD / "predict_v0" / "manual_labels" / f"manual_labels_{DETECTOR}_{CHANNEL}.csv"
    df = _check.load_manual_labels(labels_path)
    comp = _check.report_completeness(df)
    events = _check.build_reconciled_events(df, comp["split_pairs"])
    peak_lt2_ids = set(events.loc[events["peak_count"] < 2, "event_id"])
    spans = _dataset.build_event_spans(events, peak_lt2_ids)   # [event_id, onset_time, peak_time, end_time, ambiguous_peak]
    tbl = spans.merge(events[["event_id", "peak_count"]], on="event_id", how="left")
    return tbl.sort_values("onset_time").reset_index(drop=True)


def _fmt(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M")


def format_appendix_a(tbl: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "event_id": tbl["event_id"].astype(int),
        "onset (UTC)": tbl["onset_time"].map(_fmt),
        "peak (UTC)": tbl["peak_time"].map(_fmt),
        "end (UTC)": tbl["end_time"].map(_fmt),
        "peak_count": tbl["peak_count"].round(2),
        "ambiguous": tbl["ambiguous_peak"].map({True: "yes", False: ""}),
    })


# ──────────────────────────────────────────────────────────────────
# 부록 B -- NOAA SPE 카탈로그 + 부록 A 대응
# ──────────────────────────────────────────────────────────────────
def build_appendix_b(appendix_a_raw: pd.DataFrame) -> pd.DataFrame:
    """gate_persistence_sweep.py의 카탈로그 로딩 경로(vp.noaa_goes_spe_io.load +
    filter_by_date(*vp.CATALOG_ERA))를 그대로 재현하고, vp.matcher.match_events(tol_h=24)
    를 카탈로그 행마다 1개씩(단일행 cat) 호출해 대응을 구한다."""
    cache_dir = vp.C_PD / "NOAA_GOES" / "noaa_goes_spe_cache_parquet"
    cat_all, _ = vp.noaa_goes_spe_io.load(str(cache_dir))
    cat = vp.noaa_goes_spe_io.filter_by_date(cat_all, *vp.CATALOG_ERA)
    if cat.index.tz is None:
        cat.index = cat.index.tz_localize("UTC")

    det = appendix_a_raw[["event_id", "onset_time", "peak_time"]].copy()

    rows = []
    for i in range(len(cat)):
        begin_ts = cat.index[i]
        cat_row = cat.iloc[[i]]
        r = vp.matcher.match_events(det, cat_row, tol_h=TOL_H)
        eid, diff_h = None, None
        if r["n_hit"] >= 1:
            # match_events 자신이 계산한 onset_diff_h(det 온셋 - 카탈로그 begin, 시간)를
            # 역산해 매칭된 det 온셋 시각을 복원하고, det에서 그 시각에 가장 가까운
            # 행의 event_id를 찾는다 -- 매칭 판정 자체는 match_events가 이미 끝냈고,
            # 여기선 그 결과값을 되짚어 event_id로 번역만 한다(재구현 아님).
            diff_h = float(r["onset_diff_h"][0])
            matched_onset = begin_ts + pd.Timedelta(hours=diff_h)
            j = (det["onset_time"] - matched_onset).abs().idxmin()
            eid = int(det.loc[j, "event_id"])
        rows.append({
            "begin_time": begin_ts,
            "max_time": cat_row["max_time"].iloc[0],
            "max_pfu": cat_row["max_pfu"].iloc[0],
            "manual_event_id": eid,
            "onset_diff_h": diff_h,
        })
    return pd.DataFrame(rows).sort_values("begin_time").reset_index(drop=True)


def format_appendix_b(tbl: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "#": range(1, len(tbl) + 1),
        "begin (UTC)": tbl["begin_time"].map(_fmt),
        "max (UTC)": tbl["max_time"].map(lambda t: _fmt(t) if pd.notna(t) else "-"),
        "max_pfu": tbl["max_pfu"],
        "손라벨 대응 event_id": tbl["manual_event_id"].map(lambda v: str(int(v)) if pd.notna(v) else "-"),
        "onset 차이(h)": tbl["onset_diff_h"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "-"),
    })


# ──────────────────────────────────────────────────────────────────
# 출력 (CSV + Word 붙여넣기용 Markdown)
# ──────────────────────────────────────────────────────────────────
def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


CAPTION_A = (
    "**Table A1.** Complete hand-labeled solar proton enhancements (onset, peak, "
    "and end all identified), N=53. Events marked `ambiguous` (peak_count < 2 pfu, "
    "N=8) sit near the detection floor and are retained for completeness but "
    "treated as borderline cases in the analysis (Sec. 2.2)."
)
CAPTION_B = (
    "**Table B1.** NOAA GOES Solar Proton Event catalog entries, 2019-2025 "
    "(N=42), matched against the hand-labeled complete-onset/peak/end catalog "
    "(Table A1) at a 24 h tolerance. `onset diff (h)` is the hand-labeled onset "
    "minus the catalog begin time (negative: the hand label leads the catalog "
    "record). Unmatched entries are marked \"-\"."
)


def main():
    out_dir = Path(__file__).resolve().parent

    a_raw = build_appendix_a()
    n_ambiguous = int(a_raw["ambiguous_peak"].sum())
    print(f"[appendix] A: 완결 이벤트 {len(a_raw)}개 (ambiguous peak_count<2: {n_ambiguous}개)")
    a_fmt = format_appendix_a(a_raw)
    a_fmt.to_csv(out_dir / "appendix_A_manual_events.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "appendix_A_manual_events.md", "w", encoding="utf-8") as f:
        f.write(to_markdown(a_fmt) + "\n\n" + CAPTION_A + "\n")
    print(f"[appendix] 저장 -> {out_dir / 'appendix_A_manual_events.csv'} / .md")

    b_raw = build_appendix_b(a_raw)
    n_matched = int(b_raw["manual_event_id"].notna().sum())
    print(f"[appendix] B: NOAA 카탈로그 {len(b_raw)}개, 손라벨(부록 A) 매칭 {n_matched}개 (tol={TOL_H}h)")
    b_fmt = format_appendix_b(b_raw)
    b_fmt.to_csv(out_dir / "appendix_B_noaa_catalog.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "appendix_B_noaa_catalog.md", "w", encoding="utf-8") as f:
        f.write(to_markdown(b_fmt) + "\n\n" + CAPTION_B + "\n")
    print(f"[appendix] 저장 -> {out_dir / 'appendix_B_noaa_catalog.csv'} / .md")


if __name__ == "__main__":
    main()
