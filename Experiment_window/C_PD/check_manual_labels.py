"""
check_manual_labels.py
=======================
manual_labels_{detector}_{channel}.csv 품질 점검 + FSM/카탈로그 대비 리포트.
label_events_gui.py 로 만든 손라벨을 모델 착수 전에 검증하는 1회성 분석 스크립트
(라벨이 늘어날 때마다 재실행 가능하도록 스크립트로 남김).

재사용 (재구현 없음 -- import만):
  label_events_gui.py                 : load_channel_series (count 로드, --show-fsm과 동일 경로)
  fsm_count_spe_quietoff_mad_poes.py  : compute_rolling_bg (bg_median/std -- diag의 --zscore와
      동일 공식으로 peak/bg 비율 계산에 재사용)
  _match_core_poes.py                 : _cluster_indices(24h 시퀀셜 클러스터링 -- FSM 재검출
      묶음에 재사용, event_far_reeval 결론과 동일 방식), det_matched_mask(tol_h=24 매칭)
  noaa_goes_spe_io                    : load (NOAA SPE 카탈로그 로드)

수동 라벨 45개→57개(2026-07-27 기준)로 늘어난 상태 -- 정확한 개수는 이 스크립트가
직접 세어 보고(사용자가 구두로 말한 45/42는 검증 대상이지 전제가 아님).

사용:
  python check_manual_labels.py --detector metop03 --channel omni_p6 --fsm-w 7 --fsm-k 7
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "POES"))
sys.path.insert(0, str(HERE / "POES" / "event_MATCHER"))
sys.path.insert(0, str(HERE / "POES" / "count_FSM"))

from label_events_gui import load_channel_series, _POES_IO  # noqa: E402
import _match_core_poes as core                              # noqa: E402
import fsm_count_spe_quietoff_mad_poes as fsm_engine          # noqa: E402

MATCH_TOL_H = core.MATCH_TOL_H  # 24.0, 프로젝트 전역 관례
_TYPE_ORDER = {"o": 0, "p": 1, "e": 2}

_CATALOG_DIR = HERE / "NOAA_GOES" / "noaa_goes_spe_cache_parquet"
_CATALOG_IO = "noaa_goes_spe_io"

# 이미 디스크에 존재하는 사전 계산 FSM onset 산출물(quietoff_mad, w7k7) --
# 이 세션의 diag_rolling_threshold_poes.py 사용 예시들이 반복적으로 쓴
# onset_floor=0.1, peak_floor=0(피크 필터 없음 -- FSM recall을 낮게 왜곡하지 않도록)
# 조합을 그대로 채택. 새로 재계산하지 않고 기존 산출 CSV를 그대로 읽는다.
_FSM_ONSET_CSV = (HERE / "POES" / "MetOp03_count" / "metop03_output" / "2_fsm" /
                  "quietoff_mad_w7_k7_on0.1_pk0" / "fsm_onset_quietoff_mad_w7_k7_on0.1_pk0.csv")


def load_manual_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"])
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC")
    return df


def report_completeness(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 70)
    print("[1-a] 완결성 점검")
    print("=" * 70)
    eids = sorted(df["event_id"].unique())
    print(f"unique event_id: {len(eids)}개 (range {min(eids)}~{max(eids)})")
    missing_ids = [i for i in range(min(eids), max(eids) + 1) if i not in eids]
    print(f"건너뛴(삭제된) event_id: {missing_ids}")

    by_eid = df.groupby("event_id")["label_type"].apply(lambda s: sorted(s.tolist()))
    broken = {eid: types for eid, types in by_eid.items() if set(types) != {"o", "p", "e"} or len(types) != 3}
    print(f"\no/p/e 3종 미완성 event_id: {len(broken)}개")
    for eid, types in broken.items():
        rows = df[df["event_id"] == eid][["label_type", "time"]].to_string(index=False)
        print(f"  event {eid}: types={types}\n{rows}")

    # 인접 id 쌍(하나는 'o'만, 다음은 'p'+'e'만)이 서로 24h 이내면 -- GUI에서 'o'를
    # 두 번 눌러 하나의 실제 이벤트가 둘로 쪼개진 전형적 패턴(동일 물리 이벤트 추정).
    split_pairs = []
    broken_ids = sorted(broken)
    for i in range(len(broken_ids) - 1):
        a, b = broken_ids[i], broken_ids[i + 1]
        ta, tb = set(broken[a]), set(broken[b])
        if b == a + 1 and ta == {"o"} and tb == {"p", "e"}:
            t_o = df[(df["event_id"] == a)]["time"].iloc[0]
            t_p = df[(df["event_id"] == b) & (df["label_type"] == "p")]["time"].iloc[0]
            gap_h = (t_p - t_o).total_seconds() / 3600
            split_pairs.append((a, b, gap_h))
    print(f"\n'o'만 있는 id 바로 다음이 'p'+'e'만 있는 id인 쌍(동일 이벤트 분리 추정): {len(split_pairs)}개")
    for a, b, gap_h in split_pairs:
        print(f"  event {a}(o) + event {b}(p,e)  -- onset~peak gap={gap_h:.1f}h "
              f"=> 병합 권장(하나의 물리적 이벤트로 보임, 24h tol 이내)")

    nan_rows = df[df["count"].isna()]
    print(f"\ncount 결측 행: {len(nan_rows)}개")
    for _, r in nan_rows.iterrows():
        print(f"  event {r['event_id']} {r['label_type']} @ {r['time']}  <- 원본 count 시계열의 "
              f"실제 데이터 공백(라벨링 오류 아님, 클릭 스냅 지점 근방 15분 격자에 샘플 없음)")

    n_unsure = int((df["note"] == "unsure").sum())
    print(f"\nunsure 플래그: {n_unsure}개")

    dup = df[df.duplicated(subset=["event_id", "label_type"], keep=False)]
    print(f"중복(event_id,label_type) 행: {len(dup)}개")

    return {"eids": eids, "missing_ids": missing_ids, "broken": broken,
            "split_pairs": split_pairs, "nan_rows": nan_rows}


def build_reconciled_events(df: pd.DataFrame, split_pairs: list) -> pd.DataFrame:
    """57개 원시 event_id -> 병합/완결 이벤트 테이블(onset/peak/end time+count).
    split_pairs의 (a,b)는 a의 'o' + b의 'p'/'e'를 하나의 event_id=a로 병합.
    그 외 미완성(3종 아닌) id는 병합 대상이 아니면 그대로 두되 onset/peak/end 중
    없는 값은 NaT/NaN으로 채워 리포트에는 남기고, 이후 매칭/데이터셋 생성에서는 제외."""
    merge_map = {b: a for a, b, _ in split_pairs}  # b의 행을 a로 흡수
    work = df.copy()
    work["event_id"] = work["event_id"].map(lambda e: merge_map.get(e, e))

    rows = []
    for eid, g in work.groupby("event_id"):
        rec = {"event_id": eid}
        for t in ("o", "p", "e"):
            sub = g[g["label_type"] == t]
            rec[f"{t}_time"] = sub["time"].iloc[0] if len(sub) else pd.NaT
            rec[f"{t}_count"] = sub["count"].iloc[0] if len(sub) else np.nan
        rows.append(rec)
    events = pd.DataFrame(rows).sort_values("event_id").reset_index(drop=True)
    events = events.rename(columns={"o_time": "onset_time", "p_time": "peak_time", "e_time": "end_time",
                                    "o_count": "onset_count", "p_count": "peak_count", "e_count": "end_count"})
    return events


def report_peak_distribution(events: pd.DataFrame, cnt: pd.Series, w_days: int) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print(f"[1-b] peak_count 분포 + peak/bg_median 비율 (bg: compute_rolling_bg w={w_days}d 재사용)")
    print("=" * 70)
    complete = events.dropna(subset=["onset_time", "peak_time", "end_time"]).copy()
    pc = complete["peak_count"].dropna()
    print(f"완결 이벤트 {len(complete)}개 중 peak_count 유효 {len(pc)}개")
    bins = [0, 1, 2, 5, 10, 50, 100, 500, 1000, np.inf]
    labels = ["<1", "1-2", "2-5", "5-10", "10-50", "50-100", "100-500", "500-1k", "1k+"]
    hist = pd.cut(pc, bins=bins, labels=labels, right=False).value_counts().reindex(labels)
    print("\npeak_count 히스토그램:")
    for lab, n in hist.items():
        print(f"  {lab:>8}: {'#' * int(n)} ({n})")

    bg = fsm_engine.compute_rolling_bg(cnt, w_days, None, fsm_engine.BG_UPDATE_FREQ)
    bg_med = bg["bg_median"].reindex(cnt.index).ffill()

    def _bg_at(t):
        pos = bg_med.index.get_indexer([t], method="nearest", tolerance=pd.Timedelta(hours=1))[0]
        return float(bg_med.iloc[pos]) if pos >= 0 else np.nan

    low = complete[complete["peak_count"] < 2].copy()
    low["bg_median_at_peak"] = low["peak_time"].apply(_bg_at)
    low["peak_over_bg"] = low["peak_count"] / low["bg_median_at_peak"]
    low = low.sort_values("event_id")
    print(f"\npeak_count < 2 인 이벤트: {len(low)}개 (peak/bg_median 비율 포함)")
    print(low[["event_id", "onset_time", "peak_time", "peak_count",
              "bg_median_at_peak", "peak_over_bg"]].to_string(index=False))
    print("\n판단 기준: peak_over_bg 가 1근방이면 배경과 구분 안 됨(애매/재검토 후보),")
    print("           수배 이상이면 절대값은 작아도 배경 대비 유의미(진짜 약한 이벤트).")
    return low


def report_overlaps(events: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("[1-c] 겹치는 이벤트(2차 피크 후보) -- 이전 이벤트 end 전에 다음 onset 발생")
    print("=" * 70)
    complete = events.dropna(subset=["onset_time", "end_time"]).sort_values("onset_time").reset_index(drop=True)
    pairs = []
    for i in range(len(complete) - 1):
        cur, nxt = complete.iloc[i], complete.iloc[i + 1]
        if pd.notna(cur["end_time"]) and nxt["onset_time"] < cur["end_time"]:
            pairs.append({
                "event_a": int(cur["event_id"]), "event_b": int(nxt["event_id"]),
                "a_end": cur["end_time"], "b_onset": nxt["onset_time"],
                "overlap_h": (cur["end_time"] - nxt["onset_time"]).total_seconds() / 3600,
            })
    ov = pd.DataFrame(pairs)
    print(f"겹치는 쌍: {len(ov)}개")
    if len(ov):
        print(ov.to_string(index=False))
    return ov


def report_fsm_comparison(events: pd.DataFrame, tol_h: float = MATCH_TOL_H) -> dict:
    print("\n" + "=" * 70)
    print(f"[2-a] FSM(omni_p6, quietoff_mad w7k7 on0.1_pk0) 대비 (tol={tol_h}h)")
    print("=" * 70)
    manual = events.dropna(subset=["onset_time"]).copy()
    fsm_raw = pd.read_csv(_FSM_ONSET_CSV, parse_dates=["onset_time", "peak_time", "end_time"])
    fsm_raw = fsm_raw[fsm_raw["channel"] == "omni_p6"].reset_index(drop=True)
    if fsm_raw["onset_time"].dt.tz is None:
        fsm_raw["onset_time"] = fsm_raw["onset_time"].dt.tz_localize("UTC")
    print(f"손라벨(완결+병합) onset {len(manual)}개, FSM 원시 재검출 {len(fsm_raw)}개")

    # 24h 시퀀셜 클러스터링으로 FSM 재검출을 '고유 알람 사건' 단위로 묶음
    # (_match_core_poes._cluster_indices 재사용 -- event_far_reeval 결론과 동일 방식)
    clusters = core._cluster_indices(fsm_raw["onset_time"].values, tol_h)
    fsm_events = fsm_raw.iloc[[int(g[0]) for g in clusters]].copy()  # 클러스터 대표=최초 onset
    print(f"FSM 24h 클러스터링 후 고유 알람 사건: {len(fsm_events)}개")

    # manual -> FSM 매칭 (det_matched_mask 재사용: det에 'onset_time', cat은 DatetimeIndex)
    fsm_cat = fsm_events.set_index("onset_time")
    manual_hit = core.det_matched_mask(manual, fsm_cat, tol_h)
    n_recall = int(manual_hit.sum())
    print(f"\nFSM recall: 손라벨 {len(manual)}개 중 FSM이 잡은 것 {n_recall}개 "
          f"({n_recall/len(manual)*100:.1f}%)")
    missed = manual[~manual_hit]
    print(f"FSM이 놓친 손라벨(사람만 잡음): {len(missed)}개")
    print(missed[["event_id", "onset_time", "peak_count"]].to_string(index=False))

    # FSM -> manual 매칭 (역방향, 같은 함수 재사용)
    manual_cat = manual.set_index("onset_time")
    fsm_hit = core.det_matched_mask(fsm_events, manual_cat, tol_h)
    fsm_only = fsm_events[~fsm_hit]
    print(f"\nFSM만 잡고 손라벨엔 없음(사람이 노이즈로 판단했거나 미검토): {len(fsm_only)}개")
    print(fsm_only[["onset_time", "peak_count", "in_saa"]].to_string(index=False))

    return {"manual": manual, "fsm_events": fsm_events, "missed": missed, "fsm_only": fsm_only}


def report_catalog_comparison(events: pd.DataFrame, tol_h: float = MATCH_TOL_H) -> dict:
    print("\n" + "=" * 70)
    print(f"[2-b] NOAA SPE 카탈로그 대비 (tol={tol_h}h)")
    print("=" * 70)
    io = core._import_event_io(_CATALOG_IO, str(_CATALOG_DIR))
    cat_all, _ = io.load(str(_CATALOG_DIR))
    cat = io.filter_by_date(cat_all, *core.ERA)  # 공식 매처(run_matcher)와 동일 필터
    if cat.index.tz is None:
        cat.index = cat.index.tz_localize("UTC")
    print(f"NOAA SPE 카탈로그: 전체 {len(cat_all)}개 -> ERA({core.ERA[0]}~{core.ERA[1]}) 필터 후 {len(cat)}개")

    manual = events.dropna(subset=["onset_time"]).copy()
    hit = core.det_matched_mask(manual, cat, tol_h)
    n_in = int(hit.sum())
    outside = manual[~hit]
    print(f"\n손라벨 {len(manual)}개 중 카탈로그 매칭: {n_in}개, 카탈로그 밖(신규 발견): {len(outside)}개")
    print(outside[["event_id", "onset_time", "peak_count"]].to_string(index=False))
    return {"cat": cat, "outside": outside}


def main():
    ap = argparse.ArgumentParser(description="manual_labels 품질 점검 + FSM/카탈로그 대비")
    ap.add_argument("--detector", default="metop03")
    ap.add_argument("--channel", default="omni_p6")
    ap.add_argument("--fsm-w", type=int, default=7, help="peak/bg_median 계산용 rolling bg 창(일)")
    ap.add_argument("--out-dir", default=str(HERE / "manual_labels" / "quality_check"))
    args = ap.parse_args()

    labels_path = HERE / "manual_labels" / f"manual_labels_{args.detector}_{args.channel}.csv"
    df = load_manual_labels(labels_path)
    print(f"[check] {labels_path} 로드: {len(df)}행")

    comp = report_completeness(df)
    events = build_reconciled_events(df, comp["split_pairs"])
    print(f"\n[reconcile] 병합/완결 처리 후 이벤트 {len(events)}개 "
          f"(완결 3종 {events.dropna(subset=['onset_time','peak_time','end_time']).shape[0]}개)")

    cnt, _ = load_channel_series(args.detector, args.channel)
    low = report_peak_distribution(events, cnt, args.fsm_w)
    overlaps = report_overlaps(events)
    fsm_res = report_fsm_comparison(events)
    cat_res = report_catalog_comparison(events)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(out_dir / "events_reconciled.csv", index=False)
    low.to_csv(out_dir / "peak_lt2_events.csv", index=False)
    overlaps.to_csv(out_dir / "overlapping_pairs.csv", index=False)
    fsm_res["missed"].to_csv(out_dir / "fsm_missed_manual.csv", index=False)
    fsm_res["fsm_only"].to_csv(out_dir / "fsm_only_no_manual.csv", index=False)
    cat_res["outside"].to_csv(out_dir / "manual_outside_catalog.csv", index=False)
    print(f"\n[check] 리포트 CSV 저장 -> {out_dir}")


if __name__ == "__main__":
    main()
