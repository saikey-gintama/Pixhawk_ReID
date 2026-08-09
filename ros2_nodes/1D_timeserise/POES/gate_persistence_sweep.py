"""
gate_persistence_sweep.py
==========================
S0.5 -- 게이트(TCN 기동 조건) 지속성 N(틱) 스윕. 읽기 전용, C_PD 아래 아무것도 쓰지 않는다.
S0(verify_preproc.py)의 z 정합성/검출 일치/게이트 POD 3종 검증은 그대로 두고, 이 스크립트는
"PRE_ALERT를 몇 틱 지속으로 열 것인가"를 데이터로 정하기 위한 별도 산출물이다.

배경: md 5절은 PRE_ALERT=0틱(첫 초과 샘플 즉시)으로 고정돼 있는데 근거가 없다. 게이트를
늦출수록(N 증가) TCN 호출(duty_cycle)은 줄지만 카탈로그 POD가 떨어질 위험이 커진다.

재사용 (재구현 없음 -- import만):
  verify_preproc.py                  : REPO/C_PD 경로 설정, load_omni_p6_count, fsm_engine/
                                        matcher import, K/ONSET_FLOOR/BG_WINDOW_DAYS/TOL_H 상수
  fsm_count_spe_quietoff_mad_poes.py : compute_rolling_bg/build_threshold/detect_segments
  _match_core_poes.py                : match_events (그대로 재사용, 재구현 금지)
  1_check_labels.py                  : load_manual_labels/report_completeness/build_reconciled_events
  2_build_dataset.py                 : build_event_spans

정의:
  run_len(t) = t 에서 끝나는 연속 임계초과 샘플 수 (임계 미만이면 0).
  게이트 N(틱): gate_open(t) <=> run_len(t) >= N.  검출 시각 = 각 run의 N번째 샘플(run_len==N).
  N=1 이 현재 md 의 PRE_ALERT 등가(= S0-3 의 PRE_ALERT 와 동일 집합이어야 함, 교차검증).
  ALERT(참조행): 오프라인 detect_segments(min_duration_h=1)과 등가인 벽시계 1h 경과 설계.
                 재사용 엔진(detect_segments)이 벽시계 기반이므로 이 행이 "실제 구현".
                 검출 시각 = 각 run에서 (t - run_start) >= 1h 를 처음 만족하는 샘플(확정 시각).

출력: stdout 표 + CSV(같은 디렉터리, gate_persistence_sweep.csv). 그림 없음.
"""
from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_preproc as vp  # noqa: E402  -- REPO/C_PD 경로, fsm_engine/matcher, 파라미터, load_omni_p6_count 재사용

_check = importlib.import_module("1_check_labels")     # numeric prefix -> importlib
_dataset = importlib.import_module("2_build_dataset")  # numeric prefix -> importlib

N_SWEEP = [1, 2, 3, 4]
CONFIRM_H = 1.0  # ALERT 벽시계 확정 기준 (md 5절, fsm_engine.MIN_SPE_DURATION_H 과 동일 값)


def compute_run_len(above: pd.Series) -> pd.Series:
    """run_len(t) = t 에서 끝나는 연속 True 개수, False 면 0."""
    grp = (~above).cumsum()
    return above.astype(int).groupby(grp).cumsum()


def compute_run_start(cnt_index: pd.DatetimeIndex, run_len: pd.Series) -> pd.Series:
    """각 t 에 대해 현재(또는 가장 최근) run 의 시작 시각(run_len==1 인 시각)을 ffill."""
    marker = pd.Series(cnt_index, index=cnt_index).where(run_len.values == 1)
    return marker.ffill()


def load_manual_catalog(detector: str, primary: str) -> pd.DataFrame:
    """손라벨 완결 이벤트 -> match_events() 가 기대하는 catalog 형식
    (index=begin_time, columns=[max_time, max_pfu])으로 변환.
    max_pfu 는 PFU 서브지표(pfu_pod, 우리 출력엔 안 씀)에만 쓰여 NaN으로 둔다."""
    labels_path = vp.C_PD / "predict_v0" / "manual_labels" / f"manual_labels_{detector}_{primary}.csv"
    df = _check.load_manual_labels(labels_path)
    comp = _check.report_completeness(df)
    events = _check.build_reconciled_events(df, comp["split_pairs"])
    peak_lt2_ids = set(events.loc[events["peak_count"] < 2, "event_id"])
    spans = _dataset.build_event_spans(events, peak_lt2_ids)
    cat = spans.set_index("onset_time").rename(columns={"peak_time": "max_time"})
    cat["max_pfu"] = np.nan
    return cat[["max_time", "max_pfu"]].sort_index()


def make_det(times) -> pd.DataFrame:
    """match_events() 가 요구하는 최소 컬럼(onset_time, peak_time)만 채운 det 프레임.
    peak_time 은 내부 peak_diff 통계(우리 출력엔 안 씀)에만 쓰여 onset_time 을 복제."""
    t = pd.DatetimeIndex(sorted(times))
    return pd.DataFrame({"onset_time": t, "peak_time": t})


def sweep_row(label: str, det_times, n_ticks: int, total_ticks: int, total_days: float,
             cat_noaa: pd.DataFrame, cat_manual: pd.DataFrame) -> dict:
    det = make_det(det_times)
    r_noaa = vp.matcher.match_events(det, cat_noaa, tol_h=vp.TOL_H)
    r_manual = vp.matcher.match_events(det, cat_manual, tol_h=vp.TOL_H)
    duty = n_ticks / total_ticks
    return {
        "gate": label,
        "n_gate_open_ticks": n_ticks,
        "duty_cycle": duty,
        "n_invocations_day": n_ticks / total_days,
        "n_det": r_noaa["n_det"],
        "POD_noaa": r_noaa["pod"], "event_FAR_noaa": r_noaa["event_far"],
        "precision_noaa": r_noaa["precision"], "f1_noaa": r_noaa["f1"],
        "POD_manual": r_manual["pod"], "event_FAR_manual": r_manual["event_far"],
        "precision_manual": r_manual["precision"], "f1_manual": r_manual["f1"],
        "n_events_total": r_noaa["n_events_total"],
        "duty_cycle_x_base_rate": np.nan,  # main()에서 기저율 계산 후 채움
    }


def event_base_rate(dataset_parquet: Path) -> float:
    """손라벨 label!=0(rising+decreasing) 이 전체 샘플의 몇 %인지 직접 계산."""
    snap = vp.snapshot_parquet(dataset_parquet)
    try:
        ref = pd.read_parquet(snap)
    finally:
        shutil.rmtree(snap.parent, ignore_errors=True)
    return float((ref["label"] != 0).mean())


def confirm_delay_stats(onsets: pd.DatetimeIndex, confirm_times: pd.Series, label: str) -> None:
    """onsets(224개 ALERT onset) 각각에 대해 confirm_times(index=run_start)에서 확정 시각을
    찾아 (확정-onset) 분(minute) 분포의 median/p95 를 출력. confirm_times 에 없는 onset은
    '이 설계로는 확정 안 됨'으로 별도 집계(놓친 이벤트 수)."""
    matched = confirm_times.reindex(onsets)
    found = matched.notna()
    n_missing = int((~found).sum())
    delay_min = (matched[found] - onsets[found]).dt.total_seconds() / 60.0
    if len(delay_min):
        med = float(np.median(delay_min))
        p95 = float(np.percentile(delay_min, 95))
    else:
        med = p95 = float("nan")
    print(f"[{label}] median={med:.1f}min  p95={p95:.1f}min  "
          f"(확정 안 됨/놓침: {n_missing}/{len(onsets)}개)")


def main():
    cache_dir = vp.C_PD / "POES" / "MetOp03_count" / "poes_metop03_cache_parquet"
    catalog_dir = vp.C_PD / "NOAA_GOES" / "noaa_goes_spe_cache_parquet"
    dataset_parquet = vp.C_PD / "predict_v0" / "dataset_v0" / "timeseries_metop03_omni_p6.parquet"
    out_csv = Path(__file__).resolve().parent / "gate_persistence_sweep.csv"

    print(f"[gate_sweep] REPO={vp.REPO}")
    cnt = vp.load_omni_p6_count(cache_dir)
    total_ticks = len(cnt)
    total_days = (cnt.index[-1] - cnt.index[0]).total_seconds() / 86400.0
    print(f"[gate_sweep] omni_p6 표본 {total_ticks}개, 기간 {total_days:.2f}일 "
          f"({cnt.index[0]} ~ {cnt.index[-1]})")

    bg = vp.fsm_engine.compute_rolling_bg(cnt, vp.BG_WINDOW_DAYS, None, vp.fsm_engine.BG_UPDATE_FREQ)
    thr = vp.fsm_engine.build_threshold(bg, vp.K, vp.ONSET_FLOOR)
    th = thr.reindex(cnt.index).ffill()
    above = (cnt >= th) & cnt.notna() & th.notna()   # detect_segments 내부 공식과 동일(재구현 아님)

    run_len = compute_run_len(above)
    run_start_ts = compute_run_start(cnt.index, run_len)

    # ── ALERT(참조행): 벽시계 1h 경과 확정, 실제 재사용 엔진(detect_segments)과 등가 ──
    elapsed_h = pd.Series(
        (cnt.index.values - run_start_ts.values) / np.timedelta64(1, "h"), index=cnt.index)
    alert_open = above & (elapsed_h >= CONFIRM_H)
    n_alert_ticks = int(alert_open.sum())
    tmp = pd.DataFrame({
        "ts": pd.Series(cnt.index[alert_open]),
        "run_start": run_start_ts[alert_open].reset_index(drop=True),
    })  # .values 로 꺼내면 tz-aware -> tz-naive 로 유실돼 이후 reindex 매칭이 깨짐(디버깅으로 확인) -- Series 그대로 정렬
    alert_confirm_times = tmp.groupby("run_start")["ts"].min()

    segs_alert = vp.fsm_engine.detect_segments(cnt, thr, bg, vp.fsm_engine.MIN_SPE_DURATION_H)
    onsets_alert = pd.DatetimeIndex(sorted(pd.Timestamp(s["onset_time"]) for s in segs_alert))
    print(f"[gate_sweep] ALERT(S0-2 등가) 세그먼트 {len(onsets_alert)}개, "
          f"벽시계 확정 run {len(alert_confirm_times)}개")

    # ── 카탈로그 로드 ──
    cat_all, _ = vp.noaa_goes_spe_io.load(str(catalog_dir))
    cat_noaa = vp.noaa_goes_spe_io.filter_by_date(cat_all, *vp.CATALOG_ERA)
    cat_manual = load_manual_catalog("metop03", "omni_p6")
    print(f"[gate_sweep] NOAA 카탈로그 {len(cat_noaa)}개, 손라벨 완결 이벤트 {len(cat_manual)}개")

    # ── N=1..4 틱 스윕 + ALERT 참조행 ──
    rows = []
    for n in N_SWEEP:
        n_ticks = int((run_len >= n).sum())
        det_times = cnt.index[run_len == n]
        rows.append(sweep_row(f"N={n}", det_times, n_ticks, total_ticks, total_days,
                              cat_noaa, cat_manual))
    rows.append(sweep_row("ALERT(wallclock 1h, 실제엔진)", pd.DatetimeIndex(alert_confirm_times.values),
                          n_alert_ticks, total_ticks, total_days, cat_noaa, cat_manual))
    tbl = pd.DataFrame(rows)

    # ── 이벤트 기저율 + duty_cycle 배수 ──
    base_rate = event_base_rate(dataset_parquet)
    tbl["duty_cycle_x_base_rate"] = tbl["duty_cycle"] / base_rate
    print("\n" + "=" * 70)
    print("[S0.5-1] 이벤트 기저율 (손라벨 label!=0 비율, 직접 계산)")
    print("=" * 70)
    print(f"기저율: {base_rate*100:.2f}%  (참고: md/2_build_dataset 로그 기준 "
          f"rising 2.24% + decreasing 6.44% = 8.68%)")

    # ── ALERT 확정 지연 분포: 벽시계(실제 엔진) vs 가상 tick-count(N=4) ──
    print("\n" + "=" * 70)
    print("[S0.5-2] ALERT 확정 지연 분포 (확정 시각 - onset 시각, 분)")
    print("=" * 70)
    confirm_delay_stats(onsets_alert, alert_confirm_times, "wallclock(실제 엔진, elapsed>=60min)")
    mask4 = run_len == 4
    n4_times = pd.DataFrame({
        "ts": pd.Series(cnt.index[mask4]),
        "run_start": run_start_ts[mask4].reset_index(drop=True),
    }).groupby("run_start")["ts"].min()
    confirm_delay_stats(onsets_alert, n4_times, "tick-count(가상 N=4, 45min 상당)")
    print("실제 재사용 엔진(fsm_engine.detect_segments)은 벽시계 경과 방식이다 -- md 5절의 "
          "'판정 지연 45분(3 tick)' 서술은 tick-count 가정값이며 실제 구현과 다르다.")

    # ── 최종 표 ──
    print("\n" + "=" * 70)
    print("[S0.5-3] 게이트 지속성 N 스윕 (omni_p6, tol=24h)")
    print("=" * 70)
    show_cols = ["gate", "n_gate_open_ticks", "duty_cycle", "duty_cycle_x_base_rate",
                "n_invocations_day", "n_det",
                "POD_noaa", "event_FAR_noaa", "precision_noaa", "f1_noaa",
                "POD_manual", "event_FAR_manual", "precision_manual", "f1_manual",
                "n_events_total"]
    fmt = tbl[show_cols].copy()
    for c in ("duty_cycle", "POD_noaa", "event_FAR_noaa", "precision_noaa", "f1_noaa",
             "POD_manual", "event_FAR_manual", "precision_manual", "f1_manual"):
        fmt[c] = fmt[c].round(4)
    for c in ("duty_cycle_x_base_rate", "n_invocations_day"):
        fmt[c] = fmt[c].round(2)
    print(fmt.to_string(index=False))

    tbl.to_csv(out_csv, index=False)
    print(f"\n[gate_sweep] CSV 저장 -> {out_csv}")


if __name__ == "__main__":
    main()
