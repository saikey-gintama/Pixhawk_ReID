"""
verify_preproc.py
==================
POES 온보드 파이프라인 S0 검증 -- 전처리 z 정합성 / 검출 일치 / 게이트(PRE_ALERT) POD.
읽기 전용: C_PD(기존 데이터·스크립트) 아래 아무것도 쓰지 않는다.

재사용 (재구현 없음 -- import만):
  fsm_count_spe_quietoff_mad_poes.py : compute_rolling_bg / build_threshold / detect_segments
      (KSEM 원본과 byte-동일 엔진, 절대 수정 금지 -- 이 파일에서도 그대로 호출만 한다)
  _match_core_poes.py                : match_events (NOAA 카탈로그 매칭)
  poes_metop03_io.py / noaa_goes_spe_io.py : count / 카탈로그 로드

세 가지 검증 (각각 출력, 실패해도 계속 진행 -- 세 결과를 한 번에 보기 위함):
  S0-1 z 정합성    : 스트리밍 방식(15분 리샘플 + rolling 배경)으로 재계산한 z
                      vs predict_v0/dataset_v0/timeseries_metop03_omni_p6.parquet 의 zscore.
                      목표: 최대 절대오차 < 1e-9.
  S0-2 검출 일치    : detect_segments(min_duration_h=MIN_SPE_DURATION_H) 재현 onset
                      vs 오프라인 fsm_onset_quietoff_mad_w7_k7_on0.1_pk0.csv (omni_p6, 224개).
                      목표: 완전 일치.
  S0-3 게이트 POD  : detect_segments(min_duration_h=0) = PRE_ALERT 등가(임계 넘는 즉시 트리거,
                      지속성 요구 없음) onset을 NOAA SPE 카탈로그와 매칭.
                      PRE_ALERT 집합은 ALERT(S0-2) 집합의 상위집합이므로
                      POD >= 0.738(오프라인 raw onset 기준값) 이어야 한다.

경로: REPO 환경변수(없으면 플랫폼 기본값) 기준 상대경로만 사용 -- 절대경로 하드코딩 금지.
  Windows(개발) 기본값: D:/VS_code/Pixhawk_ReID
  Jetson(실행)  기본값: $HOME/jeongin/Pixhawk_ReID (KSEM/run_resource_experiment.sh 와 동일)

사용:
  python verify_preproc.py
  REPO=/custom/path python verify_preproc.py --cache ... --dataset ... --onset-csv ... --catalog ...
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd


def _default_repo() -> Path:
    if os.name == "nt":
        return Path("D:/VS_code/Pixhawk_ReID")
    return Path.home() / "jeongin" / "Pixhawk_ReID"


REPO = Path(os.environ.get("REPO", _default_repo()))
C_PD = REPO / "Experiment_window" / "C_PD"   # 기존 데이터·스크립트 -- 읽기 전용

for _p in (C_PD / "predict_v0", C_PD / "POES", C_PD / "POES" / "count_FSM",
           C_PD / "POES" / "MetOp03_count", C_PD / "POES" / "event_MATCHER",
           C_PD / "NOAA_GOES"):
    sys.path.insert(0, str(_p))

import fsm_count_spe_quietoff_mad_poes as fsm_engine  # noqa: E402
import _match_core_poes as matcher                    # noqa: E402
import poes_metop03_io as metop03_io                   # noqa: E402
import noaa_goes_spe_io                                # noqa: E402

# ── 재현 대상 파라미터 (md 1절 고정값 -- 임의 변경 금지) ────────────────────
BG_WINDOW_DAYS = 7
Z_EPS = 1.0
Z_CLIP = 10.0
K = 7
ONSET_FLOOR = 0.1
TOL_H = 24.0
CATALOG_ERA = ("2019-01-01", "2025-12-31")
GATE_POD_MIN = 0.738


def load_omni_p6_count(cache_dir: Path) -> pd.Series:
    """omni_p6 채널만 부분 로드(채널당 파일 분리 캐시) -> 15분 리샘플 count."""
    tpl = metop03_io.fname_to_tuple("omni_p6")
    df_count, _ = metop03_io.load(str(cache_dir), channels=[tpl])
    if df_count.index.tz is None:
        df_count.index = df_count.index.tz_localize("UTC")
    cnt = df_count[tpl].dropna()
    return cnt.resample("15min").mean().dropna()


def snapshot_parquet(src: Path) -> Path:
    """다른 학습 작업이 동시에 덮어쓸 수 있어 임시 경로로 복사 후 그 사본을 읽는다."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="verify_preproc_"))
    dst = tmp_dir / src.name
    shutil.copy2(src, dst)
    return dst


def check_zscore(cnt: pd.Series, dataset_parquet: Path) -> None:
    print("=" * 70)
    print("[S0-1] 전처리 z 정합성 (스트리밍 재현 z vs dataset_v0 zscore)")
    print("=" * 70)
    snap = snapshot_parquet(dataset_parquet)
    try:
        ref = pd.read_parquet(snap)
    finally:
        shutil.rmtree(snap.parent, ignore_errors=True)
    ref["time"] = pd.to_datetime(ref["time"], utc=True)
    ref = ref.set_index("time")

    bg = fsm_engine.compute_rolling_bg(cnt, BG_WINDOW_DAYS, None, fsm_engine.BG_UPDATE_FREQ)
    std_safe = bg["bg_std"].clip(lower=Z_EPS)
    z = ((cnt - bg["bg_median"]) / std_safe).clip(-Z_CLIP, Z_CLIP)

    both = ref[["zscore"]].join(z.rename("zscore_stream"), how="inner")
    diff = (both["zscore"] - both["zscore_stream"]).abs()
    n_mismatch = int((diff > 1e-9).sum())
    max_diff = float(diff.max()) if len(diff) else float("nan")
    first_mismatch = diff[diff > 1e-9].index.min() if n_mismatch else None

    print(f"대조 표본 수: {len(both)}  (offline={len(ref)}, stream={len(z)})")
    print(f"최대 절대오차: {max_diff:.3e}")
    print(f"불일치 표본 수(>1e-9): {n_mismatch}")
    print(f"첫 불일치 시각: {first_mismatch}")
    print(f"목표(max abs diff < 1e-9) 충족: {max_diff < 1e-9}")


def check_detection(cnt: pd.Series, offline_onset_csv: Path) -> list:
    print("=" * 70)
    print("[S0-2] 검출 일치 (ALERT-등가, min_duration_h=MIN_SPE_DURATION_H)")
    print("=" * 70)
    bg = fsm_engine.compute_rolling_bg(cnt, BG_WINDOW_DAYS, None, fsm_engine.BG_UPDATE_FREQ)
    thr = fsm_engine.build_threshold(bg, K, ONSET_FLOOR)
    segs_alert = fsm_engine.detect_segments(cnt, thr, bg, fsm_engine.MIN_SPE_DURATION_H)
    stream_onsets = sorted(pd.Timestamp(s["onset_time"]) for s in segs_alert)

    off = pd.read_csv(offline_onset_csv)
    off_omni = off[off["channel"] == "omni_p6"]
    offline_onsets = sorted(pd.to_datetime(off_omni["onset_time"], utc=True))

    print(f"오프라인 raw onset(omni_p6): {len(offline_onsets)}개")
    print(f"스트리밍 재현 onset: {len(stream_onsets)}개")

    set_off, set_stream = set(offline_onsets), set(stream_onsets)
    missing = sorted(set_off - set_stream)
    extra = sorted(set_stream - set_off)
    exact = offline_onsets == stream_onsets
    print(f"완전 일치: {exact}")
    if not exact:
        print(f"오프라인에만 있음: {len(missing)}개 (앞 5개: {missing[:5]})")
        print(f"스트리밍에만 있음: {len(extra)}개 (앞 5개: {extra[:5]})")
    return segs_alert


def check_gate_pod(cnt: pd.Series, segs_alert: list, catalog_dir: Path) -> None:
    print("=" * 70)
    print("[S0-3] 게이트(PRE_ALERT) POD -- NOAA SPE 카탈로그, tol=24h")
    print("=" * 70)
    bg = fsm_engine.compute_rolling_bg(cnt, BG_WINDOW_DAYS, None, fsm_engine.BG_UPDATE_FREQ)
    thr = fsm_engine.build_threshold(bg, K, ONSET_FLOOR)
    segs_prealert = fsm_engine.detect_segments(cnt, thr, bg, min_duration_h=0.0)

    cat_all, _ = noaa_goes_spe_io.load(str(catalog_dir))
    cat = noaa_goes_spe_io.filter_by_date(cat_all, *CATALOG_ERA)

    r_alert = matcher.match_events(pd.DataFrame(segs_alert), cat, tol_h=TOL_H)
    r_pre = matcher.match_events(pd.DataFrame(segs_prealert), cat, tol_h=TOL_H)

    print(f"[ALERT    ] n_det={r_alert['n_det']:>4d}  POD={r_alert['pod']:.3f}  "
          f"event_FAR={r_alert['event_far']:.3f}  precision={r_alert['precision']:.3f}  "
          f"f1={r_alert['f1']:.3f}")
    print(f"[PRE_ALERT] n_det={r_pre['n_det']:>4d}  POD={r_pre['pod']:.3f}  "
          f"event_FAR={r_pre['event_far']:.3f}  precision={r_pre['precision']:.3f}  "
          f"f1={r_pre['f1']:.3f}")
    print("오프라인 기준값(md 명시): POD=0.738 event_FAR=0.114 precision=0.886 f1=0.805 raw onset 224개")
    print(f"게이트 조건(PRE_ALERT POD >= {GATE_POD_MIN}) 충족: {r_pre['pod'] >= GATE_POD_MIN}")


def main():
    ap = argparse.ArgumentParser(
        description="POES 온보드 S0: 전처리 z 정합성 / 검출 일치 / 게이트 POD 검증 (읽기 전용)")
    ap.add_argument("--cache", default=str(C_PD / "POES" / "MetOp03_count" / "poes_metop03_cache_parquet"))
    ap.add_argument("--dataset", default=str(
        C_PD / "predict_v0" / "dataset_v0" / "timeseries_metop03_omni_p6.parquet"))
    ap.add_argument("--onset-csv", default=str(
        C_PD / "POES" / "MetOp03_count" / "metop03_output" / "2_fsm"
        / "quietoff_mad_w7_k7_on0.1_pk0" / "fsm_onset_quietoff_mad_w7_k7_on0.1_pk0.csv"))
    ap.add_argument("--catalog", default=str(C_PD / "NOAA_GOES" / "noaa_goes_spe_cache_parquet"))
    args = ap.parse_args()

    cache_dir = Path(args.cache)
    dataset_parquet = Path(args.dataset)
    onset_csv = Path(args.onset_csv)
    catalog_dir = Path(args.catalog)

    print(f"[verify_preproc] REPO={REPO}")
    print(f"[verify_preproc] cache={cache_dir}")
    cnt = load_omni_p6_count(cache_dir)
    print(f"[verify_preproc] omni_p6 15분 리샘플 표본 수: {len(cnt)}  "
          f"({cnt.index[0]} ~ {cnt.index[-1]})")

    check_zscore(cnt, dataset_parquet)
    segs_alert = check_detection(cnt, onset_csv)
    check_gate_pod(cnt, segs_alert, catalog_dir)


if __name__ == "__main__":
    main()
