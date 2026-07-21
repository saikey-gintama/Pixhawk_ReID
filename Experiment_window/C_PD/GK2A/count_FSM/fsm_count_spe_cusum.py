"""
fsm_count_spe_cusum.py
=======================
Count 단독 SPE onset 탐지 — GK2A/KSEM 판 (Poisson 우도비 CUSUM baseline).

fsm_count_spe_cusum_poes.py 의 GK2A 포팅판 — 기존 quietoff 의 POES↔GK2A 포팅과
동일한 관계: 누적 코어(_sigma/_select_quiet_samples/compute_rolling_bg/
detect_segments_cusum, 시간기준 min_duration + gap guard 포함)는 원본과
byte-identical, io/경로/채널 순회만 GK2A(ksem_io, PD_KEYS×SIDES×TARGET_LOGICS)
로 바꿨다.

GK2A(GEO 정지궤도)는 SAA를 통과하지 않는다 — Bmag/geo 데이터 자체가 없다.
POES판의 이중트랙(SAA/non-SAA) 코드는 구조 그대로 유지하되 geo를 항상 None으로
둔다: _saa_bit_series(cnt, None)이 전부 False를 반환 -> cnt_saa가 항상 빈
시계열 -> main()의 "len(cnt_saa)>=MIN_PTS_PER_CHANNEL" 가드에 걸려 SAA 트랙은
자동으로 건너뛰고 quiet 트랙(=cnt 전체)만 실질 동작한다. 별도 분기 불필요 —
이중트랙 로직이 자연히 단일트랙으로 퇴화한다.

CSV 컬럼 스키마는 GK2A quietoff(fsm_count_spe_quietoff_mad.py)와 동일
(POES판에만 있는 in_saa/onset_Bmag/onset_maglat GEO_COLS는 GK2A에 없음).

세그먼트 병합(coalescing, --merge-gap-h 기본 3h): POES판과 byte-identical
(_coalesce_segments) — 즉시강제리셋으로 조각난 지속 이벤트를 출력 직전
후처리로 복원. 검출 코어는 무변경.

사용:
  python fsm_count_spe_cusum.py --window 10 --k 3 --h 5 --onset 0.5 --peak 2
"""

from __future__ import annotations
import sys
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

# ── ksem_io 경로 (count 로드 전용) ────────────────────────────────
_THIS_DIR         = Path(__file__).parent.resolve()
KSEM_COUNT_DIR    = _THIS_DIR.parent / "KSEM_count"
COUNT_PARQUET_DIR = KSEM_COUNT_DIR / "ksem_cache_parquet"
if str(KSEM_COUNT_DIR) not in sys.path:
    sys.path.insert(0, str(KSEM_COUNT_DIR))
import ksem_io   # noqa: E402

# ══════════════════════════════════════════════════════════════════
# 파라미터 (이 블록만 조정 — 검출 로직 본문엔 숫자를 박지 않는다)
# ══════════════════════════════════════════════════════════════════
RESAMPLE_FREQ      = "15min"
BG_UPDATE_FREQ     = "1D"
MIN_SPE_DURATION_H = 0.5   # 30분 연속 warning이어야 onset 확정 (quietoff의 1h와 다름)

# ── 이 파일의 방식 고정 ──────────────────────────────────────────
BG_WINDOW_DAYS = 30        # [day] 배경용 슬라이딩 윈도우
BG_QUIET_DAYS  = None      # None=윈도우 전체 사용
SIGMA_METHOD   = "mad"     # MAD robust sigma (배경 엔진은 quietoff와 동일; CUSUM 계산엔 미사용)

PROTON_LOGICS   = ["O", "OU", "OUT", "CR"]
ELECTRON_LOGICS = ["F", "FT", "FTU", "FTUO"]
TARGET_LOGICS   = PROTON_LOGICS + ELECTRON_LOGICS
PD_KEYS         = ["PD1", "PD2", "PD3"]
SIDES           = ["A", "B"]

K              = 3.0       # λ1 = k·λ0 (event/background 배수)
H              = 5.0       # CUSUM 결정 임계
LAMBDA0_FLOOR  = 0.01      # λ0 하한 클립 (15min 리샘플 count 스케일)
MAX_GAP_H      = 2.0       # warning 진행 중 인접 유효샘플 gap 상한 -> 넘으면 warning 무효화
MERGE_GAP_H    = 3.0       # 같은 트랙 인접 세그먼트 병합 gap 상한[h] (검출 코어 무변경, 출력 후처리 전용)
ONSET_FLOOR    = 0.5
PEAK_FLOOR     = 2.0
MIN_PTS_PER_CHANNEL = 100

TAG = "cusum"              # 출력 파일명 태그

FSM_OUTPUT_DIR = _THIS_DIR / "fsm2_output"
FSM_OUTPUT_DIR.mkdir(exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(
        description=f"FSM count SPE detector [{TAG}] — Poisson 우도비 CUSUM (GK2A).")
    p.add_argument("--window", type=int,   default=BG_WINDOW_DAYS)
    p.add_argument("--k",      type=float, default=K, help="λ1=k·λ0 (event/background 배수, k>1 필수)")
    p.add_argument("--h",      type=float, default=H, help="CUSUM 결정 임계 (기본 5)")
    p.add_argument("--onset",  type=float, default=ONSET_FLOOR)
    p.add_argument("--peak",   type=float, default=PEAK_FLOOR)
    p.add_argument("--lambda0-floor", type=float, default=LAMBDA0_FLOOR,
                   help="λ0(배경률) 하한 클립 (기본 0.01)")
    p.add_argument("--max-gap-h", type=float, default=MAX_GAP_H,
                   help="warning 중 인접 유효샘플 gap 상한[h] -> 넘으면 warning 무효화 (기본 2)")
    p.add_argument("--merge-gap-h", type=float, default=MERGE_GAP_H,
                   help="출력 후처리: 같은 트랙에서 next.onset_time - prev.end_time 이 이 값[h] "
                        "미만이면 인접 세그먼트를 병합 (검출 코어 무변경, 기본 3, 0=끔)")
    p.add_argument("--quiet-days", type=int, default=(BG_QUIET_DAYS if BG_QUIET_DAYS else 0))
    p.add_argument("--out", default=None, help="출력 루트 디렉터리 (기본: count_FSM/fsm2_output)")
    return p.parse_args()


def _numstr(v):
    f = float(v)
    return str(int(f)) if f.is_integer() else str(f)


def build_runtag(tag, window, k, h, onset, peak):
    return f"{tag}_w{window}_k{_numstr(k)}_h{_numstr(h)}_on{_numstr(onset)}_pk{_numstr(peak)}"


# ══════════════════════════════════════════════════════════════════
# 배경 추정 엔진 (fsm_count_spe_cusum_poes.py 원본과 byte-identical — 절대 수정 금지)
# ══════════════════════════════════════════════════════════════════
def _sigma(x) -> float:
    """robust σ = 1.4826 · median(|x - median(x)|).

    단발 스파이크에 강건. 저카운트 채널(OU/OUT)의 quiet 표본에 스파이크가
    섞여도 임계가 폭발하지 않는다. 표본 과반이 동일값이면 0이 될 수 있고,
    그 경우 검출은 floor에 의존(의도된 동작).
    """
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def _select_quiet_samples(cnt: pd.Series, t_ref: pd.Timestamp,
                          bg_window_days: int, bg_quiet_days) -> pd.Series:
    """t_ref 이전 bg_window_days 윈도우의 배경 표본.
    bg_quiet_days=None이면 윈도우 전체, 정수면 하루 median이 가장 낮은 N일만."""
    seg = cnt.loc[t_ref - pd.Timedelta(days=bg_window_days):t_ref].dropna()
    if len(seg) < 10:
        return pd.Series(dtype=float)
    if bg_quiet_days is None:
        return seg
    daily = seg.resample("1D").median().dropna()
    if len(daily) < bg_quiet_days:
        return pd.Series(dtype=float)
    quiet_days = daily.nsmallest(bg_quiet_days)
    return seg[seg.index.normalize().isin(quiet_days.index.normalize())]


def compute_rolling_bg(cnt: pd.Series, bg_window_days: int,
                       bg_quiet_days, update_freq: str = "1D") -> pd.DataFrame:
    """rolling 배경 시계열. update_freq마다 갱신 후 ffill.
    반환: DataFrame[bg_median, bg_std]. bg_std는 이 파일의 _sigma."""
    if cnt.empty:
        return pd.DataFrame(columns=["bg_median", "bg_std"])
    update_points = pd.date_range(cnt.index[0], cnt.index[-1], freq=update_freq)
    rows = []
    for t in update_points:
        q = _select_quiet_samples(cnt, t, bg_window_days, bg_quiet_days)
        if len(q) >= 2:
            rows.append((t, float(q.median()), _sigma(q)))
        elif len(q) == 1:
            rows.append((t, float(q.median()), np.nan))
        else:
            rows.append((t, np.nan, np.nan))
    bg = pd.DataFrame(rows, columns=["t", "bg_median", "bg_std"]).set_index("t")
    return bg.reindex(cnt.index.union(bg.index)).ffill().reindex(cnt.index)


# ══════════════════════════════════════════════════════════════════
# 검정부 (fsm_count_spe_cusum_poes.py 원본과 byte-identical) — Poisson 우도비 CUSUM
# ══════════════════════════════════════════════════════════════════
def detect_segments_cusum(cnt: pd.Series, bg: pd.DataFrame, k: float, h: float,
                          onset_floor: float, min_duration_h: float,
                          lambda0_floor: float, max_gap_h: float = MAX_GAP_H) -> list:
    """Poisson 로그우도비 CUSUM. z-표준화(σ 나눗셈) 없음 — count를 우도비에 직접 투입.

    λ0=배경률(robust median, floor 클립), λ1=k·λ0.
    S[i] = max(0, S[i-1] + cnt[i]*ln(k) - λ0[i]*(k-1))

    onset이 min_duration만큼 확정되는 즉시 S=0으로 강제 리셋하고 warning 상태도
    지운다 — "S가 자연적으로 h 밑에 복귀할 때까지" 기다리지 않는다(POES판에서
    단발 대형 스파이크 이후 전체 구간이 onset 1개로 붕괴하던 문제의 근본 수정).

    지속 판정은 스텝(행) 개수가 아니라 warning 시작~현재 timestamp의 실제
    경과시간(idx[i]-idx[warn_start] >= min_duration_h)으로 한다. 추가로, warning
    진행 중 인접 두 유효 샘플의 실제 시간 gap이 max_gap_h를 넘으면(트랙이 그
    구간에서 너무 듬성해진 것) 그 warning을 무효화한다(warn_start=None, S는 유지).
    """
    if k <= 1.0:
        print(f"[fsm:{TAG}] WARNING k={k}<=1 이면 ln(k)<=0 -> 검출 불가, 전체 skip")
        return []

    idx = cnt.index
    lam0_raw = bg["bg_median"].reindex(idx).ffill().to_numpy(dtype=float)
    sig      = bg["bg_std"].reindex(idx).ffill().to_numpy(dtype=float)
    cnt_v    = cnt.to_numpy(dtype=float)
    n_pts    = len(cnt_v)

    ln_k = np.log(k)

    S = 0.0
    warn_start = None
    prev_valid_t = None
    segs = []

    for i in range(n_pts):
        c = cnt_v[i]
        l0_raw = lam0_raw[i]
        if not (np.isfinite(c) and np.isfinite(l0_raw)):
            continue  # 무효 스텝: 상태(S, warn_start) 유지하고 그냥 건너뜀
        l0 = max(l0_raw, lambda0_floor)
        if l0 <= 0:
            continue

        t = idx[i]
        if warn_start is not None and prev_valid_t is not None:
            gap_h = (t - prev_valid_t).total_seconds() / 3600
            if gap_h > max_gap_h:
                warn_start = None  # 트랙 희박 구간 -> warning 무효화 (S는 유지)
        prev_valid_t = t

        contrib = c * ln_k - l0 * (k - 1.0)
        S = max(0.0, S + contrib)

        if S > h:
            if warn_start is None:
                warn_start = i
            dur_h = (idx[i] - idx[warn_start]).total_seconds() / 3600
            if dur_h >= min_duration_h:
                onset_i, end_i = warn_start, i
                onset_t, end_t = idx[onset_i], idx[end_i]
                if cnt_v[onset_i] >= onset_floor:
                    seg_cnt = cnt.iloc[onset_i:end_i + 1]
                    l0_onset_raw = lam0_raw[onset_i]
                    l0_onset = max(l0_onset_raw, lambda0_floor) if np.isfinite(l0_onset_raw) else np.nan
                    segs.append({
                        "onset_time": onset_t, "peak_time": seg_cnt.idxmax(), "end_time": end_t,
                        "onset_count": round(float(cnt_v[onset_i]), 3),
                        "peak_count":  round(float(seg_cnt.max()), 3),
                        "end_count":   round(float(cnt_v[end_i]), 3),
                        "duration_h":  round((end_t - onset_t).total_seconds() / 3600, 2),
                        "bg_median": round(float(l0_onset), 4) if np.isfinite(l0_onset) else np.nan,
                        "bg_sigma":  round(float(sig[onset_i]), 4) if np.isfinite(sig[onset_i]) else np.nan,
                        "threshold": round(float(k * l0_onset), 4) if np.isfinite(l0_onset) else np.nan,
                    })
                # 확정 즉시 강제 리셋 (자연 복귀 대기 안 함 — 발산 방지 핵심)
                S = 0.0
                warn_start = None
        else:
            warn_start = None  # h 밑으로 내려오면 warning 취소 (S 자체는 유지)

    return segs


def _coalesce_segments(segs: list, cnt: pd.Series, merge_gap_h: float) -> list:
    """같은 트랙 안 인접 세그먼트 병합 (검출 코어 무변경 — 출력 직전 후처리 전용).
    POES판(fsm_count_spe_cusum_poes.py) 원본과 byte-identical — 로직/주석은
    그쪽 참고. GK2A는 SAA 트랙이 항상 빈 채로 퇴화하므로 quiet 트랙에만
    실질적으로 적용된다."""
    if not segs or merge_gap_h <= 0:
        return segs
    segs = sorted(segs, key=lambda s: s["onset_time"])
    merged = [dict(segs[0])]
    for s in segs[1:]:
        prev = merged[-1]
        gap_h = (s["onset_time"] - prev["end_time"]).total_seconds() / 3600
        if gap_h < merge_gap_h:
            prev["end_time"]   = s["end_time"]
            prev["end_count"]  = s["end_count"]
            prev["duration_h"] = round((prev["end_time"] - prev["onset_time"]).total_seconds() / 3600, 2)
            seg_cnt = cnt.loc[prev["onset_time"]:prev["end_time"]]
            if not seg_cnt.empty:
                prev["peak_time"]  = seg_cnt.idxmax()
                prev["peak_count"] = round(float(seg_cnt.max()), 3)
        else:
            merged.append(dict(s))
    return merged


def _saa_bit_series(cnt: pd.Series, geo) -> pd.Series:
    """POES판과 동일 시그니처 유지용. GK2A(GEO)는 SAA/Bmag 데이터가 없어 geo는
    항상 None -> 전부 False -> cnt_saa가 빈 시계열이 되어 이중트랙이 단일트랙
    (quiet만)으로 자연 퇴화한다."""
    if geo is None:
        return pd.Series(False, index=cnt.index)
    return pd.Series(False, index=cnt.index)  # GK2A는 도달하지 않는 분기(geo 항상 None)


# ══════════════════════════════════════════════════════════════════
def load_count() -> pd.DataFrame:
    df_count, _ = ksem_io.load(COUNT_PARQUET_DIR)
    if not df_count.empty and df_count.index.tz is None:
        df_count.index = df_count.index.tz_localize("UTC")
    return df_count


def main():
    args = parse_args()
    window   = args.window
    k        = args.k
    h        = args.h
    onset_fl = args.onset
    peak_fl  = args.peak
    lam0_fl  = args.lambda0_floor
    max_gap  = args.max_gap_h
    merge_gap = args.merge_gap_h
    quiet_d  = args.quiet_days if args.quiet_days > 0 else None
    runtag   = build_runtag(TAG, window, k, h, onset_fl, peak_fl)
    out_dir  = (Path(args.out) if args.out else FSM_OUTPUT_DIR) / runtag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fsm:{TAG}] params: window={window}d k={k} h={h} lambda0_floor={lam0_fl} "
          f"onset={onset_fl} peak={peak_fl} quiet_days={quiet_d}")
    print(f"[fsm:{TAG}] runtag: {runtag}")
    print(f"[fsm:{TAG}] GK2A(GEO)는 SAA 없음 -> 이중트랙이 단일(quiet)트랙으로 퇴화")
    print(f"[fsm:{TAG}] Loading count data...")
    df_count = load_count()
    cnt_rs = df_count.resample(RESAMPLE_FREQ).mean()

    geo = None  # GK2A(GEO)는 Bmag/SAA 데이터 없음

    onset_rows, event_rows = [], []
    for pd_key in PD_KEYS:
        for side in SIDES:
            for logic in TARGET_LOGICS:
                try:
                    cnt = cnt_rs[pd_key, side, logic].dropna()
                except KeyError:
                    continue
                if len(cnt) < MIN_PTS_PER_CHANNEL:
                    continue
                chan = f"{pd_key}{side}-{logic}"
                print(f"\n[fsm:{TAG}] {chan}  (n={len(cnt)} pts)")

                saa_bit   = _saa_bit_series(cnt, geo)
                cnt_quiet = cnt[~saa_bit]
                cnt_saa   = cnt[saa_bit]

                bg_quiet = compute_rolling_bg(cnt_quiet, window, quiet_d, BG_UPDATE_FREQ)
                segs_quiet_raw = detect_segments_cusum(cnt_quiet, bg_quiet, k, h, onset_fl,
                                                       MIN_SPE_DURATION_H, lam0_fl, max_gap)
                segs_quiet = _coalesce_segments(segs_quiet_raw, cnt_quiet, merge_gap)
                if len(cnt_saa) >= MIN_PTS_PER_CHANNEL:
                    bg_saa = compute_rolling_bg(cnt_saa, window, quiet_d, BG_UPDATE_FREQ)
                    segs_saa_raw = detect_segments_cusum(cnt_saa, bg_saa, k, h, onset_fl,
                                                         MIN_SPE_DURATION_H, lam0_fl, max_gap)
                    segs_saa = _coalesce_segments(segs_saa_raw, cnt_saa, merge_gap)
                else:
                    segs_saa_raw, segs_saa = [], []

                segs = sorted(segs_quiet + segs_saa, key=lambda s: s["onset_time"])
                base = {"k": k, "onset_floor": onset_fl, "pd_key": pd_key,
                        "side": side, "logic": logic, "channel": chan}
                for s in segs:
                    onset_rows.append({**base, **s})
                passed = [s for s in segs if s["peak_count"] >= peak_fl]
                for s in passed:
                    event_rows.append({**base, "peak_floor": peak_fl, **s})
                n_raw = len(segs_quiet_raw) + len(segs_saa_raw)
                print(f"    segs={len(segs)} (quiet={len(segs_quiet)} saa={len(segs_saa)})  "
                      f"raw(병합전)={n_raw} merge-gap-h={merge_gap}  peak>={peak_fl}: {len(passed)}")

    ONSET_COLS = ["k","onset_floor","pd_key","side","logic","channel",
                  "onset_time","peak_time","end_time","onset_count","peak_count",
                  "end_count","duration_h","bg_median","bg_sigma","threshold"]
    EVENT_COLS = ["k","onset_floor","peak_floor","pd_key","side","logic","channel",
                  "onset_time","peak_time","end_time","onset_count","peak_count",
                  "end_count","duration_h","bg_median","bg_sigma","threshold"]
    df_on = pd.DataFrame(onset_rows, columns=ONSET_COLS)
    df_ev = pd.DataFrame(event_rows, columns=EVENT_COLS)
    out_on = out_dir / f"fsm_onset_{runtag}.csv"
    out_ev = out_dir / f"fsm_event_{runtag}.csv"
    df_on.to_csv(out_on, index=False)
    df_ev.to_csv(out_ev, index=False)
    print(f"\n[fsm:{TAG}] onset saved: {out_on}  ({len(df_on)} rows)")
    print(f"[fsm:{TAG}] event saved: {out_ev}  ({len(df_ev)} rows)")
    print(f"[fsm:{TAG}] quiet_days={quiet_d} SIGMA_METHOD={SIGMA_METHOD} "
          f"WINDOW={window}d k={k} h={h} floor={onset_fl} peak={peak_fl}")
    if not df_on.empty:
        n_on = df_on.groupby("channel").size().rename("n_onset")
        n_ev = (df_ev.groupby("channel").size().rename("n_event")
                if not df_ev.empty else pd.Series(dtype=int, name="n_event"))
        tbl = pd.concat([n_on, n_ev], axis=1).fillna(0).astype(int).sort_values("n_onset", ascending=False)
        print(f"\n[fsm:{TAG}] per-channel counts:")
        print(tbl.to_string())


if __name__ == "__main__":
    main()
