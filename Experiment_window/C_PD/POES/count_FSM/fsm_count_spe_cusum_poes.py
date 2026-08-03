"""
fsm_count_spe_cusum_poes.py
===========================
Count 단독 SPE onset 탐지 — POES/MetOp SEM-2 판 (Poisson 우도비 CUSUM baseline).

3회 실패한 이전 버전 전부 폐기: (1) z-표준화 상수-k, (2) Anscombe변환+식3 시변-k,
(3) 순수 Poisson우도비 단일배경. (1)(2)는 (cnt-μ)/σ z-표준화가 저카운트에서
비대칭 발산을 일으켰고, (3)은 z-표준화를 없앴는데도 여전히 발산 — 원인을
추적한 결과 SAA 스파이크가 non-SAA 배경(λ0)에 섞여 들어가 "이벤트"로 오인되고,
그 스파이크가 매 궤도(~101분)마다 반복돼 평시 감쇠보다 항상 크게 누적됐다
(omni_p7 발산 누적의 85%, pro_tel0_p5 69%가 SAA 통과 구간에서 발생).

이 버전(4번째)은 Poisson 로그우도비 누적(z-표준화 없음)은 그대로 두고,
**SAA/non-SAA 위상을 배경·누적 단계에서부터 분리**한다 — SAA를 버리는 게
아니라 SAA 전용 트랙을 하나 더 두어 "SAA 안에서도 평소보다 튀면 검출"하게
한다. 각 트랙 공통 수식:
  λ0[i] = 그 트랙만의 배경률(robust rolling median, floor 클립)
  λ1[i] = k · λ0[i]                         (k = "이벤트로 볼 배수")
  S[0]  = 0
  S[i]  = max(0, S[i-1] + cnt[i]·ln(λ1[i]/λ0[i]) - (λ1[i]-λ0[i]))
        = max(0, S[i-1] + cnt[i]·ln(k) - λ0[i]·(k-1))   (ln(λ1/λ0)=ln(k) 상수)
검출: S>h 가 min_duration_h 이상 지속 → 그 직전 S가 마지막으로 0이었던 지점까지
backtrack해 onset. 두 트랙 배경이 서로 오염되지 않으므로, SAA 스파이크는
SAA배경 기준으로는 "평시"가 되어 non-SAA 트랙의 감쇠를 방해하지 않는다.

fsm_count_spe_quietoff_mad_poes.py 와 골격(입출력 어댑터·채널 루프·CSV 스키마)
동일. 배경 엔진(_sigma/_select_quiet_samples/compute_rolling_bg)과 입출력
어댑터(load_count/load_geo/tag_onset_geo), _import_io/_numstr 골격은 원본과
byte-identical 카피(재구현 아님) — 검정부만 detect_segments_cusum으로 교체.
bg_std(σ)는 CUSUM 계산에 전혀 쓰지 않는다 — CSV bg_sigma 컬럼 호환 기록용.

CSV 컬럼 스키마(ONSET_COLS/EVENT_COLS/GEO_COLS)는 quietoff와 100% 동일
— _match_core_poes(매처)/4_summarize가 그대로 읽는다. threshold 컬럼은
λ1[onset]=k·λ0[onset](호환용), bg_median은 λ0[onset](floor 클립 적용된 실사용값).

세그먼트 병합(coalescing, --merge-gap-h 기본 3h): 즉시강제리셋(발산 방지 핵심)의
대가로 실제로는 하나로 이어진 지속 이벤트가 리셋 직후 재상승마다 조각(fragment)
나는 문제를, 검출 코어는 그대로 두고 출력 직전 후처리로 복원한다 —
_coalesce_segments 참고. 같은 트랙(quiet/SAA) 안에서만 병합.

사용:
  python fsm_count_spe_cusum_poes.py --io poes_metop03_io \
      --cache MetOp03_count/poes_metop03_cache_parquet \
      --window 10 --k 3 --h 5 --onset 0.5 --peak 2
  python fsm_count_spe_cusum_poes.py --io poes_noaa19_io \
      --cache NOAA19_count/poes_noaa19_cache_parquet --window 7 --k 3 --h 5 --onset 0.5 --peak 2
"""

from __future__ import annotations
import sys
import argparse
import importlib
import math
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd

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

K              = 3.0       # λ1 = k·λ0 (event/background 배수)
H              = 5.0       # CUSUM 결정 임계
LAMBDA0_FLOOR  = 0.01      # λ0 하한 클립 (15min 리샘플 count 스케일)
MAX_GAP_H      = 2.0       # warning 진행 중 인접 유효샘플 gap 상한 -> 넘으면 warning 무효화
MERGE_GAP_H    = 3.0       # 같은 트랙 인접 세그먼트 병합 gap 상한[h] (검출 코어 무변경, 출력 후처리 전용)
ONSET_FLOOR    = 0.5
PEAK_FLOOR     = 2.0
MIN_PTS_PER_CHANNEL = 100
SAA_BMAG_NT = 25000.0      # |B| < 이 값 → in_saa (coords_igrf 와 동일 기준)

TAG = "cusum"              # 출력 파일명 태그


def _import_io(io_arg: str, cache_dir: str):
    """io 모듈명/경로로 임포트. 캐시 경로 부모들을 sys.path 에 추가."""
    name = PureWindowsPath(io_arg).name if (io_arg and ('/' in io_arg or '\\' in io_arg)) else io_arg
    cache = Path(cache_dir).resolve()
    for cand in (cache if cache.is_dir() else cache.parent,
                 cache.parent, cache.parent.parent,
                 Path.cwd(), Path(__file__).resolve().parent):
        sp = str(cand)
        if sp not in sys.path:
            sys.path.insert(0, sp)
    return importlib.import_module(name)


def parse_args():
    p = argparse.ArgumentParser(
        description=f"FSM count SPE detector POES [{TAG}] — Poisson 우도비 CUSUM.")
    p.add_argument("--io", required=True, help="io 모듈명/경로 (poes_metop03_io | poes_noaa19_io)")
    p.add_argument("--cache", required=True, help="POES count parquet 캐시 디렉터리")
    p.add_argument("--out", default=None, help="출력 루트 (기본: 캐시 옆 fsm2_output)")
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
    p.add_argument("--no-geo", action="store_true",
                   help="geo 태깅 생략(in_saa 없이). geo 캐시 없을 때.")
    return p.parse_args()


def _numstr(v):
    f = float(v)
    return str(int(f)) if f.is_integer() else str(f)


def build_runtag(tag, window, k, h, onset, peak):
    return f"{tag}_w{window}_k{_numstr(k)}_h{_numstr(h)}_on{_numstr(onset)}_pk{_numstr(peak)}"


# ══════════════════════════════════════════════════════════════════
# 배경 추정 엔진 (quietoff_mad_poes.py 원본과 byte-identical — 절대 수정 금지)
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
# 검정부 (이 baseline 고유) — Poisson 우도비 CUSUM
# ══════════════════════════════════════════════════════════════════
def detect_segments_cusum(cnt: pd.Series, bg: pd.DataFrame, k: float, h: float,
                          onset_floor: float, min_duration_h: float,
                          lambda0_floor: float, max_gap_h: float = MAX_GAP_H) -> list:
    """Poisson 로그우도비 CUSUM. z-표준화(σ 나눗셈) 없음 — count를 우도비에 직접 투입.

    λ0=배경률(robust median, floor 클립), λ1=k·λ0.
    S[i] = max(0, S[i-1] + cnt[i]*ln(k) - λ0[i]*(k-1))

    핵심(4번째 실패 이후 수정): onset이 min_duration만큼 확정되는 즉시 S=0으로
    강제 리셋하고 warning 상태도 지운다 — "S가 자연적으로 h 밑에 복귀할 때까지"
    기다리지 않는다. 이전 버전은 단발 대형 스파이크(기여 수백~수천) 이후 S가
    h 아래로 자연 복귀하는 데 수년이 걸려 전체 구간이 onset 1개로 붕괴했다.
    이 버전은 onset을 확정한 그 시점에서 end로 끊고 바로 S=0에서 재출발하므로,
    스파이크 뒤 꼬리가 배경 대비 여전히 높으면 새 onset으로 다시 잡히고(정상,
    실제 지속 이벤트), 낮으면 S가 안 올라간다(정상 감쇠) — 꼬리를 하나로
    삼키는 문제가 구조적으로 재발하지 않는다.

    지속 판정은 스텝(행) 개수가 아니라 warning 시작~현재 timestamp의 실제
    경과시간(idx[i]-idx[warn_start] >= min_duration_h)으로 한다. 이중트랙에서
    quiet/SAA 트랙은 서로의 위상 구간이 빠져 시계열이 듬성듬성하므로, 스텝
    개수로 세면 "연속 12스텝"이 실제로는 몇 주에 걸쳐 흩어져 있을 수 있어
    비현실적으로 긴 duration_h(예: 809h)를 만드는 버그가 있었다.
    추가로, warning 진행 중 인접 두 유효 샘플의 실제 시간 gap이 max_gap_h를
    넘으면(트랙이 그 구간에서 너무 듬성해진 것) 그 warning을 무효화한다
    (warn_start=None, S는 유지) — "연속 지속"으로 잘못 이어붙이는 걸 막는다.
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

    detect_segments_cusum은 onset이 min_duration 확정되는 즉시 S=0으로 강제
    리셋한다(발산 방지 핵심). 그 대가로 실제로는 하나로 이어진 지속 이벤트가
    "리셋 직후 재상승"할 때마다 여러 조각(fragment)으로 쪼개져 나온다. 이 함수는
    검출 로직을 건드리지 않고, 인접한 두 세그먼트의 next.onset_time -
    prev.end_time 이 merge_gap_h[h] 미만이면 하나로 합쳐 조각을 이벤트급
    duration으로 복원한다.

    onset은 첫 세그먼트 값을 그대로 쓰고(진짜 onset 시점), end은 마지막 세그먼트
    값, peak_time/peak_count는 병합된 [onset_time, end_time] 구간 전체에서
    재계산(중간에 원래 조각 경계 밖이던 표본도 포함해야 진짜 peak를 못 놓친다),
    bg_median/bg_sigma/threshold은 첫 세그먼트 값을 유지한다(그 배경에서 검출된
    onset이라는 사실은 병합해도 안 변함).
    """
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


# ══════════════════════════════════════════════════════════════════
# 입출력 어댑터 (POES 전용, quietoff_mad_poes.py 원본과 byte-identical)
# ══════════════════════════════════════════════════════════════════
def load_count(io, cache_dir: str) -> pd.DataFrame:
    """POES count 캐시 로드 → MultiIndex DataFrame (UTC)."""
    df_count, _ = io.load(cache_dir)
    if not df_count.empty and df_count.index.tz is None:
        df_count.index = df_count.index.tz_localize("UTC")
    return df_count


def load_geo(io, cache_dir: str):
    """geo(lat/lon/alt + Bmag/maglat) 로드. 실패 시 None."""
    try:
        geo = io.get_geo(cache_dir, with_bmag=True)
        if geo is None or geo.empty:
            return None
        if geo.index.tz is None:
            geo.index = geo.index.tz_localize("UTC")
        return geo
    except Exception as e:
        print(f"[fsm:{TAG}] WARN geo 로드 실패({e}) → in_saa 태깅 생략")
        return None


def tag_onset_geo(onset_time, geo) -> dict:
    """onset 시점의 geo 조회 → in_saa/onset_Bmag/onset_maglat.
    검출은 전 구간에서 이미 끝났고, 여기선 '그 시점이 SAA였나'만 사후 태깅."""
    if geo is None:
        return {"in_saa": np.nan, "onset_Bmag": np.nan, "onset_maglat": np.nan}
    try:
        pos = geo.index.get_indexer([onset_time], method="nearest")[0]
    except Exception:
        return {"in_saa": np.nan, "onset_Bmag": np.nan, "onset_maglat": np.nan}
    if pos < 0:
        return {"in_saa": np.nan, "onset_Bmag": np.nan, "onset_maglat": np.nan}
    row = geo.iloc[pos]
    bmag = float(row["Bmag"]) if "Bmag" in geo.columns and pd.notna(row.get("Bmag")) else np.nan
    mlat = float(row["maglat"]) if "maglat" in geo.columns and pd.notna(row.get("maglat")) else np.nan
    in_saa = bool(bmag < SAA_BMAG_NT) if np.isfinite(bmag) else np.nan
    return {"in_saa": in_saa,
            "onset_Bmag": round(bmag, 1) if np.isfinite(bmag) else np.nan,
            "onset_maglat": round(mlat, 1) if np.isfinite(mlat) else np.nan}


def _saa_bit_series(cnt: pd.Series, geo) -> pd.Series:
    """geo(Bmag)를 cnt 인덱스에 nearest 매칭 -> bool SAA 시리즈. geo=None이면 전부 False."""
    if geo is None or "Bmag" not in geo.columns:
        return pd.Series(False, index=cnt.index)
    pos = geo.index.get_indexer(cnt.index, method="nearest")
    bmag = geo["Bmag"].to_numpy()[pos]
    return pd.Series(bmag < SAA_BMAG_NT, index=cnt.index).fillna(False)


def main():
    args = parse_args()
    io = _import_io(args.io, args.cache)
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

    cache = Path(args.cache)
    out_root = Path(args.out) if args.out else (cache.parent / "fsm2_output")
    out_dir  = out_root / runtag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fsm:{TAG}] io={args.io} params: window={window}d k={k} h={h} "
          f"lambda0_floor={lam0_fl} onset={onset_fl} peak={peak_fl} quiet_days={quiet_d}")
    print(f"[fsm:{TAG}] runtag: {runtag}")
    print(f"[fsm:{TAG}] Loading count data...")
    df_count = load_count(io, args.cache)
    if df_count.empty:
        print("[fsm] 빈 캐시"); return
    cnt_rs = df_count.resample(RESAMPLE_FREQ).mean()

    geo = None if args.no_geo else load_geo(io, args.cache)
    if geo is not None:
        print(f"[fsm:{TAG}] geo 태깅 ON (Bmag/maglat, SAA<|B|{SAA_BMAG_NT:.0f}nT)")
        print(f"[fsm:{TAG}] SAA/non-SAA 이중배경 CUSUM: 각 위상 자기 배경으로 독립 누적")
    else:
        print(f"[fsm:{TAG}] WARNING geo 없음 -> 전체를 non-SAA 단일트랙으로 처리. "
              f"SAA 스파이크가 배경을 오염시켜 발산할 위험이 큼 — geo 캐시 사용을 강권함.")

    onset_rows, event_rows = [], []
    for col in df_count.columns:
        species, direction, energy = col
        try:
            cnt = cnt_rs[col].dropna()
        except KeyError:
            continue
        if len(cnt) < MIN_PTS_PER_CHANNEL:
            continue
        chan = io.tuple_to_fname(tuple(col))   # 'pro_tel0_p5'
        print(f"\n[fsm:{TAG}] {chan}  (n={len(cnt)} pts)")

        # SAA/non-SAA 위상 분리 -> 각자 자기 배경으로 독립 누적 (배경 상호오염 차단)
        saa_bit    = _saa_bit_series(cnt, geo)
        cnt_quiet  = cnt[~saa_bit]
        cnt_saa    = cnt[saa_bit]

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
        base = {"k": k, "onset_floor": onset_fl, "species": species,
                "direction": direction, "energy": energy, "channel": chan}
        for s in segs:
            geo_tag = tag_onset_geo(s["onset_time"], geo)
            onset_rows.append({**base, **s, **geo_tag})
        passed = [s for s in segs if s["peak_count"] >= peak_fl]
        for s in passed:
            geo_tag = tag_onset_geo(s["onset_time"], geo)
            event_rows.append({**base, "peak_floor": peak_fl, **s, **geo_tag})
        n_raw = len(segs_quiet_raw) + len(segs_saa_raw)
        print(f"    segs={len(segs)} (quiet={len(segs_quiet)} saa={len(segs_saa)})  "
              f"raw(병합전)={n_raw} merge-gap-h={merge_gap}  peak>={peak_fl}: {len(passed)}")

    # 공통 컬럼(quietoff와 동일) + POES 전용 geo 태깅 컬럼 末尾
    GEO_COLS = ["in_saa", "onset_Bmag", "onset_maglat"]
    ONSET_COLS = ["k","onset_floor","species","direction","energy","channel",
                  "onset_time","peak_time","end_time","onset_count","peak_count",
                  "end_count","duration_h","bg_median","bg_sigma","threshold"] + GEO_COLS
    EVENT_COLS = ["k","onset_floor","peak_floor","species","direction","energy","channel",
                  "onset_time","peak_time","end_time","onset_count","peak_count",
                  "end_count","duration_h","bg_median","bg_sigma","threshold"] + GEO_COLS
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
        if "in_saa" in df_on.columns and df_on["in_saa"].notna().any():
            saa_frac = df_on.groupby("channel")["in_saa"].mean().rename("saa_frac").round(2)
            tbl = pd.concat([n_on, n_ev, saa_frac], axis=1).fillna(0)
        else:
            tbl = pd.concat([n_on, n_ev], axis=1).fillna(0)
        tbl = tbl.sort_values("n_onset", ascending=False)
        print(f"\n[fsm:{TAG}] per-channel counts (saa_frac=onset 중 SAA 비율):")
        print(tbl.to_string())


if __name__ == "__main__":
    main()
