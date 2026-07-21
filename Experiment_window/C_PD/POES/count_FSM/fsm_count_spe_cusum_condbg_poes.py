"""
fsm_count_spe_cusum_condbg_poes.py
====================================
Count 단독 SPE onset 탐지 — POES/MetOp SEM-2 판, 조건부 배경(condbg:
conditional background) Poisson 우도비 CUSUM (fsm_count_spe_cusum_poes.py 의
condbg 확장판. 예전 명칭 "binned"는 구현방법일 뿐이라 파일명에서 뺐다 —
정체는 "배경을 무엇에 조건화하는가"이므로 condbg로 통일).

설계 전환 배경: 시간순(단일/이중트랙) 배경은 위상혼합 때문에 quietoff는 σ
부풀림으로 둔감(POD 저하), cusum은 궤도통과당 반복 스파이크로 과검출(채널당
2만+건)이 났다. 원인은 동일 — "평시"의 정의가 자기위상에 따라 실제로 다른데
하나의 배경으로 뭉뚱그린 것. 이 파일은 그 배경 비교 기준만 조건부(bin별)로 쪼갠다.

기존 quiet/SAA 이중트랙(fsm_count_spe_cusum_poes.py)은 이 설계의 2-bin
특수형이다 — SAA가 그 자체로 bin 중 하나가 된다. 이 파일은 그걸 |maglat|
구간(--maglat-bins) N개로 일반화한다: 샘플별 bin 배정은 SAA(|B|<25000nT)
우선, 아니면 |maglat| 구간. bin마다 compute_rolling_bg/detect_segments_cusum을
독립 호출 — 이는 "전체 continuous 시계열을 순회하며 bin별 S 딕셔너리를
갱신"하는 것과 수학적으로 동일하다(각 bin 서브셋을 그 배열 순서 그대로
순회하면 S가 통과~통과 사이에도 자연히 보존되므로 — 오늘의 quiet/SAA
이중트랙이 이미 이 등가성의 증거). 원래 캐시분할 래퍼(2-2_fsm_binned_run.py,
현재는 이 파일을 등록하는 얇은 러너로 재설계됨)가 디스크에 bin별 캐시 파일을
실제로 쪼개 별도 프로세스로 돌리던 것을, 이 파일은 같은 수학을 인메모리에서
단일 프로세스로 수행한다 — 원본 캐시/시계열 자체를 쪼개지 않는다.

── condbg 변경 범위 (4단계 대조 — "λ0 도출"에만 국한, 나머지 무변경) ──
  1. λ0 도출  [변경]: 샘플의 λ0를 전역 하나가 아니라 그 샘플이 속한
     bin(위도 구간 또는 SAA)의 독립 30일 rolling median으로 구한다.
     bin별로 compute_rolling_bg를 따로 호출(전역 rolling을 사후에 자르는 게
     아니라 애초에 그 bin 서브셋만으로 계산) + detect_segments_cusum도
     bin마다 별도 호출이라 S(누적 상태)도 bin마다 완전 독립.
  2. contrib = c·ln(k) − λ0·(k−1)  [무변경]: detect_segments_cusum 함수
     자체가 원본과 byte-identical 카피라서 이 줄도 그대로.
  3. S 누적(S=max(0,S+contrib)) / h(기본 5) / min_duration(0.5h)  [무변경]:
     같은 이유로 원본과 완전 동일.
  4. (참고, 위 3단계 밖의 파라미터 차이 — 로직/수식 변경 아님) gap guard
     상한만 bin 성격에 따라 다르게 전달한다(--gap-h 위도 2h vs --gap-h-saa
     SAA 12h, SAA는 경도 구역이라 통과 간격이 더 김) — detect_segments_cusum
     내부의 "이 값 넘으면 warning 무효화" 로직 자체는 그대로, 호출 인자값만
     bin별로 다름. --merge-gap-h(세그먼트 병합)도 마찬가지로 파라미터일 뿐
     검출 코어 밖의 출력 후처리(_coalesce_segments, cusum_poes.py 원본과
     동일 기능).

λ0(bin별 30일 rolling) 창 크기 근거: 30일은 태양활동(자기권 배경 변화)은
따라가되, SEP 이벤트(보통 수 시간~수일)는 그 창 안에서 희석돼 배경 자체에
흡수되지 않는 타협적 시작값이다 — 이 창 크기 자체가 최적인지는 미검증이며,
w-sweep으로 확인하는 건 후속 과제(이 파일 초판에서는 돌리지 않음).

CSV 컬럼: cusum 원본 + maglat_bin(그 onset이 속한 bin 라벨, SAA 또는 bNN_MM).
_match_core_poes(매처)/4_summarize는 알려진 컬럼만 참조하므로 추가 컬럼은 무해.

runtag에 mlb 토큰 포함 (bin 경계가 runtag만 보고 식별되게, 파일명이 바뀌어도
TAG="cusum"은 그대로라 runtag 포맷/4_summarize 파싱은 무변경):
  cusum_mlb15-30-45-60-75_w{window}_k{k}_h{h}_on{onset}_pk{peak}

GK2A(정지궤도) 판은 없음 — maglat 비닝이 무의미해 원래 설계부터 POES 전용.
장차 GK2A에 조건부 배경(예: 경도/local-time bin)을 시도한다면 동일하게
"_condbg" 접미사로 명명할 것.

사용:
  python fsm_count_spe_cusum_condbg_poes.py --io poes_metop03_io \\
      --cache MetOp03_count/poes_metop03_cache_parquet \\
      --maglat-bins 15,30,45,60,75 --window 10 --k 10 --h 5 --onset 0.5 --peak 1
"""

from __future__ import annotations
import sys
import argparse
import importlib
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════
# 파라미터
# ══════════════════════════════════════════════════════════════════
RESAMPLE_FREQ      = "15min"
BG_UPDATE_FREQ     = "1D"
MIN_SPE_DURATION_H = 0.5

BG_WINDOW_DAYS = 30        # bin별 독립 rolling 창[일] -- 근거는 모듈 docstring 참고, w-sweep은 후속 과제
BG_QUIET_DAYS  = None
SIGMA_METHOD   = "mad"

K              = 3.0
H              = 5.0
LAMBDA0_FLOOR  = 0.01
GAP_H_LAT      = 2.0       # 위도 bin: warning 중 인접 유효샘플 gap 상한[h]
GAP_H_SAA      = 12.0      # SAA bin: 경도 구역 특성상 통과 간격이 더 김
MERGE_GAP_H    = 3.0       # 같은 bin 안 인접 세그먼트 병합 gap 상한[h]
ONSET_FLOOR    = 0.5
PEAK_FLOOR     = 2.0
MIN_PTS_PER_CHANNEL = 100
SAA_BMAG_NT = 25000.0
_DEFAULT_MAGLAT_BINS = "15,30,45,60,75"
_BIN_TOLERANCE = pd.Timedelta("2min")   # geo->count 나스트 매칭 허용오차 (1분 케이던스 기준)

TAG = "cusum"


def _import_io(io_arg: str, cache_dir: str):
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
        description=f"FSM count SPE detector POES [{TAG}-condbg] — bin별 조건부배경 Poisson 우도비 CUSUM.")
    p.add_argument("--io", required=True, help="io 모듈명/경로 (poes_metop03_io | poes_noaa19_io)")
    p.add_argument("--cache", required=True, help="POES count parquet 캐시 디렉터리")
    p.add_argument("--out", default=None, help="출력 루트 (기본: 캐시 옆 fsm2_output)")
    p.add_argument("--maglat-bins", default=_DEFAULT_MAGLAT_BINS, metavar="LIST",
                   help=f'|maglat| bin 내부 경계 콤마 리스트 (기본 "{_DEFAULT_MAGLAT_BINS}", '
                        f'암묵적으로 0과 90이 양끝에 추가됨). SAA는 별도 bin으로 항상 우선 배정.')
    p.add_argument("--window", type=int,   default=BG_WINDOW_DAYS,
                   help="bin별 독립 rolling 배경 창[일] (기본 30 -- 태양활동은 추적, SEP는 희석돼 "
                        "안 흡수되는 타협값. 최적 창 탐색은 후속 w-sweep 과제)")
    p.add_argument("--k",      type=float, default=K, help="λ1=k·λ0 (event/background 배수, k>1 필수)")
    p.add_argument("--h",      type=float, default=H, help="CUSUM 결정 임계 (기본 5)")
    p.add_argument("--onset",  type=float, default=ONSET_FLOOR)
    p.add_argument("--peak",   type=float, default=PEAK_FLOOR)
    p.add_argument("--lambda0-floor", type=float, default=LAMBDA0_FLOOR,
                   help="λ0(배경률) 하한 클립 (기본 0.01)")
    p.add_argument("--gap-h", type=float, default=GAP_H_LAT,
                   help="위도 bin: warning 중 인접 유효샘플 gap 상한[h] (기본 2)")
    p.add_argument("--gap-h-saa", type=float, default=GAP_H_SAA,
                   help="SAA bin: warning 중 인접 유효샘플 gap 상한[h] (기본 12)")
    p.add_argument("--merge-gap-h", type=float, default=MERGE_GAP_H,
                   help="출력 후처리: 같은 bin에서 next.onset_time - prev.end_time 이 이 값[h] "
                        "미만이면 인접 세그먼트를 병합 (검출 코어 무변경, 기본 3, 0=끔)")
    p.add_argument("--quiet-days", type=int, default=(BG_QUIET_DAYS if BG_QUIET_DAYS else 0))
    return p.parse_args()


def _numstr(v):
    f = float(v)
    return str(int(f)) if f.is_integer() else str(f)


def build_runtag(tag, edges, window, k, h, onset, peak):
    mlb = "-".join(_numstr(e) for e in edges)
    return f"{tag}_mlb{mlb}_w{window}_k{_numstr(k)}_h{_numstr(h)}_on{_numstr(onset)}_pk{_numstr(peak)}"


# ══════════════════════════════════════════════════════════════════
# 배경 추정 엔진 (fsm_count_spe_cusum_poes.py 원본과 byte-identical -- condbg 1단계는
# 이 함수를 bin 서브셋마다 별도 호출하는 것으로 구현됨, 함수 본문 자체는 무수정)
# ══════════════════════════════════════════════════════════════════
def _sigma(x) -> float:
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def _select_quiet_samples(cnt: pd.Series, t_ref: pd.Timestamp,
                          bg_window_days: int, bg_quiet_days) -> pd.Series:
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
# 검정부 (fsm_count_spe_cusum_poes.py 원본과 byte-identical -- condbg 2/3단계 무변경 확인용)
# ══════════════════════════════════════════════════════════════════
def detect_segments_cusum(cnt: pd.Series, bg: pd.DataFrame, k: float, h: float,
                          onset_floor: float, min_duration_h: float,
                          lambda0_floor: float, max_gap_h: float) -> list:
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
            continue
        l0 = max(l0_raw, lambda0_floor)
        if l0 <= 0:
            continue

        t = idx[i]
        if warn_start is not None and prev_valid_t is not None:
            gap_h = (t - prev_valid_t).total_seconds() / 3600
            if gap_h > max_gap_h:
                warn_start = None
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
                S = 0.0
                warn_start = None
        else:
            warn_start = None

    return segs


def _coalesce_segments(segs: list, cnt: pd.Series, merge_gap_h: float) -> list:
    """같은 bin 안 인접 세그먼트 병합. fsm_count_spe_cusum_poes.py 원본과
    byte-identical — 로직/주석은 그쪽 참고. maglat_bin 등 추가 키는 dict copy로
    그대로 보존된다."""
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
# bin 배정 (이 파일 고유, condbg 1단계 핵심) — SAA 우선, 아니면 |maglat| 구간
# ══════════════════════════════════════════════════════════════════
def _parse_maglat_bins(spec: str) -> list[float]:
    try:
        vals = sorted(set(float(v.strip()) for v in spec.split(",") if v.strip() != ""))
    except ValueError:
        raise SystemExit(f"[condbg] --maglat-bins 형식 오류: '{spec}'")
    if any(v <= 0.0 or v >= 90.0 for v in vals):
        raise SystemExit(f"[condbg] --maglat-bins 값은 0~90 사이(양끝 제외)여야 함: '{spec}'")
    return vals


def _bin_name(lo: float, hi: float) -> str:
    def _e(v):
        return f"{int(round(v)):02d}" if float(v).is_integer() else str(v)
    return f"b{_e(lo)}_{_e(hi)}"


def _bin_edges(interior_edges: list[float]) -> list[tuple[float, float]]:
    edges = [0.0] + interior_edges + [90.0]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def _assign_bins(geo: pd.DataFrame, interior_edges: list[float]) -> pd.Series:
    """geo(Bmag,maglat) 인덱스 위에서 샘플별 bin 라벨. SAA(|B|<SAA_BMAG_NT) 우선,
    아니면 |maglat| 구간(bNN_MM). geo 결측(Bmag/maglat NaN) 행은 NaN(미배정)."""
    amag = geo["maglat"].abs()
    is_saa = geo["Bmag"] < SAA_BMAG_NT
    label = pd.Series(np.nan, index=geo.index, dtype=object)
    valid = geo["Bmag"].notna() & geo["maglat"].notna()
    label[valid & is_saa] = "SAA"
    edges = _bin_edges(interior_edges)
    for i, (lo, hi) in enumerate(edges):
        is_last = (i == len(edges) - 1)
        m = valid & (~is_saa) & (amag >= lo) & ((amag <= hi) if is_last else (amag < hi))
        label[m] = _bin_name(lo, hi)
    return label


# ══════════════════════════════════════════════════════════════════
# 입출력 어댑터 (POES 전용, cusum 원본과 byte-identical)
# ══════════════════════════════════════════════════════════════════
def load_count(io, cache_dir: str) -> pd.DataFrame:
    df_count, _ = io.load(cache_dir)
    if not df_count.empty and df_count.index.tz is None:
        df_count.index = df_count.index.tz_localize("UTC")
    return df_count


def load_geo(io, cache_dir: str):
    try:
        geo = io.get_geo(cache_dir, with_bmag=True)
        if geo is None or geo.empty:
            return None
        if geo.index.tz is None:
            geo.index = geo.index.tz_localize("UTC")
        return geo
    except Exception as e:
        print(f"[fsm:{TAG}] WARN geo 로드 실패({e})")
        return None


def tag_onset_geo(onset_time, geo) -> dict:
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


def main():
    args = parse_args()
    io = _import_io(args.io, args.cache)
    interior_edges = _parse_maglat_bins(args.maglat_bins)
    window   = args.window
    k        = args.k
    h        = args.h
    onset_fl = args.onset
    peak_fl  = args.peak
    lam0_fl  = args.lambda0_floor
    gap_h_lat = args.gap_h
    gap_h_saa = args.gap_h_saa
    merge_gap = args.merge_gap_h
    quiet_d  = args.quiet_days if args.quiet_days > 0 else None
    runtag   = build_runtag(TAG, interior_edges, window, k, h, onset_fl, peak_fl)

    cache = Path(args.cache)
    out_root = Path(args.out) if args.out else (cache.parent / "fsm2_output")
    out_dir  = out_root / runtag
    out_dir.mkdir(parents=True, exist_ok=True)

    band_desc = ", ".join(f"[{lo:g},{hi:g}{']' if i == len(_bin_edges(interior_edges)) - 1 else ')'}"
                          for i, (lo, hi) in enumerate(_bin_edges(interior_edges)))
    print(f"[fsm:{TAG}-condbg] io={args.io} params: window={window}d k={k} h={h} "
          f"lambda0_floor={lam0_fl} onset={onset_fl} peak={peak_fl} quiet_days={quiet_d}")
    print(f"[fsm:{TAG}-condbg] runtag: {runtag}")
    print(f"[fsm:{TAG}-condbg] bin: SAA + 위도 {len(interior_edges) + 1}개 {band_desc}  "
          f"gap-h(위도)={gap_h_lat} gap-h(SAA)={gap_h_saa} merge-gap-h={merge_gap}")
    print(f"[fsm:{TAG}-condbg] Loading count data...")
    df_count = load_count(io, args.cache)
    if df_count.empty:
        print("[fsm] 빈 캐시"); return
    cnt_rs = df_count.resample(RESAMPLE_FREQ).mean()

    geo = load_geo(io, args.cache)
    if geo is None:
        raise SystemExit(f"[fsm:{TAG}-condbg] ERROR geo 로드 실패 -- bin 배정 불가(SAA/maglat 필요)")
    bin_label_geo = _assign_bins(geo, interior_edges)
    all_labels = ["SAA"] + [_bin_name(lo, hi) for lo, hi in _bin_edges(interior_edges)]

    onset_rows, event_rows = [], []
    for col in df_count.columns:
        species, direction, energy = col
        try:
            cnt = cnt_rs[col].dropna()
        except KeyError:
            continue
        if len(cnt) < MIN_PTS_PER_CHANNEL:
            continue
        chan = io.tuple_to_fname(tuple(col))
        print(f"\n[fsm:{TAG}-condbg] {chan}  (n={len(cnt)} pts)")

        bin_of = bin_label_geo.reindex(cnt.index, method="nearest", tolerance=_BIN_TOLERANCE)

        segs_all = []
        n_raw_total = 0
        per_bin_report = []
        for label in all_labels:
            cnt_b = cnt[bin_of == label]
            if len(cnt_b) < MIN_PTS_PER_CHANNEL:
                continue
            gap_h = gap_h_saa if label == "SAA" else gap_h_lat
            bg_b = compute_rolling_bg(cnt_b, window, quiet_d, BG_UPDATE_FREQ)
            segs_b_raw = detect_segments_cusum(cnt_b, bg_b, k, h, onset_fl,
                                               MIN_SPE_DURATION_H, lam0_fl, gap_h)
            for s in segs_b_raw:
                s["maglat_bin"] = label
            segs_b = _coalesce_segments(segs_b_raw, cnt_b, merge_gap)
            n_raw_total += len(segs_b_raw)
            segs_all.extend(segs_b)
            if segs_b_raw:
                per_bin_report.append(f"{label}:{len(segs_b_raw)}->{len(segs_b)}")

        segs = sorted(segs_all, key=lambda s: s["onset_time"])
        base = {"k": k, "onset_floor": onset_fl, "species": species,
                "direction": direction, "energy": energy, "channel": chan}
        for s in segs:
            geo_tag = tag_onset_geo(s["onset_time"], geo)
            onset_rows.append({**base, **s, **geo_tag})
        passed = [s for s in segs if s["peak_count"] >= peak_fl]
        for s in passed:
            geo_tag = tag_onset_geo(s["onset_time"], geo)
            event_rows.append({**base, "peak_floor": peak_fl, **s, **geo_tag})
        n_raw = n_raw_total
        print(f"    segs={len(segs)}  raw(병합전)={n_raw}  "
              f"bin별(raw->병합): {', '.join(per_bin_report) if per_bin_report else '(검출 없음)'}  "
              f"peak>={peak_fl}: {len(passed)}")

    GEO_COLS = ["in_saa", "onset_Bmag", "onset_maglat"]
    ONSET_COLS = ["k","onset_floor","species","direction","energy","channel","maglat_bin",
                  "onset_time","peak_time","end_time","onset_count","peak_count",
                  "end_count","duration_h","bg_median","bg_sigma","threshold"] + GEO_COLS
    EVENT_COLS = ["k","onset_floor","peak_floor","species","direction","energy","channel","maglat_bin",
                  "onset_time","peak_time","end_time","onset_count","peak_count",
                  "end_count","duration_h","bg_median","bg_sigma","threshold"] + GEO_COLS
    df_on = pd.DataFrame(onset_rows, columns=ONSET_COLS)
    df_ev = pd.DataFrame(event_rows, columns=EVENT_COLS)
    out_on = out_dir / f"fsm_onset_{runtag}.csv"
    out_ev = out_dir / f"fsm_event_{runtag}.csv"
    df_on.to_csv(out_on, index=False)
    df_ev.to_csv(out_ev, index=False)
    print(f"\n[fsm:{TAG}-condbg] onset saved: {out_on}  ({len(df_on)} rows)")
    print(f"[fsm:{TAG}-condbg] event saved: {out_ev}  ({len(df_ev)} rows)")
    print(f"[fsm:{TAG}-condbg] quiet_days={quiet_d} SIGMA_METHOD={SIGMA_METHOD} "
          f"WINDOW={window}d k={k} h={h} floor={onset_fl} peak={peak_fl}")
    if not df_on.empty:
        n_on = df_on.groupby("channel").size().rename("n_onset")
        n_ev = (df_ev.groupby("channel").size().rename("n_event")
                if not df_ev.empty else pd.Series(dtype=int, name="n_event"))
        tbl = pd.concat([n_on, n_ev], axis=1).fillna(0)
        tbl = tbl.sort_values("n_onset", ascending=False)
        print(f"\n[fsm:{TAG}-condbg] per-channel counts:")
        print(tbl.to_string())
        print(f"\n[fsm:{TAG}-condbg] bin별 onset 분포:")
        print(df_on["maglat_bin"].value_counts().to_string())


if __name__ == "__main__":
    main()
