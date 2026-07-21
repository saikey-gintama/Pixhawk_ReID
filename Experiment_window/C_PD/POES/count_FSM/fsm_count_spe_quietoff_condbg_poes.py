"""
fsm_count_spe_quietoff_condbg_poes.py
========================================
Count 단독 SPE onset 탐지 — POES/MetOp SEM-2 판, 조건부 배경(condbg:
conditional background) quietoff+MAD (fsm_count_spe_quietoff_mad_poes.py 의
condbg 확장판. 예전 명칭 "binned"는 구현방법일 뿐이라 파일명에서 뺐다 —
정체는 "배경을 무엇에 조건화하는가"이므로 cusum 판과 함께 condbg로 통일).

설계 전환 배경: fsm_count_spe_cusum_condbg_poes.py 참고 — 시간순 단일배경은
위상혼합(궤도마다 반복되는 위도/SAA 변조) 때문에 quietoff 에서는 σ가
부풀려져 문턱이 과도하게 높아지고(둔감, POD 저하), cusum 에서는 반대로
매 통과마다 스파이크로 과검출을 낳는다. 이 파일은 quietoff 쪽 수정 —
threshold = bin_median(t) + k·bin_MAD(t) 로, 그 순간 위성이 속한 bin
(SAA 우선, 아니면 |maglat| 구간)의 배경만 비교 기준으로 쓴다.

quietoff 는 cusum 과 달리 누적 상태(S)가 없는 순간 문턱 비교이므로 "bin별
독립 상태"라는 개념 자체가 필요 없다 — bin마다 compute_rolling_bg 를 그
bin 서브셋에 돌려 얻은 (bg_median, bg_std) 를 원래 채널의 **연속(전체)**
시계열 인덱스 위에 bin 소속에 따라 조립(_binned_background)하기만 하면,
기존 build_threshold/detect_segments(원본과 byte-identical, 무수정)를 그대로
재사용할 수 있다. 즉 이 파일의 "새 코드"는 bin 배정 + 배경 조립뿐이고, 문턱
계산과 연속구간 탐지 로직은 원본 그대로다 — cusum_condbg 판의 4단계 대조로
치면 "λ0(여기서는 bg_median/bg_std) 도출만 조건부, 나머지(문턱 비교식·연속
구간 판정)는 무변경"인 것과 완전히 같은 구조.

λ0(=bg_median, bin별 30일 rolling) 창 크기 근거: 30일은 태양활동(자기권
배경 변화)은 따라가되, SEP 이벤트(보통 수 시간~수일)는 그 창 안에서 희석돼
배경 자체에 흡수되지 않는 타협적 시작값이다 — 최적 창인지는 미검증이며,
w-sweep으로 확인하는 건 후속 과제(이 파일 초판에서는 돌리지 않음).

시계열을 물리적으로 쪼개지 않고(원본 캐시/시계열 무변경) 연속 인덱스를
유지하는 덕에, 캐시분할 래퍼(2-2_fsm_binned_run.py, 현재는 이 파일을
등록하는 얇은 러너로 재설계됨)에서 발견됐던 "가짜 duration"(bin 재방문
사이 공백이 지속시간으로 합산되는 버그, 최대 813h) 이 근본적으로 재발하지
않는다 — detect_segments 가 보는 시계열이 처음부터 빽빽한 원본 15분
케이던스 그대로이기 때문(그 버그는 디스크에서 bin별로 쪼갠 sparse 시계열을
순회할 때만 발생했다).

CSV 컬럼: quietoff 원본 + maglat_bin(그 onset 시점이 속한 bin 라벨,
SAA 또는 bNN_MM, 사후 조회).

runtag에 mlb 토큰(TAG="quietoff_mad"는 그대로라 runtag 포맷/4_summarize
파싱은 무변경): quietoff_mad_mlb15-30-45-60-75_w{window}_k{k}_on{onset}_pk{peak}

GK2A(정지궤도) 판은 없음 — maglat 비닝이 무의미해 원래 설계부터 POES 전용.

사용:
  python fsm_count_spe_quietoff_condbg_poes.py --io poes_metop03_io \\
      --cache MetOp03_count/poes_metop03_cache_parquet \\
      --maglat-bins 15,30,45,60,75 --window 10 --k 5 --onset 0.5 --peak 2
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
MIN_SPE_DURATION_H = 1

BG_WINDOW_DAYS = 30        # bin별 독립 rolling 창[일] -- 근거는 모듈 docstring 참고, w-sweep은 후속 과제
BG_QUIET_DAYS  = None
SIGMA_METHOD   = "mad"

K           = 10
ONSET_FLOOR = 0.5
PEAK_FLOOR  = 2.0
MIN_PTS_PER_CHANNEL = 100
SAA_BMAG_NT = 25000.0
_DEFAULT_MAGLAT_BINS = "15,30,45,60,75"
_BIN_TOLERANCE = pd.Timedelta("2min")

TAG = "quietoff_mad"


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
        description=f"FSM count SPE detector POES [{TAG}-condbg] — bin별 조건부배경 quietoff+MAD.")
    p.add_argument("--io", required=True, help="io 모듈명/경로 (poes_metop03_io | poes_noaa19_io)")
    p.add_argument("--cache", required=True, help="POES count parquet 캐시 디렉터리")
    p.add_argument("--out", default=None, help="출력 루트 (기본: 캐시 옆 fsm2_output)")
    p.add_argument("--maglat-bins", default=_DEFAULT_MAGLAT_BINS, metavar="LIST",
                   help=f'|maglat| bin 내부 경계 콤마 리스트 (기본 "{_DEFAULT_MAGLAT_BINS}", '
                        f'암묵적으로 0과 90이 양끝에 추가됨). SAA는 별도 bin으로 항상 우선 배정.')
    p.add_argument("--window", type=int,   default=BG_WINDOW_DAYS,
                   help="bin별 독립 rolling 배경 창[일] (기본 30 -- 태양활동은 추적, SEP는 희석돼 "
                        "안 흡수되는 타협값. 최적 창 탐색은 후속 w-sweep 과제)")
    p.add_argument("--k",      type=float, default=K)
    p.add_argument("--onset",  type=float, default=ONSET_FLOOR)
    p.add_argument("--peak",   type=float, default=PEAK_FLOOR)
    p.add_argument("--quiet-days", type=int, default=(BG_QUIET_DAYS if BG_QUIET_DAYS else 0))
    return p.parse_args()


def _numstr(v):
    f = float(v)
    return str(int(f)) if f.is_integer() else str(f)


def build_runtag(tag, edges, window, k, onset, peak):
    mlb = "-".join(_numstr(e) for e in edges)
    return f"{tag}_mlb{mlb}_w{window}_k{_numstr(k)}_on{_numstr(onset)}_pk{_numstr(peak)}"


# ══════════════════════════════════════════════════════════════════
# 배경 추정 + 검출 엔진 (quietoff_mad_poes.py 원본과 byte-identical — 무수정)
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


def build_threshold(bg: pd.DataFrame, k: float, onset_floor: float) -> pd.Series:
    th = bg["bg_median"] + k * bg["bg_std"]
    if onset_floor > 0:
        th = th.clip(lower=onset_floor)
    return th


def detect_segments(cnt: pd.Series, thresh_series: pd.Series,
                    bg: pd.DataFrame, min_duration_h: float) -> list:
    th  = thresh_series.reindex(cnt.index).ffill()
    med = bg["bg_median"].reindex(cnt.index).ffill()
    sig = bg["bg_std"].reindex(cnt.index).ffill()
    above = (cnt >= th) & cnt.notna() & th.notna()
    segs = []; in_ev = False; onset = None
    def _close(seg, onset_t, end_t):
        if (end_t - onset_t).total_seconds() / 3600 < min_duration_h:
            return
        segs.append({
            "onset_time": onset_t, "peak_time": seg.idxmax(), "end_time": end_t,
            "onset_count": round(float(cnt.loc[onset_t]), 3),
            "peak_count":  round(float(seg.max()), 3),
            "end_count":   round(float(cnt.loc[end_t]), 3),
            "duration_h":  round((end_t - onset_t).total_seconds() / 3600, 2),
            "bg_median": round(float(med.loc[onset_t]), 4) if pd.notna(med.loc[onset_t]) else np.nan,
            "bg_sigma":  round(float(sig.loc[onset_t]), 4) if pd.notna(sig.loc[onset_t]) else np.nan,
            "threshold": round(float(th.loc[onset_t]), 4),
        })
    for t, a in above.items():
        if a and not in_ev:
            in_ev, onset = True, t
        elif not a and in_ev:
            _close(cnt.loc[onset:t], onset, t); in_ev = False
    if in_ev and onset is not None:
        seg = cnt.loc[onset:]; _close(seg, onset, seg.index[-1])
    return segs


# ══════════════════════════════════════════════════════════════════
# bin 배정 + 배경 조립 (이 파일 고유) — SAA 우선, 아니면 |maglat| 구간
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
    아니면 |maglat| 구간(bNN_MM). geo 결측 행은 NaN(미배정)."""
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


def _binned_background(cnt: pd.Series, bin_of: pd.Series, all_labels: list[str],
                       window: int, quiet_d, update_freq: str) -> pd.DataFrame:
    """레이블별 부분시계열에 compute_rolling_bg(bin 서브셋 전용)를 적용한 뒤
    원래 cnt의 **연속(전체)** 인덱스 위에 bin 소속에 따라 재조립.
    반환은 build_threshold/detect_segments가 그대로 받는 표준 bg DataFrame
    (index=cnt.index, columns=[bg_median, bg_std]) — 시계열은 쪼개지 않는다."""
    bg_full = pd.DataFrame({"bg_median": np.nan, "bg_std": np.nan}, index=cnt.index)
    for label in all_labels:
        mask = (bin_of == label)
        sub = cnt[mask]
        if sub.empty:
            continue
        bg_sub = compute_rolling_bg(sub, window, quiet_d, update_freq)
        bg_full.loc[sub.index, "bg_median"] = bg_sub["bg_median"]
        bg_full.loc[sub.index, "bg_std"]    = bg_sub["bg_std"]
    return bg_full


# ══════════════════════════════════════════════════════════════════
# 입출력 어댑터 (POES 전용, 원본과 byte-identical)
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


def _tag_onset_bin(onset_time, bin_of: pd.Series):
    try:
        pos = bin_of.index.get_indexer([onset_time], method="nearest")[0]
    except Exception:
        return np.nan
    if pos < 0:
        return np.nan
    v = bin_of.iloc[pos]
    return v if isinstance(v, str) else np.nan


def main():
    args = parse_args()
    io = _import_io(args.io, args.cache)
    interior_edges = _parse_maglat_bins(args.maglat_bins)
    window   = args.window
    k        = args.k
    onset_fl = args.onset
    peak_fl  = args.peak
    quiet_d  = args.quiet_days if args.quiet_days > 0 else None
    runtag   = build_runtag(TAG, interior_edges, window, k, onset_fl, peak_fl)

    cache = Path(args.cache)
    out_root = Path(args.out) if args.out else (cache.parent / "fsm2_output")
    out_dir  = out_root / runtag
    out_dir.mkdir(parents=True, exist_ok=True)

    band_desc = ", ".join(f"[{lo:g},{hi:g}{']' if i == len(_bin_edges(interior_edges)) - 1 else ')'}"
                          for i, (lo, hi) in enumerate(_bin_edges(interior_edges)))
    print(f"[fsm:{TAG}-condbg] io={args.io} params: window={window}d k={k} onset={onset_fl} "
          f"peak={peak_fl} quiet_days={quiet_d}")
    print(f"[fsm:{TAG}-condbg] runtag: {runtag}")
    print(f"[fsm:{TAG}-condbg] bin: SAA + 위도 {len(interior_edges) + 1}개 {band_desc}")
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
        bg   = _binned_background(cnt, bin_of, all_labels, window, quiet_d, BG_UPDATE_FREQ)
        thr  = build_threshold(bg, k, onset_fl)
        segs = detect_segments(cnt, thr, bg, MIN_SPE_DURATION_H)
        for s in segs:
            s["maglat_bin"] = _tag_onset_bin(s["onset_time"], bin_of)

        base = {"k": k, "onset_floor": onset_fl, "species": species,
                "direction": direction, "energy": energy, "channel": chan}
        for s in segs:
            geo_tag = tag_onset_geo(s["onset_time"], geo)
            onset_rows.append({**base, **s, **geo_tag})
        passed = [s for s in segs if s["peak_count"] >= peak_fl]
        for s in passed:
            geo_tag = tag_onset_geo(s["onset_time"], geo)
            event_rows.append({**base, "peak_floor": peak_fl, **s, **geo_tag})
        print(f"    segs={len(segs)}  peak>={peak_fl}: {len(passed)}")

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
          f"WINDOW={window}d K={k} floor={onset_fl} peak={peak_fl}")
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
