"""
fsm_count_spe_blc1_lowe.py
==========================
BL-Löwe: Löwe et al. (2025) nowcasting 방식을 KSEM count에 그대로 이식한
baseline. 본 연구 제안 방식(rolling MAD-robust FSM)의 직접 선행연구 대조군.

Löwe 원식 (MSL/RAD dose rate E 기준):
  - Background (B): 직전 5일 데이터에 constant linear fit (y = a).
                    0차 다항 피팅이므로 해석해는 표본 평균과 동일하다.
                    (논문 p4 "constant linear fit", p9 "average background")
  - Threshold (T): T = 1.25 · B  (배경 25% 초과; 논문 p5)
  - 배경 갱신: 매일 1회 (논문 p9)

KSEM 이식 시 차이:
  - 입력이 dose rate가 아니라 coincidence-logic count.
  - σ 항이 없다. Löwe는 median+k·σ 구조가 아니라 순수 비율(×1.25)이다.
    따라서 K / MAD / std 개념이 존재하지 않는다.
  - 평가 틀은 다른 4종 FSM과 동일한 offline batch (전체 기록에 rolling 배경을
    미리 깔고 검출). Löwe 원문의 "SEP 중 배경 pause"는 인과적 실시간 루프
    전용 장치이므로 offline 비교에서는 적용하지 않는다.

검출 엔진 뒷단(15분 리샘플, 1시간 지속 규칙, peak_floor 사후필터, 출력 스키마)은
quietoff_mad 등 4종과 byte 단위로 동일하다. 배경 추정부만 교체했다:
  compute_rolling_bg : 5일 윈도우 0차 polyfit(=mean)으로 B 산출
  build_threshold    : T = LOWE_MULT · B   (1.25배)

출력:
  fsm_onset_blc1_lowe_*.csv / fsm_event_blc1_lowe_*.csv
  스키마는 ana6_sweep_event.csv 규격(noaa_goes_spe_match 호환). σ가 없으므로
  bg_sigma 컬럼은 NaN, k는 placeholder(0)로 채운다.

사용:
  python fsm_count_spe_blc1_lowe.py                  # 5일 윈도우, 1.25배 (논문 그대로)
  python fsm_count_spe_blc1_lowe.py --window 5 --mult 1.25
  python fsm_count_spe_blc1_lowe.py --window 30      # 윈도우 민감도용

자립성: ksem_flux_config / ksem_common 비의존. count 로드만 ksem_io 사용.
"""

from __future__ import annotations
import sys
import argparse
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
MIN_SPE_DURATION_H = 1

# ── 이 파일의 방식 고정 (Löwe et al. 2025) ───────────────────────
BG_WINDOW_DAYS = 5         # [day] Löwe 원문은 직전 5일. (윈도우 민감도 시 변경)
LOWE_MULT      = 1.25      # T = 1.25 · B  (배경 25% 초과)

PROTON_LOGICS   = ["O", "OU", "OUT", "CR"]
ELECTRON_LOGICS = ["F", "FT", "FTU", "FTUO"]
TARGET_LOGICS   = PROTON_LOGICS + ELECTRON_LOGICS
PD_KEYS         = ["PD1", "PD2", "PD3"]
SIDES           = ["A", "B"]

ONSET_FLOOR = 0.5
PEAK_FLOOR  = 2.0
MIN_PTS_PER_CHANNEL = 100

TAG = "blc1_lowe"   # 출력 파일명 태그


def parse_args():
    """파라미터만 argument로 받는다. 미지정 시 위 기본값 사용.
    출력 폴더/파일명은 TAG + 이 값들에서 자동 생성되므로 직접 지정 불필요."""
    p = argparse.ArgumentParser(
        description=f"FSM count SPE detector [{TAG}] — Löwe et al. (2025) "
                    f"5d constant-fit background, T=mult·B. "
                    f"params override defaults, output names auto-built from TAG+params.")
    p.add_argument("--window", type=int,   default=BG_WINDOW_DAYS,
                   help=f"background sliding window [day] (default {BG_WINDOW_DAYS}, Löwe=5)")
    p.add_argument("--mult",   type=float, default=LOWE_MULT,
                   help=f"threshold multiplier T=mult*B (default {LOWE_MULT}, Löwe=1.25)")
    p.add_argument("--onset",  type=float, default=ONSET_FLOOR,
                   help=f"onset floor, threshold lower clip (default {ONSET_FLOOR})")
    p.add_argument("--peak",   type=float, default=PEAK_FLOOR,
                   help=f"peak floor, post-filter (default {PEAK_FLOOR})")
    return p.parse_args()


def build_runtag(tag, window, mult, onset, peak):
    """출력 파일명용 파라미터 문자열. 예: blc1_lowe_w5_m1.25_on0.5_pk2.0"""
    return f"{tag}_w{window}_m{_numstr(mult)}_on{_numstr(onset)}_pk{_numstr(peak)}"


def _numstr(v):
    """10.0 -> '10', 0.5 -> '0.5', 1.25 -> '1.25' (파일명 깔끔하게)."""
    f = float(v)
    return str(int(f)) if f.is_integer() else str(f)


FSM_OUTPUT_DIR = _THIS_DIR / "fsm2_output"
FSM_OUTPUT_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════
# 배경 중심: constant linear fit (Löwe). 0차 polyfit → 해석해는 mean.
# ══════════════════════════════════════════════════════════════════
def _constant_fit(x) -> float:
    """Löwe "constant linear fit (y = a)" 를 그대로 호출.

    np.polyfit(t, x, deg=0) 의 해는 a = mean(x) 로 떨어진다(0차 다항=상수).
    결과는 표본 평균과 동일하지만, 논문 문구(constant linear fit)를 코드가
    1:1로 따르도록 polyfit 으로 구현한다. 시간축은 균일 15분 격자이므로
    등간격 인덱스 t를 쓰며, 0차 피팅에서는 t 값 자체가 결과에 무관하다.
    """
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    if len(x) < 1:
        return np.nan
    t = np.arange(len(x), dtype=float)
    a = np.polyfit(t, x, 0)          # 0차 다항(상수) 피팅 → a == mean(x)
    return float(a[0])


# ══════════════════════════════════════════════════════════════════
# 배경 추정 + 검출 엔진 (4종 FSM과 뒷단 공통, 배경부만 Löwe)
# ══════════════════════════════════════════════════════════════════
def _window_samples(cnt: pd.Series, t_ref: pd.Timestamp,
                    bg_window_days: int) -> pd.Series:
    """t_ref 이전 bg_window_days 윈도우의 배경 표본 (전체 사용, quiet 선택 없음)."""
    seg = cnt.loc[t_ref - pd.Timedelta(days=bg_window_days):t_ref].dropna()
    if len(seg) < 10:
        return pd.Series(dtype=float)
    return seg


def compute_rolling_bg(cnt: pd.Series, bg_window_days: int,
                       update_freq: str = "1D") -> pd.DataFrame:
    """rolling 배경 시계열. update_freq마다 갱신 후 ffill.
    반환: DataFrame[bg_median, bg_std].
      bg_median 컬럼 = Löwe 배경 B (constant-fit mean). 스키마 호환 위해 이름 유지.
      bg_std    컬럼 = NaN (Löwe엔 산포 항이 없음)."""
    if cnt.empty:
        return pd.DataFrame(columns=["bg_median", "bg_std"])
    update_points = pd.date_range(cnt.index[0], cnt.index[-1], freq=update_freq)
    rows = []
    for t in update_points:
        q = _window_samples(cnt, t, bg_window_days)
        if len(q) >= 1:
            rows.append((t, _constant_fit(q), np.nan))   # B = constant fit, σ 없음
        else:
            rows.append((t, np.nan, np.nan))
    bg = pd.DataFrame(rows, columns=["t", "bg_median", "bg_std"]).set_index("t")
    return bg.reindex(cnt.index.union(bg.index)).ffill().reindex(cnt.index)


def build_threshold(bg: pd.DataFrame, mult: float, onset_floor: float) -> pd.Series:
    """Löwe 임계 = mult · B  (B = bg_median 컬럼의 constant-fit mean).
    onset_floor로 하한 클립 (저카운트 채널에서 B·mult가 바닥일 때 다른 FSM과 정합)."""
    th = mult * bg["bg_median"]
    if onset_floor > 0:
        th = th.clip(lower=onset_floor)
    return th


def detect_segments(cnt: pd.Series, thresh_series: pd.Series,
                    bg: pd.DataFrame, min_duration_h: float) -> list:
    """count >= rolling 임계가 min_duration_h 이상 연속인 구간. peak_floor는 호출부에서.
    (4종 FSM의 detect_segments와 동일 — bg_sigma만 NaN으로 기록됨)"""
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
def load_count() -> pd.DataFrame:
    df_count, _ = ksem_io.load(COUNT_PARQUET_DIR)
    if not df_count.empty and df_count.index.tz is None:
        df_count.index = df_count.index.tz_localize("UTC")
    return df_count


def main():
    args = parse_args()
    window   = args.window
    mult     = args.mult
    onset_fl = args.onset
    peak_fl  = args.peak
    runtag   = build_runtag(TAG, window, mult, onset_fl, peak_fl)
    out_dir  = FSM_OUTPUT_DIR / runtag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fsm:{TAG}] params: window={window}d mult={mult} onset={onset_fl} "
          f"peak={peak_fl}  (Löwe: 5d constant-fit mean, T={mult}·B)")
    print(f"[fsm:{TAG}] runtag: {runtag}")
    print(f"[fsm:{TAG}] Loading count data...")
    df_count = load_count()
    cnt_rs = df_count.resample(RESAMPLE_FREQ).mean()

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
                bg   = compute_rolling_bg(cnt, window, BG_UPDATE_FREQ)
                thr  = build_threshold(bg, mult, onset_fl)
                segs = detect_segments(cnt, thr, bg, MIN_SPE_DURATION_H)
                # k 컬럼은 스키마 호환용 placeholder(0). Löwe엔 k 개념이 없다.
                base = {"k": 0, "onset_floor": onset_fl, "pd_key": pd_key,
                        "side": side, "logic": logic, "channel": chan}
                for s in segs:
                    onset_rows.append({**base, **s})
                passed = [s for s in segs if s["peak_count"] >= peak_fl]
                for s in passed:
                    event_rows.append({**base, "peak_floor": peak_fl, **s})
                print(f"    segs={len(segs)}  peak>={peak_fl}: {len(passed)}")

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
    print(f"[fsm:{TAG}] METHOD=Löwe constant-fit  WINDOW={window}d MULT={mult} "
          f"floor={onset_fl} peak={peak_fl}")
    if not df_on.empty:
        n_on = df_on.groupby("channel").size().rename("n_onset")
        n_ev = (df_ev.groupby("channel").size().rename("n_event")
                if not df_ev.empty else pd.Series(dtype=int, name="n_event"))
        tbl = pd.concat([n_on, n_ev], axis=1).fillna(0).astype(int).sort_values("n_onset", ascending=False)
        print(f"\n[fsm:{TAG}] per-channel counts:")
        print(tbl.to_string())


if __name__ == "__main__":
    main()
