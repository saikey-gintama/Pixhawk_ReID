"""
ksem_channel_diagnostics.py
===========================
KSEM count 채널별 배경 구조 진단 — onset 탐지 적합성 판정용.

FSM(rolling robust-σ onset 탐지)을 어느 채널에 적용할지 결정하기 위해, 각 채널의
배경 통계 구조를 정량화한다. "왜 OU/OUT은 되고 CR·전자는 안 되는가"를 데이터로
보이는 진단 도구.

핵심 지표:
  median, MAD-σ(robust), std, std/MAD 비
    └ std/MAD가 크면 "평소 조용 + 드문 거대 스파이크"(OU형, 탐지 적합).
      작으면(~1) "항상 출렁이는 배경"(FT형, 이벤트 분리 곤란).
  일변화 진폭 (hour별 median 변동 / 전체 median)
  장주기 변동 (월별 median 최대/최소 비)
  detrend 설명력 (일변화·계절 제거 후 std가 몇 % 줄어드는가)
    └ 낮으면(<10%) 변동이 매끄러운 주기가 아니라 산발 스파이크 → detrend 무용.
  high-count 점유율 (count>HIGH_THR 인 시간 비율)
    └ 낮으면 평소 조용(탐지 적합), 높으면 배경이 항상 활발.

MAD vs std는 배경 산포 추정의 두 방식이다(FSM에서 비교 대상):
  MAD = 1.4826·median(|x−median|), 단발 스파이크에 강건.
  std = 표본 표준편차, 스파이크에 폭발.

출력:
  diag_output/channel_diag/channel_diagnostics.csv   전 채널 지표표
  diag_output/channel_diag/profile_{channel}.png     채널별 hour·월 프로파일 + 분포

자립형: ksem_io만 의존. 경로는 상단에서 조정.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 경로 (ksem_io 로드 전용) ──────────────────────────────────────
_THIS_DIR         = Path(__file__).parent.resolve()
KSEM_COUNT_DIR    = _THIS_DIR              # 이 파일을 KSEM_count/에 두는 전제
COUNT_PARQUET_DIR = KSEM_COUNT_DIR / "ksem_cache_parquet"
if str(KSEM_COUNT_DIR) not in sys.path:
    sys.path.insert(0, str(KSEM_COUNT_DIR))
import ksem_io   # noqa: E402

# ── 파라미터 ──────────────────────────────────────────────────────
RESAMPLE_FREQ = "15min"
HIGH_THR      = 10        # high-count 점유율 기준 (count > HIGH_THR)
TARGET_LOGICS = ["O", "OU", "OUT", "CR", "F", "FT", "FTU", "FTUO"]
PD_KEYS       = ["PD1", "PD2", "PD3"]
SIDES         = ["A", "B"]
MIN_PTS       = 100

OUT_DIR = _THIS_DIR / "diag_output" / "channel_diag"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── 통계 헬퍼 ─────────────────────────────────────────────────────
def robust_sigma(x) -> float:
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2: return np.nan
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def diagnose(cnt: pd.Series) -> dict:
    """단일 채널 배경 구조 진단 → 지표 dict."""
    med = float(cnt.median())
    rs  = robust_sigma(cnt)
    sd  = float(cnt.std())

    hourly  = cnt.groupby(cnt.index.hour).median()
    monthly = cnt.groupby(cnt.index.to_period("M").astype(str)).median()

    # detrend: (hour) 및 (hour, month) median 제거 후 std 감소율
    detr_hour = cnt - cnt.index.hour.map(hourly)
    key = list(zip(cnt.index.hour, cnt.index.month))
    hm  = pd.Series(cnt.values,
                    index=pd.MultiIndex.from_tuples(key)).groupby(level=[0, 1]).median()
    detr_hm = cnt.values - np.array([hm[k] for k in key])

    diurnal_amp = (hourly.max() - hourly.min()) / med if med > 0 else np.nan
    season_ratio = monthly.max() / max(monthly.min(), 0.1)
    return {
        "median": round(med, 3),
        "mad_sigma": round(rs, 3),
        "std": round(sd, 2),
        "std_over_mad": round(sd / rs, 1) if rs and rs > 0 else np.nan,
        "p99": round(float(np.percentile(cnt, 99)), 1),
        "p99_9": round(float(np.percentile(cnt, 99.9)), 1),
        "max": round(float(cnt.max()), 1),
        "high_frac_pct": round(100 * float((cnt > HIGH_THR).mean()), 2),
        "diurnal_amp": round(float(diurnal_amp), 2) if np.isfinite(diurnal_amp) else np.nan,
        "season_ratio": round(float(season_ratio), 1),
        "detrend_diurnal_pct": round((1 - detr_hour.std() / sd) * 100, 1) if sd > 0 else np.nan,
        "detrend_seasonal_pct": round((1 - detr_hm.std() / sd) * 100, 1) if sd > 0 else np.nan,
        "n_pts": int(len(cnt)),
        "_hourly": hourly, "_monthly": monthly,   # 플롯용 (CSV에선 제외)
    }


def plot_profile(chan: str, cnt: pd.Series, d: dict):
    """채널별 4-panel: hour 프로파일 / 월 프로파일 / 분포(log) / MAD vs std 임계 비교."""
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"{chan}  channel background diagnostics", fontsize=12)

    # (1) 일변화
    h = d["_hourly"]
    ax[0, 0].bar(h.index, h.values, color="#3498db")
    ax[0, 0].set_title(f"Diurnal (hour) median   amp/median={d['diurnal_amp']}")
    ax[0, 0].set_xlabel("UTC hour"); ax[0, 0].set_ylabel("median count")

    # (2) 월별 장주기
    m = d["_monthly"]
    ax[0, 1].plot(range(len(m)), m.values, color="#e67e22", marker=".", ms=3)
    ax[0, 1].set_title(f"Monthly median   season_ratio={d['season_ratio']}x")
    ax[0, 1].set_xlabel("month index"); ax[0, 1].set_ylabel("median count")
    ax[0, 1].set_yscale("log")

    # (3) count 분포 (log-y 히스토그램)
    vals = cnt[cnt > 0]
    ax[1, 0].hist(np.log10(vals + 1), bins=60, color="#7f8c8d")
    ax[1, 0].axvline(np.log10(d["median"] + 1), color="b", ls="--", label="median")
    ax[1, 0].axvline(np.log10(d["p99"] + 1), color="r", ls=":", label="p99")
    ax[1, 0].set_title(f"log10(count+1) dist   std/MAD={d['std_over_mad']}")
    ax[1, 0].set_xlabel("log10(count+1)"); ax[1, 0].legend(fontsize=8)

    # (4) MAD vs std 임계 비교 (median + k*sigma, k=10)
    k = 10
    thr_mad = d["median"] + k * d["mad_sigma"]
    thr_std = d["median"] + k * d["std"]
    bars = ax[1, 1].bar(["median", "MAD σ", "std σ", f"med+{k}·MAD", f"med+{k}·std"],
                        [d["median"], d["mad_sigma"], d["std"], thr_mad, thr_std],
                        color=["#95a5a6", "#27ae60", "#c0392b", "#27ae60", "#c0392b"])
    ax[1, 1].set_title("MAD vs std threshold (k=10)")
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_ylabel("count")
    for b, v in zip(bars, [d["median"], d["mad_sigma"], d["std"], thr_mad, thr_std]):
        ax[1, 1].text(b.get_x() + b.get_width()/2, v, f"{v:.1f}",
                      ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    out = OUT_DIR / f"profile_{chan}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def load_count() -> pd.DataFrame:
    df, _ = ksem_io.load(COUNT_PARQUET_DIR)
    if not df.empty and df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def main():
    print("[diag] loading count...")
    cnt_rs = load_count().resample(RESAMPLE_FREQ).mean()

    rows = []
    for pd_key in PD_KEYS:
        for side in SIDES:
            for logic in TARGET_LOGICS:
                try:
                    cnt = cnt_rs[pd_key, side, logic].dropna()
                except KeyError:
                    continue
                if len(cnt) < MIN_PTS:
                    continue
                chan = f"{pd_key}{side}-{logic}"
                d = diagnose(cnt)
                plot_profile(chan, cnt, d)
                d_csv = {k: v for k, v in d.items() if not k.startswith("_")}
                rows.append({"channel": chan, "pd_key": pd_key,
                             "side": side, "logic": logic, **d_csv})
                print(f"  {chan:14s} med={d['median']:8.2f} std/MAD={d['std_over_mad']:6} "
                      f"high%={d['high_frac_pct']:5} detrend%={d['detrend_seasonal_pct']:5}")

    df = pd.DataFrame(rows).sort_values("std_over_mad", ascending=False)
    out_csv = OUT_DIR / "channel_diagnostics.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[diag] saved: {out_csv}  ({len(df)} channels)")
    print(f"[diag] profiles: {OUT_DIR}/profile_*.png")
    print("\n[diag] std/MAD 큰 채널 = 평소 조용+드문 스파이크 = 탐지 적합 (OU형)")
    print("[diag] high_frac 작을수록 적합. detrend% 낮으면 산발 스파이크(주기제거 무용).")


if __name__ == "__main__":
    main()
