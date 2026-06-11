"""
ana_event_count_profile.py
===========================
SPE / eSPE 카탈로그 기반 채널별 count 시계열 진단.

[기능 1] 이벤트별 패널 (ana2 참고)
  - onset 앞 24h ~ peak 뒤 72h(SPE) / 48h(eSPE) 구간
  - 행 구성: [flux proxy 없음 → 직접 count만] 채널별 행
  - PD1=파랑 / PD2=초록 / PD3=보라 / A=실선 / B=점선
  - onset(주황 --), peak(빨강 :) 수직선

[기능 2] Superposed epoch (ana4 참고)
  - peak 기준 정규화, 전 이벤트 겹쳐서 median + IQR 음영
  - 3행(PD1/PD2/PD3) × 채널수 열

end_time 처리:
  NOAA SPE  카탈로그에 end 없음 → peak + POST_PEAK_NOAA_SPE_H 를 end 대리로 사용.
  SWPC eSPE 카탈로그에 end 없음 → peak + POST_PEAK_SWPC_ESPE_H 를 end 대리로 사용.
  (SPE decay는 느리고 eSPE는 onset==peak 비율이 26%로 높아 짧게 설정.)

자립형: ksem_io + spe/espe parquet 경로만 맞추면 실행.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── 경로 설정 (여기만 수정) ───────────────────────────────────────
# ksem_io.py 가 있는 디렉터리 (KSEM_count/ 와 같은 위치)
KSEM_IO_DIR       = Path(__file__).parent   # 이 파일을 KSEM_count/에 두는 전제

# count parquet 디렉터리 (ksem_flux_config.py 의 COUNT_PARQUET_DIR 와 동일)
COUNT_PARQUET_DIR = KSEM_IO_DIR / "ksem_cache_parquet"

# ── 이벤트 카탈로그 (io 모듈 캐시 디렉터리) ───────────────────────
# match 코드와 동일하게 io 모듈로 load. 단일 parquet 직접 읽기(X) → io.load(dir).
_C_KSEM           = KSEM_IO_DIR.parent          # C_KSEM/
# NOAA SPE: noaa_goes_spe_io.load(dir) → (df, meta)
NOAA_IO_DIR       = _C_KSEM / "NOAA_GOES"
NOAA_SPE_CATALOG_DIR  = NOAA_IO_DIR / "noaa_goes_spe_cache_parquet"
# SWPC eSPE: swpc_alert_espe_io.load(dir) → (df, meta)
SWPC_IO_DIR       = _C_KSEM / "SWPC_Alert"
SWPC_ESPE_CATALOG_DIR = SWPC_IO_DIR / "espe_cache_parquet"

# KSEM 데이터 기간 (match 코드 KSEM_ERA와 동일)
KSEM_ERA          = ("2019-01-01", "2024-12-31")

# 출력 디렉터리 — KSEM_count/ana_output/ (이 스크립트 전용 신규)
#   패널: ana_output/panels_noaa_spe/, ana_output/panels_swpc_espe/ (이벤트별)
#   epoch/overplot: ana_output/ 직속
OUT_DIR           = KSEM_IO_DIR / "ana_output"

# io 모듈 import (ksem_io는 KSEM_count/, spe/espe io는 각 폴더에 있음)
for _p in (KSEM_IO_DIR, NOAA_IO_DIR, SWPC_IO_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import ksem_io               # noqa: E402
import noaa_goes_spe_io as spe_io      # noqa: E402
import swpc_alert_espe_io as espe_io   # noqa: E402

# ── 파라미터 ──────────────────────────────────────────────────────
RESAMPLE_FREQ       = "15min"   # count 리샘플 (원본 1분 → 15분)

PRE_ONSET_H              = 24        # onset 앞 여백 [h]
POST_PEAK_NOAA_SPE_H     = 72        # NOAA SPE: peak 뒤 end 대리 [h]
POST_PEAK_SWPC_ESPE_H    = 48        # SWPC eSPE: peak 뒤 end 대리 [h]

# superposed epoch 범위 (peak 기준)
EPOCH_PRE_H         = 24        # peak 앞 [h]
EPOCH_POST_H        = 72        # peak 뒤 [h] (NOAA SPE 기준, SWPC eSPE는 48h로 잘림)
EPOCH_BIN_MIN       = 60        # epoch 보간 분해능 [min]

NOAA_SPE_MIN_PFU    = 10        # NOAA >10 MeV 양성자 최소 10 pfu 이상만
SWPC_ESPE_MIN_PFU   = 1000      # SWPC >2 MeV 전자 1000 pfu 이상

# ── 기능3(통계) 전용 파라미터 ───────────────────────────────────
# 넓은 마스크(onset~peak+post_h)는 상승 전·감쇠 후 낮은 count가 섞여
# event median을 희석한다. 좁은 마스크(onset~peak+STAT_NARROW_POST_H)를
# 추가로 산출해 신호 수준을 정직하게 본다. peak 수준 통계도 함께.
STAT_NARROW_POST_H  = 12        # 좁은 이벤트 마스크: onset ~ peak+12h
STAT_PEAK_WIN_H     = 6         # 이벤트별 peak±이 시간 내 최댓값을 "이벤트 peak"로
STAT_TARGET_FPR     = 0.05      # FPR 제약 임계: quiet 오경보율 ≤ 이 값 하의 최대 TPR

# 채널 설정
# ── 그룹 분류 (노션 "KSEM 채널 기본 통계" p.31~34 기준) ──────────────
# quiet 배경 거동으로 매긴 그룹. 판정어(suitable/unsuitable)는 박지 않고
# 관측된 배경 특성만 라벨에 둔다 — 이벤트 거동은 이 분석으로 처음 보므로
# 결론을 입력에 미리 넣지 않기 위함.
#   A: OU/OUT       — 평소 거의 0 (low quiet bg, 드문 스파이크)
#   B: FTU/FTUO     — 중간 배경 (intermediate bg, 계절변동 큼)
#   C: O/F/FT/CR    — 배경 항상 활발 (active bg)
GROUP_LOGICS = {
    "A": ["OU", "OUT"],
    "B": ["FTU", "FTUO"],
    "C": ["O", "F", "FT", "CR"],
}

# ── 패널 행 배치 ────────────────────────────────────────────────
# 그룹 C(O/F/FT/CR)는 count 스케일이 서로 크게 달라(O~수천, CR~만, F~만,
# FT~수백) 한 행에 겹치면 y축이 안 맞는다. → O&CR, F&FT 두 행으로 분리.
# 총 4행: A(OU/OUT), B(FTU/FTUO), C1(O/CR), C2(F/FT).
PANEL_ROWS = [
    ("A",  ["OU", "OUT"],   "Group A: OU / OUT  (low quiet bg)"),
    ("B",  ["FTU", "FTUO"], "Group B: FTU / FTUO  (intermediate bg)"),
    ("C1", ["O", "CR"],     "Group C: O / CR  (active bg)"),
    ("C2", ["F", "FT"],     "Group C: F / FT  (active bg)"),
]

# 전 채널 동일 비교: 양성자/전자/CR 구분 없이 두 카탈로그 패널 모두
# 같은 4행 구성으로 그린다. (NOAA SPE 이벤트든 SWPC eSPE 이벤트든
#  패널 행 구성은 동일, 기준 시각만 다름)
PD_KEYS             = ["PD1", "PD2", "PD3"]
SIDES               = ["A", "B"]

PALETTE = {"PD1": "#2980b9", "PD2": "#27ae60", "PD3": "#8e44ad"}
LS_MAP  = {"A": "-",  "B": "--"}


# ── 카탈로그 로드 (match 코드와 동일하게 io 모듈 사용) ────────────
def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """인덱스를 UTC datetime으로 보장 (begin_time index 전제)."""
    if df.index.tz is None:
        df.index = pd.to_datetime(df.index, utc=True)
    return df


def _filter_era(df: pd.DataFrame, io_mod) -> pd.DataFrame:
    """KSEM 기간으로 필터. io 모듈에 filter_by_date 있으면 사용, 없으면 인덱스로."""
    if hasattr(io_mod, "filter_by_date"):
        return io_mod.filter_by_date(df, *KSEM_ERA)
    lo, hi = pd.Timestamp(KSEM_ERA[0], tz="UTC"), pd.Timestamp(KSEM_ERA[1], tz="UTC")
    return df[(df.index >= lo) & (df.index <= hi)]


def load_noaa_spe() -> pd.DataFrame:
    """NOAA SPE 카탈로그 — noaa_goes_spe_io.load(dir) → (df, meta)."""
    df, meta = spe_io.load(str(NOAA_SPE_CATALOG_DIR))
    df = _ensure_utc_index(df)
    df = _filter_era(df, spe_io)
    df = df[df["max_pfu"] >= NOAA_SPE_MIN_PFU].copy()
    df["end_time"] = df["max_time"] + pd.Timedelta(hours=POST_PEAK_NOAA_SPE_H)
    print(f"[catalog] NOAA SPE {KSEM_ERA[0][:4]}+: {len(df)}개  "
          f"(≥{NOAA_SPE_MIN_PFU} pfu)  meta={'ok' if meta else 'none'}")
    return df


def load_swpc_espe() -> pd.DataFrame:
    """SWPC eSPE 카탈로그 — swpc_alert_espe_io.load(dir) → (df, meta)."""
    df, meta = espe_io.load(str(SWPC_ESPE_CATALOG_DIR))
    df = _ensure_utc_index(df)
    df = _filter_era(df, espe_io)
    df = df[df["max_pfu"] >= SWPC_ESPE_MIN_PFU].copy()
    df["end_time"] = df["max_time"] + pd.Timedelta(hours=POST_PEAK_SWPC_ESPE_H)
    print(f"[catalog] SWPC eSPE: {len(df)}개  (≥{SWPC_ESPE_MIN_PFU} pfu)  "
          f"meta={'ok' if meta else 'none'}")
    return df


# ── count 로드 ────────────────────────────────────────────────────
def load_count() -> pd.DataFrame:
    print(f"[data] count 로드: {COUNT_PARQUET_DIR}")
    df, _ = ksem_io.load(COUNT_PARQUET_DIR)
    if not df.empty and df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df.resample(RESAMPLE_FREQ).mean()
    print(f"[data] count shape: {df.shape},  기간: {df.index[0].date()} ~ {df.index[-1].date()}")
    return df


# ─────────────────────────────────────────────────────────────────
# 기능 1: 이벤트별 패널 (그룹 A/B/C 3행)
# ─────────────────────────────────────────────────────────────────
# logic 구분: 같은 그룹 안에 여러 logic이 겹치므로 marker/linewidth로 구분.
LOGIC_LW = {  # 그룹 내 logic 구분용 선 두께
    "OU": 1.3, "OUT": 0.8,
    "FTU": 1.3, "FTUO": 0.8,
    "O": 1.3, "F": 1.0, "FT": 0.8, "CR": 0.6,
}


def plot_event_panels(df_count: pd.DataFrame,
                      catalog: pd.DataFrame,
                      mode: str,          # "noaa_spe" | "swpc_espe"
                      out_prefix: str,
                      panel_subdir: str):
    """
    이벤트별 count 시계열 패널 — PANEL_ROWS 4행 고정.
    행: A(OU/OUT), B(FTU/FTUO), C1(O/CR), C2(F/FT).
    각 행 = 그 행 logic × PD × side 겹침.
      PD1=blue / PD2=green / PD3=purple, A=solid / B=dashed,
      logic은 선 두께로 구분(범례 표기).
    그룹 C는 count 스케일 차로 두 행(O&CR, F&FT)으로 분리.
    NOAA SPE / SWPC eSPE 두 카탈로그 모두 동일 4행 구성.
    출력: OUT_DIR/panel_subdir/ (ana2 규약), {out_prefix}_event_NNN_panel.png
    """
    panel_dir = OUT_DIR / panel_subdir
    panel_dir.mkdir(parents=True, exist_ok=True)

    label  = "NOAA-SPE" if mode == "noaa_spe" else "SWPC-eSPE"
    post_h = POST_PEAK_NOAA_SPE_H if mode == "noaa_spe" else POST_PEAK_SWPC_ESPE_H
    n_rows = len(PANEL_ROWS)   # 4

    for ei, (onset, row) in enumerate(catalog.iterrows()):
        peak     = row["max_time"]
        end_plot = peak + pd.Timedelta(hours=post_h)
        t0       = onset - pd.Timedelta(hours=PRE_ONSET_H)
        t1       = end_plot

        if t0 > df_count.index[-1] or t1 < df_count.index[0]:
            print(f"  [{label} #{ei+1}] 데이터 범위 밖, 스킵")
            continue

        fig, axes = plt.subplots(n_rows, 1,
                                  figsize=(14, 2.8 * n_rows + 0.6),
                                  sharex=True)

        pfu_str   = f"{row['max_pfu']:.0f} pfu"
        flare_str = ""
        if mode == "noaa_spe" and row.get("flare_class", ""):
            flare_str = f"  |  Flare: {row['flare_class']}"
        title = (f"{label} Event #{ei+1}   "
                 f"Onset: {onset.strftime('%Y-%m-%d %H:%M')}   "
                 f"Peak: {peak.strftime('%Y-%m-%d %H:%M')}   "
                 f"Max: {pfu_str}{flare_str}\n"
                 f"End(proxy): peak+{post_h}h  |  "
                 f"PD1=blue / PD2=green / PD3=purple / A=solid / B=dashed / logic=linewidth")
        fig.suptitle(title, fontsize=9, y=1.005)

        for ri, (rkey, logics, rlabel) in enumerate(PANEL_ROWS):
            ax = axes[ri]
            any_plotted = False

            for logic in logics:
                lw = LOGIC_LW.get(logic, 1.0)
                for pd_key in PD_KEYS:
                    color = PALETTE[pd_key]
                    for side in SIDES:
                        try:
                            s = df_count[pd_key, side, logic].loc[t0:t1]
                        except KeyError:
                            continue
                        if s.dropna().empty:
                            continue
                        ax.plot(s.index, s.values,
                                color=color, ls=LS_MAP[side], lw=lw, alpha=0.8,
                                label=f"{pd_key}{side}-{logic}")
                        any_plotted = True

            ax.axvline(onset,    ls="--", color="orange", lw=1.2, alpha=0.9, label="onset")
            ax.axvline(peak,     ls=":",  color="red",    lw=1.2, alpha=0.9, label="peak")
            ax.axvline(end_plot, ls="--", color="gray",   lw=0.9, alpha=0.6, label=f"end(+{post_h}h)")

            ax.set_ylim(bottom=0)
            ax.set_ylabel(f"Count\n{rlabel}", fontsize=7.5)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=5, ncol=8, loc="upper left", framealpha=0.7)

            if not any_plotted:
                ax.text(0.5, 0.5, "no data",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=9, color="gray")

        axes[-1].xaxis.set_major_formatter(
            mdates.DateFormatter("%m-%d\n%H:%M"))
        axes[-1].xaxis.set_major_locator(
            mdates.AutoDateLocator(minticks=6, maxticks=12))
        fig.tight_layout()

        fname = panel_dir / f"{out_prefix}_event_{ei+1:03d}_panel.png"
        fig.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  [{label} #{ei+1}] {onset.date()} / {row['max_pfu']:.0f} pfu  → {fname.name}")

    print(f"[panel] {label} {len(catalog)}개 저장 → {panel_dir}")


# ─────────────────────────────────────────────────────────────────
# 기능 2: Superposed Epoch
# ─────────────────────────────────────────────────────────────────
def build_epoch_matrix(cnt: pd.Series,
                        catalog: pd.DataFrame,
                        h_axis: np.ndarray) -> pd.DataFrame:
    """
    peak 기준 epoch 행렬. 행=이벤트, 열=peak 기준 시간[h].
    값 = count / count(at catalog peak)  (ana4와 동일: 카탈로그 peak 시각의
    count 한 점으로 정규화). count peak가 카탈로그 peak와 어긋나면 어긋나는
    대로 — 선행/후행 정보이므로 데이터를 보정하지 않는다.
    정규화 분모가 0/NaN이면 그 이벤트 스킵.
    """
    rows = []
    for onset, row in catalog.iterrows():
        peak = row["max_time"]
        idx_pk = cnt.index.get_indexer([peak], method="nearest")[0]
        pk_cnt = cnt.iloc[idx_pk] if 0 <= idx_pk < len(cnt) else np.nan
        if not (np.isfinite(pk_cnt) and pk_cnt > 0):
            continue
        vals = []
        for dh in h_axis:
            t   = peak + pd.Timedelta(hours=float(dh))
            idx = cnt.index.get_indexer([t], method="nearest")[0]
            vals.append(cnt.iloc[idx] / pk_cnt if 0 <= idx < len(cnt) else np.nan)
        rows.append(vals)
    return pd.DataFrame(rows, columns=h_axis) if rows else pd.DataFrame()


def plot_superposed_epoch(df_count: pd.DataFrame,
                           catalog: pd.DataFrame,
                           logics: list[str],
                           mode: str,          # "noaa_spe" | "swpc_espe"
                           out_prefix: str):
    """
    3행(PD1/PD2/PD3) × len(logics)열 superposed epoch.
    line=median, shade=IQR, solid=A side, dashed=B side.
    출력: OUT_DIR(=KSEM_count/) 직속, {out_prefix}_superposed_epoch.png
    """
    label    = "NOAA-SPE" if mode == "noaa_spe" else "SWPC-eSPE"
    post_h   = POST_PEAK_NOAA_SPE_H if mode == "noaa_spe" else POST_PEAK_SWPC_ESPE_H
    epoch_post = min(EPOCH_POST_H, post_h)

    h_axis = np.arange(-EPOCH_PRE_H,
                        epoch_post + EPOCH_BIN_MIN / 60,
                        EPOCH_BIN_MIN / 60)

    nrows = len(PD_KEYS)
    ncols = len(logics)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.8 * ncols, 3.5 * nrows),
                              squeeze=False)
    fig.suptitle(
        f"{label} Superposed Epoch  (peak-normalized, {EPOCH_BIN_MIN}-min bin)\n"
        "line=median  shade=IQR  solid=A side  dashed=B side",
        fontsize=11)

    for ri, pd_key in enumerate(PD_KEYS):
        for ci, logic in enumerate(logics):
            ax = axes[ri][ci]
            n_plotted = 0

            for side in SIDES:
                try:
                    s = df_count[pd_key, side, logic].dropna()
                except KeyError:
                    continue
                mat = build_epoch_matrix(s, catalog, h_axis)
                if mat.empty:
                    continue
                n   = len(mat)
                h   = mat.columns.values.astype(float)
                med = mat.median(axis=0).values
                q25 = mat.quantile(0.25, axis=0).values
                q75 = mat.quantile(0.75, axis=0).values
                c   = PALETTE[pd_key]

                ax.plot(h, med, color=c, ls=LS_MAP[side], lw=1.6,
                        label=f"{side} (n={n})")
                if side == "A":
                    ax.fill_between(h, q25, q75, color=c, alpha=0.15)
                n_plotted += 1

            ax.axvline(0,  ls="--", color="red",  lw=1,   label="Peak")
            ax.axvline(-((catalog["max_time"] - catalog.index)
                          .dt.total_seconds() / 3600).median(),
                       ls=":", color="orange", lw=0.9, alpha=0.7,
                       label=f"onset(median)")
            ax.axhline(1,  ls=":",  color="gray", lw=0.8)
            ax.set_xlim(-EPOCH_PRE_H, epoch_post)
            ax.set_xlabel("Time from peak [h]", fontsize=8)
            ax.set_ylabel("Normalized Count", fontsize=8)
            ax.set_title(f"{pd_key} / {logic}", fontsize=9)
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=7)

            if n_plotted == 0:
                ax.text(0.5, 0.5, "no data",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=9, color="gray")

    fig.tight_layout()
    out_path = OUT_DIR / f"{out_prefix}_superposed_epoch.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[epoch] {label} superposed epoch → {out_path.name}")


# ─────────────────────────────────────────────────────────────────
# 기능 2b: 전 이벤트 raw 겹치기 (정규화 없이 count 원값)
# ─────────────────────────────────────────────────────────────────
def plot_overplot_raw(df_count: pd.DataFrame,
                      catalog: pd.DataFrame,
                      logics: list[str],
                      mode: str,          # "noaa_spe" | "swpc_espe"
                      out_prefix: str):
    """
    peak 기준으로 시간축 정렬 후 count 원값을 모두 겹쳐 그림.
    median 굵은 선 + 개별 이벤트 가는 선(반투명).
    → 일변화 구조나 rise time 시각적 확인에 유용.
    출력: OUT_DIR(=KSEM_count/) 직속, {out_prefix}_overplot_raw.png
    """
    label    = "NOAA-SPE" if mode == "noaa_spe" else "SWPC-eSPE"
    post_h   = POST_PEAK_NOAA_SPE_H if mode == "noaa_spe" else POST_PEAK_SWPC_ESPE_H
    epoch_post = min(EPOCH_POST_H, post_h)

    h_axis = np.arange(-EPOCH_PRE_H,
                        epoch_post + EPOCH_BIN_MIN / 60,
                        EPOCH_BIN_MIN / 60)

    nrows = len(PD_KEYS)
    ncols = len(logics)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.8 * ncols, 3.5 * nrows),
                              squeeze=False)
    fig.suptitle(
        f"{label} all-event overplot  (peak-aligned, raw count)\n"
        "thick=median  thin=individual events  solid=A side  dashed=B side",
        fontsize=11)

    for ri, pd_key in enumerate(PD_KEYS):
        for ci, logic in enumerate(logics):
            ax = axes[ri][ci]

            for side in SIDES:
                try:
                    s = df_count[pd_key, side, logic].dropna()
                except KeyError:
                    continue

                # 각 이벤트별 peak 기준 재샘플
                event_traces = []
                for onset, row in catalog.iterrows():
                    peak = row["max_time"]
                    vals = []
                    for dh in h_axis:
                        t   = peak + pd.Timedelta(hours=float(dh))
                        idx = s.index.get_indexer([t], method="nearest")[0]
                        vals.append(s.iloc[idx] if 0 <= idx < len(s) else np.nan)
                    event_traces.append(vals)

                mat = np.array(event_traces, dtype=float)
                if mat.shape[0] == 0:
                    continue

                c  = PALETTE[pd_key]
                ls = LS_MAP[side]

                # 개별 이벤트 (가는 선)
                for trace in mat:
                    ax.plot(h_axis, trace, color=c, ls=ls,
                            lw=0.5, alpha=0.2)
                # median 굵은 선
                med = np.nanmedian(mat, axis=0)
                ax.plot(h_axis, med, color=c, ls=ls, lw=2.2,
                        alpha=0.95, label=f"{side} med (n={mat.shape[0]})")

            ax.axvline(0,  ls="--", color="red",  lw=1.2, label="Peak")
            # onset 중앙값 표시
            med_rise = (catalog["max_time"] - catalog.index
                        ).dt.total_seconds().median() / 3600
            ax.axvline(-med_rise, ls=":", color="orange",
                       lw=1, alpha=0.8, label=f"onset(median)")
            ax.set_xlim(-EPOCH_PRE_H, epoch_post)
            ax.set_ylim(bottom=0)
            ax.set_xlabel("Time from peak [h]", fontsize=8)
            ax.set_ylabel("Count (raw)", fontsize=8)
            ax.set_title(f"{pd_key} / {logic}", fontsize=9)
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=7)

    fig.tight_layout()
    out_path = OUT_DIR / f"{out_prefix}_overplot_raw.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[overplot] {label} raw overplot → {out_path.name}")


# ─────────────────────────────────────────────────────────────────
# 기능 3: 이벤트 vs 비이벤트 count 통계 + 고정임계 C 후보 (CSV)
# ─────────────────────────────────────────────────────────────────
# 목적: BL-C1 고정임계 C 결정의 정량 근거.
#   - 이벤트 마스크: 카탈로그 [onset, peak+post_h] 구간 (패널과 동일 정의)
#   - quiet 마스크: 그 외 전 구간 (ana3 get_quiet_mask 차용 — 단 여기선
#     flux proxy가 아니라 카탈로그 이벤트 구간을 기준으로 quiet를 정의.
#     proxy 대역 불일치 문제를 우회하는 것이 이 분석의 핵심)
# C 후보 두 방식:
#   - quiet 분위수 (Papaioannou식 정온기 배경): quiet count의 p50~p99.9
#   - ROC/Youden 최적 임계: 이벤트/quiet 두 분포를 가장 잘 가르는 count 값
#     (Youden J = TPR - FPR 최대점). sklearn 없이 직접 계산.
def _build_event_mask(cnt_index: pd.DatetimeIndex,
                      catalog: pd.DataFrame,
                      post_h: float,
                      pre_h: float = 0.0) -> np.ndarray:
    """카탈로그 [onset-pre_h, peak+post_h] 구간에 속하면 True (이벤트).
    pre_h=0이면 onset부터. 좁은 마스크는 post_h를 작게 줘서 호출."""
    mask = np.zeros(len(cnt_index), dtype=bool)
    for onset, row in catalog.iterrows():
        peak  = row["max_time"]
        start = onset - pd.Timedelta(hours=pre_h)
        end   = peak  + pd.Timedelta(hours=post_h)
        mask |= (cnt_index >= start) & (cnt_index <= end)
    return mask


def _event_peak_levels(cnt: pd.Series,
                       catalog: pd.DataFrame,
                       win_h: float) -> np.ndarray:
    """이벤트별 peak±win_h 윈도우 내 count 최댓값 = '이벤트 peak 수준'.
    median이 희석되는 문제를 피해, 이벤트당 대표 신호 1개를 뽑는다."""
    peaks = []
    for onset, row in catalog.iterrows():
        peak = row["max_time"]
        seg = cnt.loc[peak - pd.Timedelta(hours=win_h):
                      peak + pd.Timedelta(hours=win_h)]
        v = seg.max()
        if np.isfinite(v):
            peaks.append(float(v))
    return np.array(peaks, dtype=float)


def _roc_youden_threshold(event_vals: np.ndarray,
                          quiet_vals: np.ndarray) -> tuple:
    """이벤트(양성)/quiet(음성) count 분포를 가르는 Youden 최적 임계.
    반환: (best_threshold, best_J, tpr_at_best, fpr_at_best).
    후보 임계는 두 분포 합집합의 고유값들. sklearn 비의존."""
    ev = event_vals[np.isfinite(event_vals)]
    qt = quiet_vals[np.isfinite(quiet_vals)]
    if len(ev) < 5 or len(qt) < 5:
        return (np.nan, np.nan, np.nan, np.nan)
    cand = np.unique(np.concatenate([ev, qt]))
    if len(cand) > 2000:
        cand = np.quantile(cand, np.linspace(0, 1, 2000))
    best = (np.nan, -np.inf, np.nan, np.nan)
    n_ev, n_qt = len(ev), len(qt)
    for thr in cand:
        tpr = (ev >= thr).sum() / n_ev
        fpr = (qt >= thr).sum() / n_qt
        J = tpr - fpr
        if J > best[1]:
            best = (float(thr), float(J), float(tpr), float(fpr))
    return best


def _fpr_constrained_threshold(event_vals: np.ndarray,
                               quiet_vals: np.ndarray,
                               target_fpr: float) -> tuple:
    """FPR ≤ target_fpr 제약 하의 최대 TPR 임계. FAR 통제용(BL-C1에 적합).
    반환: (threshold, tpr, fpr). 만족 임계 없으면 NaN."""
    ev = event_vals[np.isfinite(event_vals)]
    qt = quiet_vals[np.isfinite(quiet_vals)]
    if len(ev) < 5 or len(qt) < 5:
        return (np.nan, np.nan, np.nan)
    cand = np.unique(np.concatenate([ev, qt]))
    if len(cand) > 4000:
        cand = np.quantile(cand, np.linspace(0, 1, 4000))
    n_ev, n_qt = len(ev), len(qt)
    best = (np.nan, -np.inf, np.nan)   # (thr, tpr, fpr)
    for thr in cand:
        fpr = (qt >= thr).sum() / n_qt
        if fpr <= target_fpr:
            tpr = (ev >= thr).sum() / n_ev
            if tpr > best[1]:
                best = (float(thr), float(tpr), float(fpr))
    if not np.isfinite(best[1]):
        return (np.nan, np.nan, np.nan)
    return best


def save_event_count_stats(df_count: pd.DataFrame,
                           catalog: pd.DataFrame,
                           mode: str,
                           out_prefix: str):
    """채널별 이벤트/quiet count 통계 + C 후보(분위수, ROC-Youden) → CSV.
    BL-C1 고정임계 C 결정의 정량 근거."""
    label  = "NOAA-SPE" if mode == "noaa_spe" else "SWPC-eSPE"
    post_h = POST_PEAK_NOAA_SPE_H if mode == "noaa_spe" else POST_PEAK_SWPC_ESPE_H

    all_logics = GROUP_LOGICS["A"] + GROUP_LOGICS["B"] + GROUP_LOGICS["C"]
    rows = []
    for pd_key in PD_KEYS:
        for side in SIDES:
            for logic in all_logics:
                try:
                    cnt = df_count[pd_key, side, logic].dropna()
                except KeyError:
                    continue
                if len(cnt) < 100:
                    continue
                # 넓은 마스크(전체 이벤트 구간) — 참고용
                wide_mask = _build_event_mask(cnt.index, catalog, post_h)
                # 좁은 마스크(onset~peak+12h) — 희석 적은 신호 구간 (분리도 계산 기준)
                narrow_mask = _build_event_mask(cnt.index, catalog, STAT_NARROW_POST_H)
                qt_vals  = cnt.values[~wide_mask]          # quiet = 어떤 이벤트에도 안 든 구간
                ev_wide  = cnt.values[wide_mask]
                ev_narrow = cnt.values[narrow_mask]
                if len(ev_narrow) < 5 or len(qt_vals) < 5:
                    continue
                # 이벤트별 peak 수준 (이벤트당 대표 신호 1개)
                epeak = _event_peak_levels(cnt, catalog, STAT_PEAK_WIN_H)

                # quiet 분위수 (Papaioannou식 정온기 배경 → C 후보)
                qp = {f"quiet_p{p}": float(np.nanpercentile(qt_vals, p))
                      for p in (50, 90, 95, 99, 99.9)}
                # 이벤트 분포 (넓은=희석, 좁은=신호, peak수준)
                emed_wide   = float(np.nanmedian(ev_wide))
                emed_narrow = float(np.nanmedian(ev_narrow))
                epeak_med   = float(np.nanmedian(epeak)) if len(epeak) else np.nan
                epeak_p25   = float(np.nanpercentile(epeak, 25)) if len(epeak) else np.nan
                qmed = float(np.nanmedian(qt_vals))
                ratio_narrow = emed_narrow / qmed if qmed > 0 else np.nan
                ratio_epeak  = epeak_med / qmed if qmed > 0 else np.nan

                # 분리도/임계는 좁은 마스크(신호) vs quiet 기준
                c_roc, j_roc, tpr_roc, fpr_roc = _roc_youden_threshold(ev_narrow, qt_vals)
                c_fpr, tpr_fpr, fpr_act = _fpr_constrained_threshold(
                    ev_narrow, qt_vals, STAT_TARGET_FPR)

                rows.append({
                    "pd_key": pd_key, "side": side, "logic": logic,
                    "channel": f"{pd_key}{side}-{logic}",
                    "group": ("A" if logic in GROUP_LOGICS["A"]
                              else "B" if logic in GROUP_LOGICS["B"] else "C"),
                    "n_quiet_pts": int(len(qt_vals)),
                    "n_event_narrow": int(len(ev_narrow)),
                    "n_event_peaks": int(len(epeak)),
                    "quiet_median": round(qmed, 4),
                    "event_median_wide": round(emed_wide, 4),
                    "event_median_narrow": round(emed_narrow, 4),
                    "event_peak_median": round(epeak_med, 4) if np.isfinite(epeak_med) else np.nan,
                    "event_peak_p25": round(epeak_p25, 4) if np.isfinite(epeak_p25) else np.nan,
                    "ratio_narrow": round(ratio_narrow, 3) if np.isfinite(ratio_narrow) else np.nan,
                    "ratio_peak": round(ratio_epeak, 3) if np.isfinite(ratio_epeak) else np.nan,
                    **{k: round(v, 4) for k, v in qp.items()},
                    # C 후보 1: ROC-Youden (분리도 최대)
                    "C_roc_youden": round(c_roc, 4) if np.isfinite(c_roc) else np.nan,
                    "roc_J": round(j_roc, 4) if np.isfinite(j_roc) else np.nan,
                    "roc_tpr": round(tpr_roc, 4) if np.isfinite(tpr_roc) else np.nan,
                    "roc_fpr": round(fpr_roc, 4) if np.isfinite(fpr_roc) else np.nan,
                    # C 후보 2: FPR 제약 (FAR 통제, BL-C1에 적합)
                    f"C_fpr{STAT_TARGET_FPR}": round(c_fpr, 4) if np.isfinite(c_fpr) else np.nan,
                    "fpr_tpr": round(tpr_fpr, 4) if np.isfinite(tpr_fpr) else np.nan,
                    "fpr_actual": round(fpr_act, 4) if np.isfinite(fpr_act) else np.nan,
                })

    df_stats = pd.DataFrame(rows)
    out_csv = OUT_DIR / f"{out_prefix}_event_count_stats.csv"
    df_stats.to_csv(out_csv, index=False)
    print(f"[stats] {label} channel stats → {out_csv.name}  ({len(df_stats)} channels)")
    if not df_stats.empty:
        # 콘솔에 그룹별 best 채널 — 양성자/전자 트리거 채널 분별이 핵심
        cfpr_col = f"C_fpr{STAT_TARGET_FPR}"
        for grp in ("A", "B", "C"):
            g = df_stats[df_stats["group"] == grp].sort_values("roc_J", ascending=False)
            if g.empty:
                continue
            top = g.head(3)[["channel", "ratio_peak", "quiet_p95",
                             "event_peak_median", "C_roc_youden", "roc_J",
                             cfpr_col, "fpr_tpr"]]
            print(f"[stats] {label} Group {grp} — roc_J 상위 3:")
            print(top.to_string(index=False))
    return df_stats


# ─────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 카탈로그 ────────────────────────────────────────────────
    noaa_spe_catalog  = load_noaa_spe()
    swpc_espe_catalog = load_swpc_espe()

    # ── count 로드 ──────────────────────────────────────────────
    df_count = load_count()

    # 패널: PANEL_ROWS 4행 (전 채널). epoch/overplot: PD행 × logic열 그리드
    # (peak 정규화라 y축 스케일 무관, 전 8 logic 펼침).
    ALL_LOGICS = GROUP_LOGICS["A"] + GROUP_LOGICS["B"] + GROUP_LOGICS["C"]

    print("\n=== NOAA SPE (양성자 카탈로그 기준) 전 채널 분석 ===")
    plot_event_panels(df_count, noaa_spe_catalog,
                      mode="noaa_spe", out_prefix="noaa_spe",
                      panel_subdir="panels_noaa_spe")
    plot_superposed_epoch(df_count, noaa_spe_catalog, ALL_LOGICS,
                          mode="noaa_spe", out_prefix="noaa_spe")
    plot_overplot_raw(df_count, noaa_spe_catalog, ALL_LOGICS,
                      mode="noaa_spe", out_prefix="noaa_spe")
    save_event_count_stats(df_count, noaa_spe_catalog,
                           mode="noaa_spe", out_prefix="noaa_spe")

    print("\n=== SWPC eSPE (전자 카탈로그 기준) 전 채널 분석 ===")
    plot_event_panels(df_count, swpc_espe_catalog,
                      mode="swpc_espe", out_prefix="swpc_espe",
                      panel_subdir="panels_swpc_espe")
    plot_superposed_epoch(df_count, swpc_espe_catalog, ALL_LOGICS,
                          mode="swpc_espe", out_prefix="swpc_espe")
    plot_overplot_raw(df_count, swpc_espe_catalog, ALL_LOGICS,
                      mode="swpc_espe", out_prefix="swpc_espe")
    save_event_count_stats(df_count, swpc_espe_catalog,
                           mode="swpc_espe", out_prefix="swpc_espe")

    print(f"\n[완료] 출력 디렉터리: {OUT_DIR}")
    print("  panels_noaa_spe/   : NOAA SPE 이벤트별 패널 (4행)")
    print("  panels_swpc_espe/  : SWPC eSPE 이벤트별 패널 (4행)")
    print("  noaa_spe_superposed_epoch.png / noaa_spe_overplot_raw.png")
    print("  swpc_espe_superposed_epoch.png / swpc_espe_overplot_raw.png")
    print("  noaa_spe_event_count_stats.csv / swpc_espe_event_count_stats.csv  ← BL-C1 C 근거")


if __name__ == "__main__":
    main()