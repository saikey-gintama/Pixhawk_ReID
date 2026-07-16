"""
diag_rolling_threshold_poes.py
===============================
POES 카탈로그 이벤트 vs raw count 프리커서를 눈으로 검사하는 진단 스크립트.
기본 모드는 raw count + 카탈로그/SAA 음영만 그리는 경량 모드(FSM 계산 전부 스킵)이고,
--fsm 지정 시에만 quietoff_mad 롤링 배경/임계값/onset 오버레이를 복원한다.

재사용 (재구현 없음 — import만):
  fsm_count_spe_quietoff_mad_poes.py : compute_rolling_bg / detect_segments /
      load_count / load_geo (롤링 배경 엔진 그대로, --fsm 모드에서만 호출)
  _match_core_poes.py                : _load_count_channel, _import_event_io
  noaa_goes_spe_io                   : load, filter_by_date
  coords_igrf                        : load_geo(with_bmag=True) 내부에서 이미 호출됨
                                        (poes_*_io.get_geo(with_bmag=True));
                                        dipole_maglat()은 극관 밴드 판정에 재사용
                                        (좌표만 사용 — Bmag/IGRF 추가 계산 없음)

그림 1장 = 채널 1개 x 이벤트 1개.
  검은 실선     : raw count
  옅은 초록 음영: --catalog 이벤트 카탈로그(기본 noaa)의 이벤트 begin~max_time
  초록 점선     : 이벤트 begin / max_time 세로선
  옅은 주황 음영: SAA 구간 (|B|<25000nT)
  파란 계단선   : 극관 통과(pass)별 count median (--maglat-band, 기본 |mlat|>=60)
  옅은 파랑 음영: 극관 통과(pass) 구간
  (--fsm 지정 시에만 추가)
  회색 실선   : rolling bg_median(t)
  빨강 점선   : threshold(t) = bg_median + k*MAD  (onset_floor=0 기본, 순수 공식)
  주황 세로선 : FSM onset (창 안, 라이브로 detect_segments 재계산)
  빨강 세로선 : FSM peak (사전계산 fsm_event_*.csv 있으면만 — --peak 지정 시)

사용:
  # 기본(raw) 모드 -- FSM 계산 스킵, 카탈로그 vs raw count만 육안 검사
  python diag_rolling_threshold_poes.py --detector metop03 --channels pro_tel0_p5 --top-events 2
  python diag_rolling_threshold_poes.py --detector metop03 --catalog swpc --channels omni_p7 --top-events 3

  # 극관 pass median 오버레이 (기본 |mlat|>=60, raw/--fsm 공통)
  python diag_rolling_threshold_poes.py --detector metop03 --channels pro_tel0_p5 --maglat-band 60:80
  python diag_rolling_threshold_poes.py --detector metop03 --channels pro_tel0_p5 --maglat-band ""   # 끔

  # --fsm 모드 -- 롤링 배경/임계값/onset 오버레이 복원 (--w/--k/--channel-params/--onset/--peak은
  # --fsm 지정 시에만 사용 가능)
  python diag_rolling_threshold_poes.py --fsm w30k10 --channels pro_tel0_p5,omni_p7 --top-events 5
  python diag_rolling_threshold_poes.py --fsm w30k10 \
      --channel-params "pro_tel0_p5:w10k3,omni_p7:w1k7" --top-events 3
  python diag_rolling_threshold_poes.py --fsm w10k3 --channels pro_tel0_p5 --all-events
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = Path(__file__).resolve().parent  # C_PD/

sys.path.insert(0, str(HERE / "POES"))
sys.path.insert(0, str(HERE / "POES" / "event_MATCHER"))
sys.path.insert(0, str(HERE / "POES" / "count_FSM"))
import coords_igrf                                    # dipole_maglat (극관 밴드용, 좌표만)
import _match_core_poes as core                      # _load_count_channel, _import_event_io
import fsm_count_spe_quietoff_mad_poes as fsm_engine  # 롤링 배경 엔진 (byte-identical to KSEM)

_POES_IO = {
    "metop03": ("poes_metop03_io", HERE / "POES" / "MetOp03_count" / "poes_metop03_cache_parquet"),
    "noaa19":  ("poes_noaa19_io",  HERE / "POES" / "NOAA19_count"  / "poes_noaa19_cache_parquet"),
}
_FSM_ROOT = {
    "metop03": HERE / "POES" / "MetOp03_count" / "metop03_output" / "2_fsm",
    "noaa19":  HERE / "POES" / "NOAA19_count"  / "noaa19_output"  / "2_fsm",
}
# 값은 4_summarize.py의 _CATALOG와 동일 (사전 조사로 두 카탈로그 모두
# index=begin_time, columns에 max_time/max_pfu 존재 확인 -- 폴백 불필요).
_CATALOG = {
    "noaa": (HERE / "NOAA_GOES"  / "noaa_goes_spe_cache_parquet", "noaa_goes_spe_io"),
    "swpc": (HERE / "SWPC_Alert" / "swpc_espe_cache_parquet",     "swpc_alert_espe_io"),
}

_DEFAULT_OUT = HERE / "diag_output"

_WK_PATTERN = r"w(?P<w>[\d.]+)k(?P<k>[\d.]+)"
_CHPARAM_RE = re.compile(rf"^(?P<chan>[^:]+):{_WK_PATTERN}$")
_FSM_RE = re.compile(rf"^{_WK_PATTERN}$")


def _parse_fsm_arg(fsm_arg: str) -> tuple[int, float]:
    """--fsm "w{W}k{K}" 파싱 -> (w_default, k_default). _CHPARAM_RE와 동일한 w/k 패턴(_WK_PATTERN) 재사용."""
    m = _FSM_RE.match(fsm_arg.strip())
    if not m:
        raise SystemExit(f"[diag] --fsm 형식 오류: '{fsm_arg}' (기대: w{{W}}k{{K}}, 예: w30k10)")
    return int(float(m.group("w"))), float(m.group("k"))


def _parse_channel_params(channels: list[str], w_default: int, k_default: float,
                          channel_params_arg: str | None) -> dict[str, tuple[int, float]]:
    """--channels 기본 (w,k) + --channel-params 오버라이드 병합.
    w는 항상 int(day) — build_runtag()가 window를 _numstr 없이 그대로 f-string에
    박으므로 float(30.0)을 주면 'w30.0'이 되어 실제 runtag 폴더명(w30)과 어긋난다."""
    out = {ch: (w_default, k_default) for ch in channels}
    if channel_params_arg:
        for part in channel_params_arg.split(","):
            part = part.strip()
            if not part:
                continue
            m = _CHPARAM_RE.match(part)
            if not m:
                raise SystemExit(f"[diag] --channel-params 형식 오류: '{part}' "
                                 f"(기대: chan:w{{W}}k{{K}})")
            out[m.group("chan")] = (int(float(m.group("w"))), float(m.group("k")))
    return out


def _contiguous_spans(mask: pd.Series) -> list[tuple]:
    """True 연속 구간 -> [(start, end), ...] (axvspan용)."""
    if mask.empty or not mask.any():
        return []
    idx = mask.index
    m = mask.to_numpy()
    edges = np.flatnonzero(np.diff(np.r_[0, m.view(np.int8), 0]))
    spans = []
    for i in range(0, len(edges), 2):
        s, e = edges[i], edges[i + 1] - 1
        spans.append((idx[s], idx[e]))
    return spans


def _saa_mask_from_geo(geo: pd.DataFrame | None) -> pd.Series | None:
    """geo(Bmag) -> |B|<25000nT 불리언 시리즈. geo 없음/Bmag 컬럼 없으면 None."""
    if geo is None or "Bmag" not in geo.columns:
        return None
    return (geo["Bmag"] < fsm_engine.SAA_BMAG_NT).fillna(False)


def _parse_maglat_band(spec: str) -> tuple[float, float | None] | None:
    """--maglat-band 파싱. "" -> None(기능 끔). "60" -> (60.0, None)(하한만,
    |maglat|>=60). "60:80" -> (60.0, 80.0)(60<=|maglat|<=80). 형식/범위 오류 시 SystemExit."""
    if not spec:
        return None
    parts = spec.split(":")
    if len(parts) > 2:
        raise SystemExit(f"[diag] --maglat-band 형식 오류: '{spec}' (기대: \"60\" 또는 \"60:80\")")
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        raise SystemExit(f"[diag] --maglat-band 형식 오류: '{spec}' (숫자로 파싱 불가)")
    if any(v < 0 or v > 90 for v in vals):
        raise SystemExit(f"[diag] --maglat-band 값 오류: '{spec}' (0~90 범위여야 함)")
    if len(vals) == 1:
        return (vals[0], None)
    lo, hi = vals
    if lo > hi:
        raise SystemExit(f"[diag] --maglat-band 값 오류: '{spec}' (하한 {lo:g}가 상한 {hi:g}보다 큼)")
    return (lo, hi)


def _maglat_band_label(band: tuple[float, float | None]) -> str:
    lo, hi = band
    if hi is None:
        return f"|mlat|>={lo:g}"
    return f"{lo:g}<=|mlat|<={hi:g}"


def _maglat_mask_from_geo(geo: pd.DataFrame | None,
                          band: tuple[float, float | None] | None) -> pd.Series | None:
    """geo[lat,lon] -> coords_igrf.dipole_maglat(좌표만, Bmag/IGRF 재계산 없음) -> band 불리언 마스크.
    geo 없음/lat·lon 컬럼 없으면 None."""
    if geo is None or band is None:
        return None
    if "lat" not in geo.columns or "lon" not in geo.columns:
        return None
    maglat = coords_igrf.dipole_maglat(geo["lat"].to_numpy(), geo["lon"].to_numpy())
    amag = np.abs(maglat)
    lo, hi = band
    m = (amag >= lo) if hi is None else ((amag >= lo) & (amag <= hi))
    return pd.Series(m, index=geo.index)


def _reindex_mask_to(mask: pd.Series, target_index: pd.Index) -> pd.Series:
    """geo 인덱스 마스크를 count 인덱스로 정렬. 다르면 nearest reindex(15분 허용오차), NaN은 False."""
    if mask.index.equals(target_index):
        return mask
    aligned = mask.reindex(target_index, method="nearest", tolerance=pd.Timedelta("15min"))
    return aligned.fillna(False).astype(bool)


def _polar_pass_medians(cnt: pd.Series, polar_mask: pd.Series) -> pd.Series:
    """polar_mask의 True 연속구간(pass 1회)마다 cnt median 1점 집계. 인덱스=구간 중앙 시각."""
    spans = _contiguous_spans(polar_mask)
    if not spans:
        return pd.Series(dtype=float)
    times, vals = [], []
    for s, e in spans:
        seg = cnt.loc[s:e]
        if seg.empty:
            continue
        vals.append(seg.median())
        times.append(s + (e - s) / 2)
    return pd.Series(vals, index=pd.DatetimeIndex(times)).sort_index()


def _load_catalog(catalog: str) -> pd.DataFrame:
    cache_dir, io_name = _CATALOG[catalog]
    io = core._import_event_io(io_name, str(cache_dir))
    cat_all, _ = io.load(str(cache_dir))
    return io.filter_by_date(cat_all, *core.ERA)


def _find_peak_times(detector: str, channel: str, w: int, k: float,
                     onset_floor: float, peak_floor: float | None) -> list:
    """--peak 지정 시 사전계산 fsm_event_*.csv에서 peak_time 조회 (있으면만)."""
    if peak_floor is None:
        return []
    runtag = fsm_engine.build_runtag(fsm_engine.TAG, w, k, onset_floor, peak_floor)
    ev_csv = _FSM_ROOT[detector] / runtag / f"fsm_event_{runtag}.csv"
    if not ev_csv.exists():
        print(f"[diag] {channel}: event CSV 없음({ev_csv.name}) -> peak 마킹 생략")
        return []
    ev = pd.read_csv(ev_csv, parse_dates=["peak_time"])
    return pd.to_datetime(ev.loc[ev["channel"] == channel, "peak_time"]).tolist()


def plot_one(channel: str, cnt: pd.Series, saa_mask: pd.Series | None,
            event_begin, event_max_time, event_pfu, t0, t1, out_path: Path,
            detector: str, catalog: str, fsm: dict | None = None, polar: dict | None = None):
    """fsm=None -> raw count + 카탈로그/SAA 음영만. fsm={bg,thr,onset_times,peak_times,w,k}
    -> 롤링 배경/임계값/onset/peak 오버레이 추가. polar={mask,median,band}(count 인덱스로
    정렬된 극관 마스크 + pass median 시계열) -> 극관 pass median 계단선/음영 추가."""
    win = cnt.loc[t0:t1]

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(win.index, win.values, color="black", lw=0.8, zorder=3, label="raw count")

    if polar is not None:
        pmed_win = polar["median"].loc[t0:t1]
        if not pmed_win.empty:
            ax.plot(pmed_win.index, pmed_win.values, color="blue", drawstyle="steps-mid",
                   marker="o", ms=3, lw=1.2, zorder=3.5,
                   label=f"pass median ({_maglat_band_label(polar['band'])})")

    if fsm is not None:
        bwin = fsm["bg"].loc[t0:t1]
        twin = fsm["thr"].loc[t0:t1]
        ax.plot(bwin.index, bwin["bg_median"], color="#888888", lw=1.0, zorder=2,
                label="rolling bg_median")
        ax.plot(twin.index, twin.values, color="red", ls=":", lw=1.2, zorder=2,
                label="threshold = bg_median + k*MAD")
        for i, ot in enumerate([o for o in fsm["onset_times"] if t0 <= o <= t1]):
            ax.axvline(ot, color="orange", ls="-", lw=1.3, alpha=0.85, zorder=4,
                       label="FSM onset" if i == 0 else None)
        for i, pt in enumerate([p for p in fsm["peak_times"] if t0 <= p <= t1]):
            ax.axvline(pt, color="crimson", ls="-", lw=1.3, alpha=0.85, zorder=4,
                       label="FSM peak" if i == 0 else None)

    ax.axvspan(event_begin, event_max_time, color="green", alpha=0.15, zorder=1,
              label=f"{catalog.upper()} catalog (pfu={event_pfu:.0f})")
    ax.axvline(event_begin, color="darkgreen", ls="--", lw=1.0, alpha=0.7, zorder=2,
              label="event begin")
    ax.axvline(event_max_time, color="darkgreen", ls="--", lw=1.0, alpha=0.7, zorder=2,
              label="event max_time")

    if saa_mask is not None:
        swin = saa_mask.loc[t0:t1] if not saa_mask.loc[t0:t1].empty else saa_mask.reindex([]).astype(bool)
        for i, (s, e) in enumerate(_contiguous_spans(swin)):
            ax.axvspan(s, e, color="orange", alpha=0.12, zorder=0,
                      label="SAA (|B|<25000nT)" if i == 0 else None)

    if polar is not None:
        pmask_win = polar["mask"].loc[t0:t1]
        for i, (s, e) in enumerate(_contiguous_spans(pmask_win)):
            ax.axvspan(s, e, color="tab:blue", alpha=0.08, zorder=0,
                      label=f"polar cap ({_maglat_band_label(polar['band'])})" if i == 0 else None)

    ax.set_ylabel("count rate [15-min mean]", fontsize=9)
    ax.set_yscale("log")
    if fsm is not None:
        title = (f"{channel}  detector={detector} catalog={catalog}  "
                f"w={fsm['w']} k={fsm['k']}  event={event_begin:%Y-%m-%d} pfu={event_pfu:.0f}")
    else:
        title = (f"{channel}  detector={detector} catalog={catalog}  "
                f"event={event_begin:%Y-%m-%d} pfu={event_pfu:.0f}")
    ax.set_title(title, fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[diag] saved -> {out_path}")


def run(detector: str, catalog: str, channels: list[str],
       fsm_enabled: bool, channel_params: dict[str, tuple[int, float]] | None,
       onset_floor: float, peak_floor: float | None,
       pad_before: float, pad_after: float,
       top_events: int, all_events: bool, out_dir: Path,
       maglat_band_spec: str, maglat_band: tuple[float, float | None] | None):
    io_name, cache_dir = _POES_IO[detector]
    io = core._import_event_io(io_name, str(cache_dir))

    cat = _load_catalog(catalog)
    if all_events:
        print(f"[diag] --all-events: 카탈로그 {len(cat)}개 전부 -> "
              f"채널당 그림 {len(cat)}장, 총 {len(cat) * len(channels)}장 생성 예정 (경고)")
        events = cat.sort_values("max_pfu", ascending=False)
    else:
        events = cat.sort_values("max_pfu", ascending=False).head(top_events)
    print(f"[diag] detector={detector} catalog={catalog}  이벤트 {len(events)}개 선택 "
          f"(pfu범위 {events['max_pfu'].min():.0f}~{events['max_pfu'].max():.0f})")

    # geo는 SAA 마스크와 극관 마스크가 공유 (한 번만 로드)
    geo = fsm_engine.load_geo(io, str(cache_dir))
    saa_mask = _saa_mask_from_geo(geo)
    if saa_mask is None:
        print("[diag] WARNING geo/Bmag 로드 실패 -> SAA 음영 생략")

    polar_mask_geo = None
    if maglat_band is not None:
        polar_mask_geo = _maglat_mask_from_geo(geo, maglat_band)
        if polar_mask_geo is None:
            print("[diag] WARNING geo/lat-lon 로드 실패 -> 극관 pass 오버레이 생략")

    mlat_suffix = (f"_mlat{maglat_band_spec.replace(':', '-')}"
                  if maglat_band_spec and maglat_band_spec != "60" else "")

    saved = []
    for channel in channels:
        cnt = core._load_count_channel(io, str(cache_dir), channel)
        if cnt.empty:
            print(f"[diag] {channel}: count 없음 -> skip")
            continue

        fsm_data = None
        if fsm_enabled:
            w, k = channel_params[channel]
            bg  = fsm_engine.compute_rolling_bg(cnt, int(w), None, fsm_engine.BG_UPDATE_FREQ)
            thr = fsm_engine.build_threshold(bg, k, onset_floor)
            segs = fsm_engine.detect_segments(cnt, thr, bg, fsm_engine.MIN_SPE_DURATION_H)
            onset_times = [s["onset_time"] for s in segs]
            peak_times  = _find_peak_times(detector, channel, w, k, onset_floor, peak_floor)
            print(f"[diag] {channel} w={w} k={k}: count n={len(cnt)}  onset 검출 {len(onset_times)}개"
                  + (f"  peak {len(peak_times)}개" if peak_floor is not None else ""))
            fsm_data = dict(bg=bg, thr=thr, onset_times=onset_times, peak_times=peak_times, w=w, k=k)
        else:
            print(f"[diag] {channel}: count n={len(cnt)} (raw 모드 -- FSM 계산 생략)")

        polar_data = None
        if polar_mask_geo is not None:
            pmask = _reindex_mask_to(polar_mask_geo, cnt.index)
            pmedian = _polar_pass_medians(cnt, pmask)
            polar_data = dict(mask=pmask, median=pmedian, band=maglat_band)
            print(f"[diag] {channel}: 극관 pass {len(pmedian)}개 검출 "
                  f"({_maglat_band_label(maglat_band)})")

        for begin, row in events.iterrows():
            t0 = begin - pd.Timedelta(days=pad_before)
            t1 = begin + pd.Timedelta(days=pad_after)
            if cnt.loc[t0:t1].empty:
                print(f"[diag] {channel} {begin.date()}: 창 안에 count 없음 -> skip")
                continue
            if fsm_data is not None:
                suffix = f"_w{fsm_data['w']}k{fsm_engine._numstr(fsm_data['k'])}"
            else:
                suffix = ""
            out_path = out_dir / f"{catalog}_{detector}_{channel}_{begin:%Y%m%d}{suffix}{mlat_suffix}.png"
            plot_one(channel, cnt, saa_mask, begin, row["max_time"], row["max_pfu"],
                    t0, t1, out_path, detector, catalog, fsm=fsm_data, polar=polar_data)
            saved.append(out_path)

    print(f"\n[diag] 완료: {len(saved)}장 저장 -> {out_dir}")
    return saved


def main():
    ap = argparse.ArgumentParser(
        description="POES 카탈로그 vs raw count 프리커서 육안 검사 진단 스크립트 "
                     "(--fsm 지정 시 롤링 배경/임계값/onset 오버레이 추가)")
    ap.add_argument("--detector", default="metop03", choices=list(_POES_IO))
    ap.add_argument("--catalog", default="noaa", choices=list(_CATALOG),
                    help="이벤트 카탈로그: noaa(양성자 SPE) 또는 swpc(>2MeV 전자 경보) "
                         "(기본 noaa)")
    ap.add_argument("--channels", default="pro_tel0_p5,omni_p7",
                    help="콤마 리스트 (기본 pro_tel0_p5,omni_p7)")
    ap.add_argument("--fsm", default=None,
                    help='FSM 오버레이 모드 활성화. "w{W}k{K}" 형식 문자열 (예: w30k10). '
                         '지정 시 rolling bg/threshold/onset 오버레이를 그리고 '
                         '--w/--k/--channel-params/--onset/--peak 사용이 가능해진다. '
                         '미지정(기본)이면 raw count + 카탈로그/SAA 음영만 그리는 경량 모드 '
                         '(FSM 계산 자체를 스킵).')
    ap.add_argument("--w", type=int, default=None,
                    help="채널 공통 window[day] 오버라이드 (--fsm 지정 시에만 유효)")
    ap.add_argument("--k", type=float, default=None,
                    help="채널 공통 k 오버라이드 (--fsm 지정 시에만 유효)")
    ap.add_argument("--channel-params", default=None,
                    help='채널별 (w,k) 오버라이드. 예: "pro_tel0_p5:w10k3,omni_p7:w1k7" '
                         '(--fsm 지정 시에만 유효)')
    ap.add_argument("--onset", type=float, default=None,
                    help="onset_floor 하한 클립 (기본 0=순수 bg_median+k*MAD, --fsm 지정 시에만 유효)")
    ap.add_argument("--peak", type=float, default=None,
                    help="peak_floor -- 지정하면 사전계산 fsm_event_*.csv에서 "
                         "peak_time을 찾아 빨강 세로선으로 표시 (--fsm 지정 시에만 유효)")
    ap.add_argument("--pad-before", type=float, default=10.0,
                    help="이벤트 begin 이전 표시 범위[day] (기본 10)")
    ap.add_argument("--pad-after", type=float, default=5.0,
                    help="이벤트 begin 이후 표시 범위[day] (기본 5)")
    ap.add_argument("--top-events", type=int, default=5,
                    help="PFU 상위 N개 이벤트만 (기본 5)")
    ap.add_argument("--all-events", action="store_true",
                    help="카탈로그 전체 이벤트 -- 그림 수 폭발 경고 출력")
    ap.add_argument("--maglat-band", default="60",
                    help='극관 pass median 오버레이 밴드 (raw/--fsm 공통, 기본 "60"). '
                         '"60" -> |maglat|>=60 (하한만), "60:80" -> 60<=|maglat|<=80. '
                         '빈 문자열("")이면 오버레이를 끈다. 기본값("60") 이외 지정 시 '
                         '파일명에 "_mlat{spec}" suffix 추가(콜론은 "-"로 치환).')
    ap.add_argument("--out", default=str(_DEFAULT_OUT))
    args = ap.parse_args()

    fsm_enabled = args.fsm is not None
    if not fsm_enabled:
        locked = {"--w": args.w, "--k": args.k, "--channel-params": args.channel_params,
                  "--onset": args.onset, "--peak": args.peak}
        bad = [name for name, v in locked.items() if v is not None]
        if bad:
            raise SystemExit(f"[diag] {', '.join(bad)}는 --fsm 지정 시에만 유효합니다 "
                             f"(예: --fsm w30k10). raw 모드에서는 사용할 수 없습니다.")

    channels = [c.strip() for c in args.channels.split(",") if c.strip()]

    channel_params = None
    onset_floor = 0.0
    if fsm_enabled:
        w_default, k_default = _parse_fsm_arg(args.fsm)
        if args.w is not None:
            w_default = args.w
        if args.k is not None:
            k_default = args.k
        onset_floor = args.onset if args.onset is not None else 0.0
        channel_params = _parse_channel_params(channels, w_default, k_default, args.channel_params)

    maglat_band = _parse_maglat_band(args.maglat_band)

    run(args.detector, args.catalog, channels, fsm_enabled, channel_params,
       onset_floor, args.peak, args.pad_before, args.pad_after,
       args.top_events, args.all_events, Path(args.out),
       args.maglat_band, maglat_band)


if __name__ == "__main__":
    main()
