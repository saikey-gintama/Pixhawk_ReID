"""
diag_rolling_threshold_poes.py
===============================
POES quietoff_mad 롤링 배경 임계값이 실제 이벤트에서 어떻게 동작하는지
raw count와 겹쳐 눈으로 검증하는 진단 스크립트. "롤링배경이 POES에서
성공했나"에 답하는 그림을 채널 x 이벤트 조합마다 1장씩 만든다.

재사용 (재구현 없음 — import만):
  fsm_count_spe_quietoff_mad_poes.py : compute_rolling_bg / detect_segments /
      load_count / load_geo (롤링 배경 엔진 그대로)
  _match_core_poes.py                : _load_count_channel, _import_event_io
  noaa_goes_spe_io                   : load, filter_by_date
  coords_igrf                        : load_geo(with_bmag=True) 내부에서 이미 호출됨
                                        (poes_*_io.get_geo(with_bmag=True))

그림 1장 = 채널 1개 x 이벤트 1개.
  검은 실선   : raw count
  회색 실선   : rolling bg_median(t)
  빨강 점선   : threshold(t) = bg_median + k*MAD  (onset_floor=0 기본, 순수 공식)
  주황 세로선 : FSM onset (창 안, 라이브로 detect_segments 재계산)
  빨강 세로선 : FSM peak (사전계산 fsm_event_*.csv 있으면만 — --peak 지정 시)
  옅은 초록 음영: --catalog 이벤트 카탈로그(기본 noaa)의 이벤트 begin~max_time
  옅은 주황 음영: SAA 구간 (|B|<25000nT)

사용:
  python diag_rolling_threshold_poes.py --channels pro_tel0_p5,omni_p7 --w 30 --k 10 --top-events 5
  python diag_rolling_threshold_poes.py \
      --channel-params "pro_tel0_p5:w10k3,omni_p7:w1k7" --top-events 3
  python diag_rolling_threshold_poes.py --channels pro_tel0_p5 --w 10 --k 3 --all-events
  python diag_rolling_threshold_poes.py --detector metop03 --catalog swpc --channels omni_p7 --top-events 3
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

sys.path.insert(0, str(HERE / "POES" / "event_MATCHER"))
sys.path.insert(0, str(HERE / "POES" / "count_FSM"))
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

_CHPARAM_RE = re.compile(r"^(?P<chan>[^:]+):w(?P<w>[\d.]+)k(?P<k>[\d.]+)$")


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


def _load_saa_mask(io, cache_dir: str) -> pd.Series | None:
    """geo(Bmag) 로드 -> |B|<25000nT 불리언 시리즈. 실패 시 None."""
    geo = fsm_engine.load_geo(io, str(cache_dir))
    if geo is None or "Bmag" not in geo.columns:
        return None
    return (geo["Bmag"] < fsm_engine.SAA_BMAG_NT).fillna(False)


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


def plot_one(channel: str, cnt: pd.Series, bg: pd.DataFrame, thr: pd.Series,
            onset_times: list, peak_times: list, saa_mask: pd.Series | None,
            event_begin, event_max_time, event_pfu, t0, t1, out_path: Path,
            w: float, k: float, detector: str, catalog: str):
    win = cnt.loc[t0:t1]
    bwin = bg.loc[t0:t1]
    twin = thr.loc[t0:t1]

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(win.index, win.values, color="black", lw=0.8, zorder=3, label="raw count")
    ax.plot(bwin.index, bwin["bg_median"], color="#888888", lw=1.0, zorder=2,
            label="rolling bg_median")
    ax.plot(twin.index, twin.values, color="red", ls=":", lw=1.2, zorder=2,
            label="threshold = bg_median + k*MAD")

    for i, ot in enumerate([o for o in onset_times if t0 <= o <= t1]):
        ax.axvline(ot, color="orange", ls="-", lw=1.3, alpha=0.85, zorder=4,
                   label="FSM onset" if i == 0 else None)
    for i, pt in enumerate([p for p in peak_times if t0 <= p <= t1]):
        ax.axvline(pt, color="crimson", ls="-", lw=1.3, alpha=0.85, zorder=4,
                   label="FSM peak" if i == 0 else None)

    ax.axvspan(event_begin, event_max_time, color="green", alpha=0.15, zorder=1,
              label=f"{catalog.upper()} catalog (pfu={event_pfu:.0f})")

    if saa_mask is not None:
        swin = saa_mask.loc[t0:t1] if not saa_mask.loc[t0:t1].empty else saa_mask.reindex([]).astype(bool)
        for i, (s, e) in enumerate(_contiguous_spans(swin)):
            ax.axvspan(s, e, color="orange", alpha=0.12, zorder=0,
                      label="SAA (|B|<25000nT)" if i == 0 else None)

    ax.set_ylabel("count rate [15-min mean]", fontsize=9)
    ax.set_yscale("log")
    ax.set_title(f"{channel}  detector={detector} catalog={catalog}  "
                f"w={w} k={k}  event={event_begin:%Y-%m-%d} pfu={event_pfu:.0f}",
                fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[diag] saved -> {out_path}")


def run(detector: str, catalog: str, channel_params: dict[str, tuple[float, float]],
       onset_floor: float, peak_floor: float | None, pad_days: float,
       top_events: int, all_events: bool, out_dir: Path):
    io_name, cache_dir = _POES_IO[detector]
    io = core._import_event_io(io_name, str(cache_dir))

    cat = _load_catalog(catalog)
    if all_events:
        print(f"[diag] --all-events: 카탈로그 {len(cat)}개 전부 -> "
              f"채널당 그림 {len(cat)}장, 총 {len(cat) * len(channel_params)}장 생성 예정 (경고)")
        events = cat.sort_values("max_pfu", ascending=False)
    else:
        events = cat.sort_values("max_pfu", ascending=False).head(top_events)
    print(f"[diag] detector={detector} catalog={catalog}  이벤트 {len(events)}개 선택 "
          f"(pfu범위 {events['max_pfu'].min():.0f}~{events['max_pfu'].max():.0f})")

    saa_mask = _load_saa_mask(io, str(cache_dir))
    if saa_mask is None:
        print("[diag] WARNING geo/Bmag 로드 실패 -> SAA 음영 생략")

    saved = []
    for channel, (w, k) in channel_params.items():
        cnt = core._load_count_channel(io, str(cache_dir), channel)
        if cnt.empty:
            print(f"[diag] {channel}: count 없음 -> skip")
            continue
        bg  = fsm_engine.compute_rolling_bg(cnt, int(w), None, fsm_engine.BG_UPDATE_FREQ)
        thr = fsm_engine.build_threshold(bg, k, onset_floor)
        segs = fsm_engine.detect_segments(cnt, thr, bg, fsm_engine.MIN_SPE_DURATION_H)
        onset_times = [s["onset_time"] for s in segs]
        peak_times  = _find_peak_times(detector, channel, w, k, onset_floor, peak_floor)
        print(f"[diag] {channel} w={w} k={k}: count n={len(cnt)}  onset 검출 {len(onset_times)}개"
              + (f"  peak {len(peak_times)}개" if peak_floor is not None else ""))

        for begin, row in events.iterrows():
            t0 = begin - pd.Timedelta(days=pad_days)
            t1 = begin + pd.Timedelta(days=pad_days)
            if cnt.loc[t0:t1].empty:
                print(f"[diag] {channel} {begin.date()}: 창 안에 count 없음 -> skip")
                continue
            out_path = out_dir / f"{detector}_{catalog}_{channel}_{begin:%Y%m%d}.png"
            plot_one(channel, cnt, bg, thr, onset_times, peak_times, saa_mask,
                    begin, row["max_time"], row["max_pfu"], t0, t1, out_path, w, k,
                    detector, catalog)
            saved.append(out_path)

    print(f"\n[diag] 완료: {len(saved)}장 저장 -> {out_dir}")
    return saved


def main():
    ap = argparse.ArgumentParser(
        description="POES quietoff_mad 롤링 배경 임계값 시각 검증 진단 스크립트")
    ap.add_argument("--detector", default="metop03", choices=list(_POES_IO))
    ap.add_argument("--catalog", default="noaa", choices=list(_CATALOG),
                    help="이벤트 카탈로그: noaa(양성자 SPE) 또는 swpc(>2MeV 전자 경보) "
                         "(기본 noaa)")
    ap.add_argument("--channels", default="pro_tel0_p5,omni_p7",
                    help="콤마 리스트 (기본 pro_tel0_p5,omni_p7)")
    ap.add_argument("--w", type=int, default=30, help="채널 공통 window[day] (기본 30)")
    ap.add_argument("--k", type=float, default=10.0, help="채널 공통 k (기본 10)")
    ap.add_argument("--channel-params", default=None,
                    help='채널별 (w,k) 오버라이드. 예: "pro_tel0_p5:w10k3,omni_p7:w1k7"')
    ap.add_argument("--onset", type=float, default=0.0,
                    help="onset_floor 하한 클립 (기본 0=순수 bg_median+k*MAD)")
    ap.add_argument("--peak", type=float, default=None,
                    help="peak_floor -- 지정하면 사전계산 fsm_event_*.csv에서 "
                         "peak_time을 찾아 빨강 세로선으로 표시 (없으면 생략)")
    ap.add_argument("--pad-days", type=float, default=10.0,
                    help="이벤트 begin 전후 표시 범위[day] (기본 10)")
    ap.add_argument("--top-events", type=int, default=5,
                    help="PFU 상위 N개 이벤트만 (기본 5)")
    ap.add_argument("--all-events", action="store_true",
                    help="카탈로그 전체 이벤트 (42개) -- 그림 수 폭발 경고 출력")
    ap.add_argument("--out", default=str(_DEFAULT_OUT))
    args = ap.parse_args()

    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    channel_params = _parse_channel_params(channels, args.w, args.k, args.channel_params)

    run(args.detector, args.catalog, channel_params, args.onset, args.peak, args.pad_days,
       args.top_events, args.all_events, Path(args.out))


if __name__ == "__main__":
    main()
