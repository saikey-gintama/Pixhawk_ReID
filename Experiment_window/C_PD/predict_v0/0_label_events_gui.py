"""
0_label_events_gui.py
====================
수동 이벤트 라벨링 GUI -- FSM/카탈로그가 놓친 실제 이벤트에 onset/peak/end를
사람이 직접 표시해 예측 모델용 ground-truth를 만든다. 채널 1개씩 개별 라벨링
(전 채널 동시 라벨링 안 함 -- 나중에 채널별 비교를 위해 필요).

재사용 (재구현 없음 -- import만):
  _match_core_poes.py                 : _import_event_io, _load_count_channel (POES count 로드)
  fsm_count_spe_quietoff_mad_poes.py  : load_geo/tag_onset_geo (POES geo 태깅. GK2A는
      geo=None으로 같은 함수 재사용 -- tag_onset_geo가 geo=None을 이미 NaN dict로 처리),
      compute_rolling_bg/build_threshold/detect_segments (--show-fsm 참고선용, KSEM과
      byte-identical한 범용 엔진이라 GK2A 채널에도 그대로 적용 가능)
  ksem_io.py                          : load/get_series (GK2A count 로드)

GK2A는 이 코드베이스 어디에도 geo/IGRF 모듈이 없음(정지궤도라 LEO식 SAA/maglat
개념 자체가 안 맞기도 함) -- maglat/Bmag/in_saa 열은 GK2A일 때 전부 NaN.

화면: 단일 채널 raw count(로그축) + --window-days(기본 4일) 슬라이딩 창(좌우
화살표=1일 이동, xlim만 바꿔 빠름 -- 전체 재그리기 안 함) + 라벨 세로선
(onset=초록/peak=빨강/end=파랑, event_id 주석) + (--show-fsm 지정 시 회색 참고선).

조작:
  그래프 빈 공간 클릭   -> 15분 격자로 스냅된 '대기 시각' 지정
  o                     -> 대기 시각에 onset 기록(새 event_id 발번, 활성 이벤트로 지정)
  p / e                 -> 대기 시각에 peak/end 기록(활성 event_id 귀속).
                           같은 event_id+종류가 이미 있으면 덮어씀(재라벨링).
  기존 세로선 클릭       -> 선택(굵게 표시) + 그 event_id를 활성 이벤트로 전환
                           (이어작업 때 기존 이벤트에 peak/end를 추가하기 위함)
  u                     -> 선택된 라벨(없으면 활성 이벤트의 onset)의 note를
                           ''<->'unsure' 토글
  Shift+Left/Right      -> 선택된 라벨 ±15분 이동
  Delete/Backspace      -> 선택된 라벨 삭제
  Left/Right            -> 보이는 창 ±1일 이동 (--window-days 폭 유지)
  s                     -> 수동 저장(+ 모든 조작 후 자동저장도 항상 수행)

출력: {out-dir}/manual_labels_{detector}_{channel}.csv
  컬럼: event_id,label_type,time,count,maglat,Bmag,in_saa,detector,channel,note
  기존 파일 있으면 시작 시 로드(이어작업). event_id 당 o/p/e 미완성이면 제목에
  incomplete 카운트로 경고 표시(막지는 않음).

사용 (predict_v0/ 안에서 실행):
  python 0_label_events_gui.py --detector metop03 --channel omni_p6
  python 0_label_events_gui.py --detector gk2a --channel PD1B-OU --show-fsm w7k7
  python 0_label_events_gui.py --detector metop03 --channel omni_p6 --window-days 4 --out-dir predict_v0/manual_labels
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent   # C_PD/predict_v0/
C_PD = HERE.parent                        # C_PD/ -- POES/GK2A 등 공용 모듈 위치

sys.path.insert(0, str(C_PD / "POES"))
sys.path.insert(0, str(C_PD / "POES" / "event_MATCHER"))
sys.path.insert(0, str(C_PD / "POES" / "count_FSM"))
sys.path.insert(0, str(C_PD / "GK2A" / "KSEM_count"))
import _match_core_poes as core                      # _import_event_io, _load_count_channel
import fsm_count_spe_quietoff_mad_poes as fsm_engine  # load_geo/tag_onset_geo + 롤링배경 엔진(--show-fsm)
import ksem_io                                        # GK2A count 로드

# core가 import 시점에 matplotlib.use("Agg")를 이미 고정하므로(_match_core_poes.py),
# 여기서는 그 이후에 pyplot을 가져오고 main()에서 --backend로 다시 전환한다
# (matplotlib.pyplot.switch_backend는 사후 전환을 위해 만들어진 공식 메커니즘).
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

_POES_IO = {
    "metop03": ("poes_metop03_io", C_PD / "POES" / "MetOp03_count" / "poes_metop03_cache_parquet"),
    "noaa19":  ("poes_noaa19_io",  C_PD / "POES" / "NOAA19_count"  / "poes_noaa19_cache_parquet"),
}
_GK2A_CACHE = C_PD / "GK2A" / "KSEM_count" / "ksem_cache_parquet"

_DEFAULT_OUT_DIR = HERE / "manual_labels"  # predict_v0/manual_labels/
_CSV_COLS = ["event_id", "label_type", "time", "count", "maglat", "Bmag", "in_saa",
            "detector", "channel", "note"]
_LABEL_COLORS = {"o": "green", "p": "red", "e": "blue"}
_TYPE_ORDER = {"o": 0, "p": 1, "e": 2}
_SELECT_TOL_FRAC = 0.01   # 보이는 창 폭의 1% 이내 클릭이면 기존 세로선 '선택'으로 처리

# --show-fsm "w{W}k{K}" 파서. diag_rolling_threshold_poes._parse_fsm_arg와 로직은
# 같지만 그 모듈 자체는 import하지 않음(무관한 별개 진단 스크립트를 통째로 끌어오지
# 않기 위함 -- 실제 엔진 로직은 위에서 core/fsm_engine/ksem_io를 그대로 재사용 중).
_WK_RE = re.compile(r"^w(?P<w>[\d.]+)k(?P<k>[\d.]+)$")


def _parse_fsm_arg(spec: str) -> tuple[int, float]:
    m = _WK_RE.match(spec.strip())
    if not m:
        raise SystemExit(f"[label] --show-fsm 형식 오류: '{spec}' (기대: w{{W}}k{{K}}, 예: w7k7)")
    return int(float(m.group("w"))), float(m.group("k"))


def _gk2a_channel_to_tuple(channel: str) -> tuple[str, str, str]:
    """'PD1B-OU' -> ('PD1','B','OU'). f'{pd_key}{side}-{logic}' 조립 규칙의 역변환
    (fsm_count_spe_quietoff_std.py/4_summarize.py의 채널명 관례와 동일)."""
    if "-" not in channel:
        raise SystemExit(f"[label] GK2A 채널 형식 오류: '{channel}' (기대: 'PD1B-OU' 형태)")
    prefix, logic = channel.split("-", 1)
    if len(prefix) < 2:
        raise SystemExit(f"[label] GK2A 채널 형식 오류: '{channel}'")
    pd_key, side = prefix[:-1], prefix[-1]
    return pd_key, side, logic


def load_channel_series(detector: str, channel: str):
    """(count 15분 Series, geo DataFrame|None) 반환. POES는 기존 재사용 모듈 그대로,
    GK2A는 geo 없음(코드베이스에 GK2A geo 모듈 자체가 없음)."""
    if detector in _POES_IO:
        io_name, cache_dir = _POES_IO[detector]
        io = core._import_event_io(io_name, str(cache_dir))
        cnt = core._load_count_channel(io, str(cache_dir), channel)
        geo = fsm_engine.load_geo(io, str(cache_dir))
        return cnt, geo
    if detector == "gk2a":
        df_count, _ = ksem_io.load(str(_GK2A_CACHE))
        if not df_count.empty and df_count.index.tz is None:
            df_count.index = df_count.index.tz_localize("UTC")
        pd_key, side, logic = _gk2a_channel_to_tuple(channel)
        try:
            cnt = ksem_io.get_series(df_count, pd_key, side, logic).dropna()
        except KeyError:
            cnt = pd.Series(dtype=float)
        cnt = cnt.resample("15min").mean().dropna()
        return cnt, None
    raise SystemExit(f"[label] unknown --detector {detector}")


def compute_fsm_onsets(cnt: pd.Series, w: int, k: float) -> list:
    """--show-fsm 회색 참고선용 onset 시각 리스트. onset_floor=0 고정(단순 참고용)."""
    bg = fsm_engine.compute_rolling_bg(cnt, w, None, fsm_engine.BG_UPDATE_FREQ)
    thr = fsm_engine.build_threshold(bg, k, 0.0)
    segs = fsm_engine.detect_segments(cnt, thr, bg, fsm_engine.MIN_SPE_DURATION_H)
    return [s["onset_time"] for s in segs]


class LabelGUI:
    """단일 채널 수동 라벨링 상태 + matplotlib 인터랙티브 화면.

    콜백(_on_click/_on_key)은 얇은 디스패처일 뿐 -- 실제 로직은 _add_label 등
    별도 메서드에 있어 matplotlib 이벤트 없이도(헤드리스 테스트) 직접 호출/검증 가능.
    """

    def __init__(self, detector, channel, cnt, geo, out_path, window_days, fsm_onsets=None):
        self.detector = detector
        self.channel = channel
        self.cnt = cnt
        self.geo = geo
        self.out_path = Path(out_path)
        self.window_days = window_days
        self.fsm_onsets = fsm_onsets or []

        self.labels: list[dict] = []
        self.next_event_id = 1
        self.active_event_id = None
        self.selected_idx = None
        self.pending_time = None
        self._label_artists = []

        self._load_existing()

        if self.labels:
            center = max(r["time"] for r in self.labels)
            self.t0 = center - pd.Timedelta(days=window_days / 2)
        else:
            self.t0 = cnt.index[0]
        self.t1 = self.t0 + pd.Timedelta(days=window_days)

        self._build_figure()
        self._connect_events()
        self._redraw_labels()

    # ── 데이터 조회 헬퍼 ────────────────────────────────────────
    def _snap15(self, t) -> pd.Timestamp:
        return pd.Timestamp(t).round("15min")

    def _count_at(self, t) -> float:
        if self.cnt.empty:
            return float("nan")
        pos = self.cnt.index.get_indexer([t], method="nearest", tolerance=pd.Timedelta(minutes=7.5))[0]
        if pos < 0:
            return float("nan")
        return float(self.cnt.iloc[pos])

    def _geo_at(self, t) -> dict:
        d = fsm_engine.tag_onset_geo(t, self.geo)
        return {"maglat": d["onset_maglat"], "Bmag": d["onset_Bmag"], "in_saa": d["in_saa"]}

    # ── 영속화 ──────────────────────────────────────────────────
    def _load_existing(self):
        if not self.out_path.exists():
            return
        df = pd.read_csv(self.out_path, parse_dates=["time"])
        if len(df) and df["time"].dt.tz is None:
            df["time"] = df["time"].dt.tz_localize("UTC")
        self.labels = df.to_dict("records")
        for r in self.labels:
            if pd.isna(r.get("note")):
                r["note"] = ""
        if self.labels:
            self.next_event_id = max(int(r["event_id"]) for r in self.labels) + 1
        n_events = len({r["event_id"] for r in self.labels})
        print(f"[label] 기존 라벨 {len(self.labels)}행({n_events}개 이벤트) 로드 <- {self.out_path}")

    def _save(self):
        rows = sorted(self.labels, key=lambda r: (r["event_id"], _TYPE_ORDER.get(r["label_type"], 9)))
        df = pd.DataFrame(rows, columns=_CSV_COLS)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.out_path, index=False)

    # ── 라벨 조작 (matplotlib 이벤트 없이 직접 호출 가능) ─────────
    def _add_label(self, label_type: str):
        if self.pending_time is None:
            print("[label] 먼저 그래프를 클릭해 시각을 지정하세요.")
            return
        t = self.pending_time
        if label_type == "o":
            eid = self.next_event_id
            self.next_event_id += 1
            self.active_event_id = eid
        else:
            if self.active_event_id is None:
                print(f"[label] '{label_type}' 라벨을 붙일 활성 이벤트가 없습니다 -- "
                      f"먼저 'o'로 새 이벤트를 만들거나 기존 onset 세로선을 클릭해 선택하세요.")
                return
            eid = self.active_event_id
        row = {
            "event_id": eid, "label_type": label_type, "time": t,
            "count": self._count_at(t), **self._geo_at(t),
            "detector": self.detector, "channel": self.channel, "note": "",
        }
        existing = [i for i, r in enumerate(self.labels)
                    if r["event_id"] == eid and r["label_type"] == label_type]
        if existing:
            row["note"] = self.labels[existing[0]].get("note", "")
            self.labels[existing[0]] = row
        else:
            self.labels.append(row)
        self.pending_time = None
        self._redraw_labels()
        self._save()

    def _select_nearest_line(self, x_time: pd.Timestamp):
        if not self.labels:
            return None
        tol = (self.t1 - self.t0) * _SELECT_TOL_FRAC
        best_i, best_dt = None, None
        for i, r in enumerate(self.labels):
            dt = abs(r["time"] - x_time)
            if dt <= tol and (best_dt is None or dt < best_dt):
                best_i, best_dt = i, dt
        return best_i

    def _move_selected(self, minutes: float):
        if self.selected_idx is None:
            print("[label] 이동할 선택된 라벨이 없습니다.")
            return
        r = self.labels[self.selected_idx]
        new_t = r["time"] + pd.Timedelta(minutes=minutes)
        r["time"] = new_t
        r["count"] = self._count_at(new_t)
        r.update(self._geo_at(new_t))
        self._redraw_labels()
        self._save()

    def _delete_selected(self):
        if self.selected_idx is None:
            print("[label] 삭제할 선택된 라벨이 없습니다.")
            return
        del self.labels[self.selected_idx]
        self.selected_idx = None
        self._redraw_labels()
        self._save()

    def _toggle_unsure(self):
        if self.selected_idx is not None:
            r = self.labels[self.selected_idx]
        elif self.active_event_id is not None:
            cand = [r for r in self.labels
                    if r["event_id"] == self.active_event_id and r["label_type"] == "o"]
            if not cand:
                print("[label] unsure 토글 대상이 없습니다.")
                return
            r = cand[0]
        else:
            print("[label] unsure 토글 대상이 없습니다 -- 라벨을 선택하거나 활성 이벤트를 만드세요.")
            return
        r["note"] = "" if r.get("note") == "unsure" else "unsure"
        self._redraw_labels()
        self._save()

    def _pan(self, days: float):
        self.t0 = self.t0 + pd.Timedelta(days=days)
        self.t1 = self.t1 + pd.Timedelta(days=days)
        self.ax.set_xlim(self.t0, self.t1)
        self._update_title()
        self.fig.canvas.draw_idle()

    # ── 렌더링 ──────────────────────────────────────────────────
    def _build_figure(self):
        self.fig, self.ax = plt.subplots(figsize=(13, 5))
        self.ax.plot(self.cnt.index, self.cnt.values, color="black", lw=0.6, zorder=1)
        self.ax.set_yscale("log")
        self.ax.set_ylabel(f"{self.channel} count [15-min mean]", fontsize=9)
        self.ax.grid(True, alpha=0.3, which="both")
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        for onset_t in self.fsm_onsets:
            self.ax.axvline(onset_t, color="gray", lw=0.6, alpha=0.35, zorder=0)
        self.ax.set_xlim(self.t0, self.t1)
        self.fig.text(
            0.01, 0.01,
            "click=set time | o/p/e=label | u=unsure | s=save | "
            "left/right=pan 1d | shift+left/right=move 15min | delete=remove selected",
            fontsize=7, color="gray")
        self.fig.tight_layout(rect=(0, 0.03, 1, 1))

    def _redraw_labels(self):
        for art in self._label_artists:
            art.remove()
        self._label_artists = []
        for i, r in enumerate(self.labels):
            selected = (i == self.selected_idx)
            lw = 2.4 if selected else 1.2
            color = _LABEL_COLORS.get(r["label_type"], "black")
            ln = self.ax.axvline(r["time"], color=color, lw=lw, alpha=0.9, zorder=5)
            self._label_artists.append(ln)
            if r["label_type"] == "o":
                txt = self.ax.annotate(str(r["event_id"]), (r["time"], 1.0),
                                       xycoords=("data", "axes fraction"),
                                       fontsize=7, color="green", ha="center", va="bottom")
                self._label_artists.append(txt)
            if r.get("note") == "unsure":
                mark = self.ax.axvline(r["time"], color="black", lw=lw + 2, alpha=0.15, zorder=4)
                self._label_artists.append(mark)
        self._update_title()
        self.fig.canvas.draw_idle()

    def _update_title(self):
        n_events = len({r["event_id"] for r in self.labels})
        by_eid: dict = {}
        for r in self.labels:
            by_eid.setdefault(r["event_id"], set()).add(r["label_type"])
        n_incomplete = sum(1 for types in by_eid.values() if types != {"o", "p", "e"})
        warn = f"  [WARN incomplete={n_incomplete}]" if n_incomplete else ""
        self.ax.set_title(
            f"{self.detector}/{self.channel}   {self.t0:%Y-%m-%d %H:%M}~{self.t1:%Y-%m-%d %H:%M}"
            f"   events={n_events}{warn}", fontsize=10)

    # ── matplotlib 콜백 (얇은 디스패처) ────────────────────────────
    def _xdata_to_time(self, xdata: float) -> pd.Timestamp:
        t = pd.Timestamp(mdates.num2date(xdata))
        return t.tz_convert("UTC") if t.tzinfo is not None else t.tz_localize("UTC")

    def _connect_events(self):
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        x_time = self._xdata_to_time(event.xdata)
        idx = self._select_nearest_line(x_time)
        if idx is not None:
            self.selected_idx = idx
            self.active_event_id = self.labels[idx]["event_id"]
            self.pending_time = None
        else:
            self.selected_idx = None
            self.pending_time = self._snap15(x_time)
        self._redraw_labels()

    def _on_key(self, event):
        k = event.key
        if k == "o":
            self._add_label("o")
        elif k == "p":
            self._add_label("p")
        elif k == "e":
            self._add_label("e")
        elif k == "u":
            self._toggle_unsure()
        elif k == "s":
            self._save()
            print("[label] 저장 완료(수동 s)")
        elif k == "left":
            self._pan(-1)
        elif k == "right":
            self._pan(1)
        elif k == "shift+left":
            self._move_selected(-15)
        elif k == "shift+right":
            self._move_selected(15)
        elif k in ("delete", "backspace"):
            self._delete_selected()

    def show(self):
        plt.show()


def main():
    ap = argparse.ArgumentParser(description="수동 이벤트 라벨링 GUI (onset/peak/end)")
    ap.add_argument("--detector", required=True, choices=["metop03", "noaa19", "gk2a"])
    ap.add_argument("--channel", required=True, help="단일 채널 (예: omni_p6, PD1B-OU)")
    ap.add_argument("--window-days", type=float, default=4.0)
    ap.add_argument("--show-fsm", default=None,
                    help='FSM 참고선(회색) 표시. "w{W}k{K}" 형식(예: w7k7). onset_floor=0 고정.')
    ap.add_argument("--out-dir", default=str(_DEFAULT_OUT_DIR))
    ap.add_argument("--backend", default="TkAgg",
                    help="matplotlib 백엔드(기본 TkAgg -- 로컬 인터랙티브 창). "
                         "tkinter 없으면 오류 -- 그 경우 다른 GUI 백엔드 지정.")
    args = ap.parse_args()

    try:
        plt.switch_backend(args.backend)
    except Exception as e:
        raise SystemExit(f"[label] 백엔드 '{args.backend}' 전환 실패: {e}")

    cnt, geo = load_channel_series(args.detector, args.channel)
    if cnt.empty:
        raise SystemExit(f"[label] ERROR: {args.detector}/{args.channel} count 데이터 없음 -- 채널명 확인.")
    print(f"[label] {args.detector}/{args.channel} count 로드 완료 "
          f"(n={len(cnt)}, {cnt.index[0]}~{cnt.index[-1]})")
    if geo is None and args.detector in _POES_IO:
        print("[label] WARN geo 로드 실패 -- maglat/Bmag/in_saa 전부 NaN으로 기록됨.")
    elif args.detector == "gk2a":
        print("[label] GK2A: geo 모듈 없음(정지궤도) -- maglat/Bmag/in_saa 전부 NaN으로 기록됨.")

    fsm_onsets = []
    if args.show_fsm:
        w, k = _parse_fsm_arg(args.show_fsm)
        fsm_onsets = compute_fsm_onsets(cnt, w, k)
        print(f"[label] --show-fsm {args.show_fsm}: 참고선 {len(fsm_onsets)}개")

    out_path = Path(args.out_dir) / f"manual_labels_{args.detector}_{args.channel}.csv"
    gui = LabelGUI(args.detector, args.channel, cnt, geo, out_path, args.window_days, fsm_onsets)
    gui.show()


if __name__ == "__main__":
    main()
