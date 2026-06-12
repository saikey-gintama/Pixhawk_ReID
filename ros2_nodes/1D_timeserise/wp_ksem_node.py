"""
ksem_node.py  —  cFS LC_watch (WP 층)
=====================================
가상 관측기기(KSEM count parquet/CSV)에서 데이터를 cadence로 읽어
**채널별 임계 판정(rolling MAD)**까지 수행하고 결과를 발행한다.

cFS 매핑: LC_watch (Watchpoint).
  NASA LC 는  if (count > ComparisonValue 고정상수)  로 판정한다.
  본 노드는 그 한 줄만  if (count > bg_median + K*sigma)  로 교체한다.   ★기여점
  나머지(STALE 처리, FALSE→TRUE 전이 기록)는 LC 구조를 그대로 따른다.

발행:
  /wp_results (String JSON):
    {
      "ts": <unix_sec 물리시각>,
      "results": [
        {"channel": "PD3A-OU", "watch": "TRUE"/"FALSE"/"STALE"/"ERROR",
         "count": 1234.5, "threshold": 456.7,
         "bg": 300.0, "sigma": 15.6,
         "onset_ts": <FALSE→TRUE 전이 unix_sec 또는 null>},
        ...
      ]
    }

설계 근거 (lc_watch.c / lc_action.c 정독):
  - NaN 배경(cold-start, 결손) → STALE  (LC_FloatCompare: NaN 은 비교 안 함)
  - count 가 inf/overflow → TRUE        (LC_FloatCompare: inf 는 위반으로 둠)
  - FALSE→TRUE 전이 시각 = onset 시각    (LC_ProcessWP: LastFalseToTrue.Timestamp)
  - 연속 카운트(persistence)는 fsm_node(AP)가 담당. 여기선 채널 boolean 까지만.

파라미터(상단 블록만 수정):
  DATA_PATH, CHANNELS, CADENCE_SEC, REPLAY_SPEED
  BG_WINDOW_DAYS, K, ONSET_FLOOR
"""

from __future__ import annotations
import os
import sys
import json
import time
import math
import argparse
from pathlib import Path
from collections import deque
from time import perf_counter

import numpy as np
import pandas as pd
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

RESULT_DIR = os.environ.get("RESULT_DIR", ".")

# ══════════════════════════════════════════════════════
# 파라미터 블록 — 기본값(argparse 로 런타임 덮어쓰기, cFS TBL 철학)
#   변경 가능 인자: --window --k --replay-speed --onset-floor --data
# ══════════════════════════════════════════════════════
DATA_PATH = Path("ksem_cache_parquet")    # parquet 디렉토리 또는 .csv 경로

# voting 4채널 (확정): PD3A-OU, PD3A-OUT, PD1A-OUT, PD2A-OU
CHANNELS: list[tuple] = [                  # (pd_key, side, logic)
    ("PD3", "A", "OU"),
    ("PD3", "A", "OUT"),
    ("PD1", "A", "OUT"),
    ("PD2", "A", "OU"),
]

CADENCE_SEC   = 900        # 실제 위성 cadence (15min) — 고정
REPLAY_SPEED  = 60.0       # active 구간 배속. 1.0=실시간. 측정은 1800 권장.
WARMUP_SPEED  = 7200.0     # warm-up(배경 버퍼 채우기) 배속. 측정 불필요라 초고속.
EVENT_TIME: str | None = None  # 이 시각 이후 WARMUP_SPEED→REPLAY_SPEED 전환
                               # (None 이면 전체 REPLAY_SPEED 단일 배속)

# WP 임계 파라미터 (rolling MAD)
BG_WINDOW_DAYS = 30        # rolling 배경 윈도우
K              = 10        # bg_median + K*sigma  (proton; electron 테스트 시 20)
ONSET_FLOOR    = 0.5       # 임계 하한 (proton=0.5, electron=0.0)

# 24h 보관 (DS 역할) — 이벤트 시 스냅샷용
KEEP_24H       = True

# 리플레이 구간 슬라이스 (None = 전체). argparse --start/--end 로 주입.
# 주의: rolling 배경은 trailing W일이 필요하므로, START 는 첫 이벤트보다
# 최소 W일 앞이어야(warm-up) 트리거가 cold-start 를 벗어난다. 러너가 자동 처리.
START_DATE: str | None = None
END_DATE:   str | None = None
# ══════════════════════════════════════════════════════

# 파생 상수 — argparse 후 _recompute_derived() 로 재계산됨
CADENCE_MIN   = CADENCE_SEC // 60                  # 15
PTS_PER_DAY   = 24 * 60 // CADENCE_MIN             # 96
BG_WINDOW_PTS = BG_WINDOW_DAYS * PTS_PER_DAY       # 2880
PTS_24H       = PTS_PER_DAY                         # 96


def _recompute_derived() -> None:
    """argparse 가 BG_WINDOW_DAYS 등을 바꾼 뒤 파생 상수를 다시 계산."""
    global CADENCE_MIN, PTS_PER_DAY, BG_WINDOW_PTS, PTS_24H
    CADENCE_MIN   = CADENCE_SEC // 60
    PTS_PER_DAY   = 24 * 60 // CADENCE_MIN
    BG_WINDOW_PTS = BG_WINDOW_DAYS * PTS_PER_DAY
    PTS_24H       = PTS_PER_DAY

# WP 판정 결과 (LC_WATCH_* 대응)
WATCH_TRUE  = "TRUE"
WATCH_FALSE = "FALSE"
WATCH_STALE = "STALE"
WATCH_ERROR = "ERROR"


def _mad_sigma(arr: np.ndarray) -> float:
    """robust sigma = 1.4826 * MAD. 유효표본 < 2 이면 NaN."""
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return float("nan")
    med = np.median(arr)
    return float(1.4826 * np.median(np.abs(arr - med)))


# ──────────────────────────────────────────────────────
# 단일 채널 Watchpoint
# ──────────────────────────────────────────────────────
class Watchpoint:
    """
    채널 하나의 rolling 배경 + 임계 판정.
    cFS LC_watch 의 WP 한 개에 대응.
    persistence(연속카운트)는 여기서 하지 않는다 — 그건 AP(fsm_node) 책임.
    """

    def __init__(self, name: str):
        self.name      = name
        self.buf       = deque(maxlen=BG_WINDOW_PTS)   # rolling 배경 버퍼
        self.prev_watch = WATCH_FALSE                  # 직전 판정 (전이 검출용)
        self.onset_ts  : float | None = None           # FALSE→TRUE 전이 시각

    def evaluate(self, count: float, phys_ts: float) -> dict:
        """
        count 1포인트 입력 → WP 판정 dict 반환.
        반환: {channel, watch, count, threshold, bg, sigma, onset_ts}
        """
        # ── 배경 버퍼 갱신 (유한값만 누적) ──
        if np.isfinite(count):
            self.buf.append(float(count))

        bg    = float(np.median(self.buf)) if len(self.buf) > 0 else float("nan")
        sigma = _mad_sigma(np.array(self.buf))

        # ── WP 판정 (LC_FloatCompare 교훈 적용) ──
        watch     = WATCH_FALSE
        threshold = float("nan")

        if not np.isfinite(count):
            # 데이터 결손 → STALE (NaN 비교 금지)
            watch = WATCH_STALE
        elif math.isinf(count):
            # 극단 폭증 → 확실한 위반 (NASA: inf 는 일부러 TRUE 로 둠)
            watch = WATCH_TRUE
        elif (not np.isfinite(bg)) or (not np.isfinite(sigma)):
            # cold-start / 배경 추정 불가 → STALE (트리거 막음)
            watch = WATCH_STALE
        else:
            sigma_eff = sigma if sigma > 0 else 0.0
            threshold = bg + K * sigma_eff
            if ONSET_FLOOR > 0:
                threshold = max(threshold, ONSET_FLOOR)
            watch = WATCH_TRUE if (count > threshold) else WATCH_FALSE

        # ── FALSE→TRUE 전이 = onset 시각 (LC_ProcessWP 교훈) ──
        if watch == WATCH_TRUE and self.prev_watch != WATCH_TRUE:
            self.onset_ts = phys_ts
        elif watch == WATCH_FALSE:
            # 정상 복귀 시 onset 리셋 (STALE 동안은 유지)
            self.onset_ts = None
        self.prev_watch = watch

        return {
            "channel":   self.name,
            "watch":     watch,
            "count":     round(float(count), 3) if np.isfinite(count) else None,
            "threshold": round(threshold, 4) if np.isfinite(threshold) else None,
            "bg":        round(bg, 4) if np.isfinite(bg) else None,
            "sigma":     round(sigma, 4) if np.isfinite(sigma) else None,
            "onset_ts":  self.onset_ts,
        }


# ──────────────────────────────────────────────────────
# 데이터 로딩 (기존 ksem_node 유지)
# ──────────────────────────────────────────────────────
def _load_data(data_path: Path) -> pd.DataFrame:
    if data_path.is_dir():
        try:
            _dir = data_path.parent
            if str(_dir) not in sys.path:
                sys.path.insert(0, str(_dir))
            import ksem_io
            df, _ = ksem_io.load(data_path)
        except Exception:
            parquets = sorted(data_path.glob("*.parquet"))
            df = pd.concat([pd.read_parquet(p) for p in parquets])
    else:
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.resample(f"{CADENCE_MIN}min").mean()


# ──────────────────────────────────────────────────────
# KsemNode
# ──────────────────────────────────────────────────────
class KsemNode(Node):
    def __init__(self):
        super().__init__("ksem_node")

        self.pub = self.create_publisher(String, "/wp_results", 10)
        # (선택) 이벤트 시 24h 스냅샷 발행용
        self.snap_pub = self.create_publisher(String, "/ksem_snapshot", 10)

        self.get_logger().info(f"Loading KSEM data from {DATA_PATH} ...")
        df_raw = _load_data(DATA_PATH)

        cols = []
        self.channel_names: list[str] = []
        for pd_key, side, logic in CHANNELS:
            try:
                col = df_raw[pd_key, side, logic]
                cols.append(col)
                self.channel_names.append(f"{pd_key}{side}-{logic}")
            except KeyError:
                self.get_logger().warn(f"Channel not found: {pd_key}{side}-{logic}")

        if not cols:
            self.get_logger().error("No valid channels found.")
            return

        self.data = pd.concat(cols, axis=1)
        self.data.columns = self.channel_names

        # ── 리플레이 구간 슬라이스 (warm-up 포함 START..END) ──
        n_full = len(self.data)
        if START_DATE is not None:
            self.data = self.data[self.data.index >= pd.Timestamp(START_DATE, tz="UTC")]
        if END_DATE is not None:
            self.data = self.data[self.data.index <= pd.Timestamp(END_DATE, tz="UTC")]
        if START_DATE is not None or END_DATE is not None:
            self.get_logger().info(
                f"Replay window: {START_DATE}..{END_DATE}  "
                f"({len(self.data)}/{n_full} pts after slice)"
            )

        self.timestamps   = self.data.index
        self.cursor       = 0

        # 채널별 Watchpoint
        self.wps = {n: Watchpoint(n) for n in self.channel_names}

        # 24h 링버퍼 (DS 역할): 채널별 최근 96포인트 (ts, value)
        self.snapshot_buf = {n: deque(maxlen=PTS_24H) for n in self.channel_names}

        # ── 분리 배속: EVENT_TIME 이전=WARMUP_SPEED, 이후=REPLAY_SPEED ──
        # 배속은 타이머 주기일 뿐 판정·측정값(perf_counter, phys_ts)에 들어가지 않음.
        if EVENT_TIME is not None:
            self.event_phys_ts = pd.Timestamp(EVENT_TIME, tz="UTC").timestamp()
            self.in_warmup = True
            start_speed = WARMUP_SPEED
        else:
            self.event_phys_ts = None
            self.in_warmup = False
            start_speed = REPLAY_SPEED

        self._active_speed = start_speed
        timer_sec = CADENCE_SEC / start_speed
        self.timer = self.create_timer(timer_sec, self.tick)

        self.get_logger().info(
            f"KsemNode (WP) ready. channels={self.channel_names} "
            f"n_pts={len(self.timestamps)} timer={timer_sec:.3f}s "
            f"(warmup x{WARMUP_SPEED:.0f} -> active x{REPLAY_SPEED:.0f}, "
            f"event={EVENT_TIME})  BG_WINDOW={BG_WINDOW_DAYS}d K={K}"
        )

    def tick(self):
        if self.cursor >= len(self.timestamps):
            self.get_logger().info("Replay finished.")
            self.timer.cancel()
            # 완료 마커 파일 (러너가 폴링) + 자체 종료 요청
            try:
                marker = os.path.join(RESULT_DIR, "REPLAY_DONE")
                with open(marker, "w") as f:
                    f.write(f"{time.time()}\n")
            except Exception:
                pass
            # rclpy.spin 을 빠져나오도록 shutdown 요청
            rclpy.shutdown()
            return

        row     = self.data.iloc[self.cursor]
        ts      = self.timestamps[self.cursor]
        phys_ts = float(ts.timestamp())
        self.cursor += 1

        # ── warm-up→active 배속 전환 (첫 이벤트 도달 시 1회) ──
        # HK/측정과 무관 — 타이머 주기만 바꾼다. 측정 구간만 천천히 돌려
        # tegrastats aggregate 샘플을 확보하기 위함.
        if self.in_warmup and self.event_phys_ts is not None \
                and phys_ts >= self.event_phys_ts:
            self.in_warmup = False
            self.timer.cancel()
            self.timer = self.create_timer(CADENCE_SEC / REPLAY_SPEED, self.tick)
            self.get_logger().info(
                f"--- warm-up done at {ts.isoformat()}: "
                f"switch x{WARMUP_SPEED:.0f} -> x{REPLAY_SPEED:.0f} (active 측정 구간 시작) ---"
            )

        # ── 채널별 WP 판정 (HK telemetry: wp_eval_ms 계측) ──
        # 측정은 검출 로직을 "옆에서 관찰만" 한다. perf_counter 는 phys_ts 와
        # 무관하며 판정 입력으로 절대 들어가지 않음 (cFS HK/FDC 분리 원칙).
        results = []
        any_trigger = False
        t_wp = perf_counter()
        for name in self.channel_names:
            raw   = row[name]
            count = float(raw) if pd.notna(raw) else float("nan")
            res   = self.wps[name].evaluate(count, phys_ts)
            results.append(res)

            if res["watch"] == WATCH_TRUE:
                any_trigger = True
        wp_eval_ms = (perf_counter() - t_wp) * 1000.0   # rolling MAD 배경+판정 비용

        # snapshot 버퍼는 측정 구간 밖에서 (덤프 비용을 WP 비용에 섞지 않음)
        if KEEP_24H:
            for name in self.channel_names:
                raw = row[name]
                self.snapshot_buf[name].append(
                    (phys_ts, float(raw) if pd.notna(raw) else float("nan"))
                )

        # ── 발행 ──
        # ts        = phys_ts  (물리 관측시각, onset 판정 전용 — CCSDS primary)
        # ksem_pub_ts = wall-clock 발행시각 (telemetry 전용 — CCSDS secondary)
        # 두 시각은 의도적으로 분리: 판정 로직(AP)은 ts 만, transport 측정은 ksem_pub_ts 만.
        msg = String()
        msg.data = json.dumps({
            "ts":          phys_ts,
            "ksem_pub_ts": time.time(),
            "wp_eval_ms":  round(wp_eval_ms, 4),
            "results":     results,
        })
        self.pub.publish(msg)

        # ── 이벤트 시 24h 스냅샷 1회 발행 (가벼운 alert ≠ 무거운 덤프 분리) ──
        if KEEP_24H and any_trigger:
            snap = {
                "ts": phys_ts,
                "window_h": 24,
                "channels": {
                    n: list(self.snapshot_buf[n]) for n in self.channel_names
                },
            }
            snap_msg = String()
            snap_msg.data = json.dumps(snap)
            self.snap_pub.publish(snap_msg)

        # ── 로그 ──
        summary = " | ".join(
            f"{r['channel']}={r['watch']}" for r in results
        )
        self.get_logger().info(
            f"[{self.cursor}/{len(self.timestamps)}] {ts.isoformat()} | "
            f"wp_eval={wp_eval_ms:.3f}ms | {summary}"
        )


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="ksem_node (WP) — rolling MAD trigger, cFS LC_watch"
    )
    p.add_argument("--window", type=int, default=BG_WINDOW_DAYS,
                   help="rolling 배경 윈도우 (days). 기본 30, 측정 비교 시 7.")
    p.add_argument("--k", type=int, default=K,
                   help="bg_median + K*sigma 의 K. proton=10, electron 테스트=20.")
    p.add_argument("--replay-speed", type=float, default=REPLAY_SPEED,
                   help="active(측정) 구간 배속. 측정은 1800 권장.")
    p.add_argument("--warmup-speed", type=float, default=WARMUP_SPEED,
                   help="warm-up 구간 배속(측정 불필요라 초고속). 기본 7200.")
    p.add_argument("--event-time", type=str, default=None,
                   help="이 시각(YYYY-MM-DD[ HH:MM]) 이후 warmup→active 배속 전환. "
                        "보통 윈도우 첫 이벤트 begin_time.")
    p.add_argument("--onset-floor", type=float, default=ONSET_FLOOR,
                   help="임계 하한. proton=0.5, electron=0.0.")
    p.add_argument("--data", type=str, default=str(DATA_PATH),
                   help="parquet 디렉토리 또는 .csv 경로.")
    p.add_argument("--start", type=str, default=None,
                   help="리플레이 시작일 (YYYY-MM-DD). warm-up 포함해 지정.")
    p.add_argument("--end", type=str, default=None,
                   help="리플레이 종료일 (YYYY-MM-DD).")
    # ROS2 가 주입하는 args 무시 (rclpy.init 가 따로 처리)
    return p.parse_known_args(argv)[0]


def main(argv=None):
    args = _parse_args(argv)

    # ── argparse → 모듈 전역 덮어쓰기 (cFS TBL: 재빌드 없이 런타임 주입) ──
    global BG_WINDOW_DAYS, K, REPLAY_SPEED, WARMUP_SPEED, ONSET_FLOOR, DATA_PATH, START_DATE, END_DATE, EVENT_TIME
    BG_WINDOW_DAYS = args.window
    K              = args.k
    REPLAY_SPEED   = args.replay_speed
    WARMUP_SPEED   = args.warmup_speed
    ONSET_FLOOR    = args.onset_floor
    DATA_PATH      = Path(args.data)
    START_DATE     = args.start
    END_DATE       = args.end
    EVENT_TIME     = args.event_time
    _recompute_derived()   # BG_WINDOW_PTS 등 파생 상수 재계산 (윈도우 변경 반영)

    rclpy.init()
    node = KsemNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception:
        # tick 에서 rclpy.shutdown() 호출 시 spin 이 빠져나오며 예외 가능 — 정상 종료 경로
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
