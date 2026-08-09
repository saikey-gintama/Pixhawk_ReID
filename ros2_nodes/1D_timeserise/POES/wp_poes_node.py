"""
wp_poes_node.py  --  cFS LC_watch (WP 층), POES 온보드 파이프라인 S1
======================================================================
POES count parquet(1분 원시)을 cadence(15분)로 리플레이하며
  1) 1분 원시 -> 15분 평균 (M1)
  2) 배경(median/MAD) 일 1회 갱신 (M4)
  3) z-score 계산 (TCN 입력용, /wp_results 로 발행)
  4) 채널별 임계 판정 TRUE/FALSE/STALE (M2, WDT 기반)
  5) run_len(tick-count 지속성 카운터) 갱신 -- PRE_ALERT/ALERT 게이트
까지 수행하고 /wp_results 로 발행한다. 판정 로직(ChannelState/WDT/resample_window)은
Node 클래스 밖 독립 클래스/함수라 rclpy 없이도 import 가능하다(md 2절, bench_micro.py 요구사항).

cFS 매핑
  LC WDT              : WDT(dict) -- 채널별 bg_window_days/k/onset_floor/gate_n/alert_n.
                         --gate-n/--alert-n 로 런타임 주입(TBL 철학), 채널 추가는 --channels만.
  LC FloatCompare 규칙 : ChannelState.eval_watchpoint() -- NaN=STALE(비교 안 함), inf=TRUE.
  LC LastFalseToTrue   : ChannelState.onset_ts (run_len 0->1 전이 시각).
  LC ACTIVE/PASSIVE    : --dry-run -- 판정은 그대로 계산·발행하되 snapshot 저장(부작용)만 생략.
  CS 체크섬            : SystemState.snapshot()/restore() 의 sha256(_checksum).
  SUCHAI Status repo   : SystemState -- bg_median/bg_std/run_len/onset_ts/최근 14개 z 를
                         JSON 비휘발성 저장. "7일 배경을 RAM에만 두면 재부팅 시 7일 무능력"
                         문제의 해법(논문 3.4절 기여점). restore 실패 시 콜드스타트 폴백.
  SUCHAI Flight plan   : 일 1회 bg 갱신 + 15분 주기 실행 (tick() 루프 자체).

재사용 (재구현 없음 -- import만, 엔진 수정 금지):
  fsm_count_spe_quietoff_mad_poes.py : _sigma (robust MAD sigma) 만 재사용.
      compute_rolling_bg 자체(전체 시계열 벡터화)는 여기서 못 쓴다 -- 이 노드는 온라인/증분
      계산이라 매 tick 마다 trailing 버퍼로 median+_sigma 를 계산한다(공식은 100% 동일,
      호출 형태만 증분식). build_threshold 의 공식(bg_median + k*bg_std, onset_floor 클립)도
      스칼라 한 줄이라 그대로 인라인 -- 값을 다르게 만드는 어떤 변경도 없음.
  poes_metop03_io.py                 : load (채널별 1개 파일, 부분 로드).

게이트/확정 파라미터 -- S0.5(gate_persistence_sweep.py) 실측 결과로 확정, 2026-08-08:
  PRE_ALERT(게이트) gate_n=2 (연속 2샘플=30분 초과 지속)
      POD 1.000(NOAA·손라벨 양쪽), duty_cycle 6.84%(상시구동 대비 14.6배 절감), 6.3회/일.
      gate_n=1(현 md 초안)은 duty 18.08%로 2.6배 더 비싼데 POD 이득이 없어 기각.
      gate_n=3은 POD 0.833/0.566 으로 붕괴 -- 게이트로 쓸 수 없음.
  ALERT(확정) alert_n=4 (연속 4샘플=45분 초과 지속)
      POD 0.738 로 오프라인 배치와 동일, event_FAR 0.061(오프라인 0.114 의 절반), precision 0.939.
  판정 방식은 **tick-count**(연속 유효샘플 수) 다. 벽시계 경과(elapsed>=1h) 방식이 아니다.
      -- 원래 md 4절 "정합 항목 ②: 벽시계 경과로 판정" 지시는 S0.5 실측으로 철회됨:
         tick-count 가 오프라인 POD 를 그대로 재현하면서 FAR 은 더 낮았다
         (wallclock 은 데이터 공백 때문에 224개 중 29개를 확정 못 해 POD 0.690 으로 떨어짐).
  결측(STALE)은 run_len 을 리셋하지 않고 건너뛴다 -- md 4절 정합 항목 ③은 유지
      (오프라인 배치의 dropna() 인과와 동일하게: 결측은 있으나 없으나 취급, 카운터 동결).

  인과 실현 갭 (기록만, 맞추려 하지 않음):
      오프라인 배치는 duration 을 "onset ~ 첫 미달 샘플"로 재서 데이터 공백이 duration 을
      채워준다(회복 샘플이 늦게 오면 그만큼 duration 이 길게 잡힘). 인과적 온보드 tick-count
      구현은 실제로 관측한 초과 샘플 수만 셀 수 있어 그럴 수 없다.
        오프라인 배치 raw onset(omni_p6, min_duration_h=1h)  224개
        온보드 tick-count(alert_n=4) 인과 재현                216개
      이 8개 차이는 버그가 아니라 배치(사후관측)와 인과(실시간) 알고리즘의 근본적 차이다.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    _HAVE_RCLPY = True
except ImportError:                       # Windows 개발 환경 -- 로직 단위 검증용
    _HAVE_RCLPY = False
    rclpy = None
    String = None

    class Node:                            # type: ignore -- 더미 베이스, 인스턴스화 안 함
        pass


def _default_repo() -> Path:
    if os.name == "nt":
        return Path("D:/VS_code/Pixhawk_ReID")
    return Path.home() / "jeongin" / "Pixhawk_ReID"


REPO = Path(os.environ.get("REPO", _default_repo()))
C_PD = REPO / "Experiment_window" / "C_PD"   # 기존 데이터·스크립트 -- 읽기 전용

for _p in (C_PD / "POES" / "count_FSM", C_PD / "POES" / "MetOp03_count"):
    sys.path.insert(0, str(_p))

import fsm_count_spe_quietoff_mad_poes as fsm_engine   # noqa: E402  -- _sigma 재사용
import poes_metop03_io as metop03_io                    # noqa: E402

RESULT_DIR = os.environ.get("RESULT_DIR", ".")

# ══════════════════════════════════════════════════════
# 파라미터 블록 -- 기본값(argparse 로 런타임 덮어쓰기, cFS TBL 철학)
# ══════════════════════════════════════════════════════
DATA_PATH      = C_PD / "POES" / "MetOp03_count" / "poes_metop03_cache_parquet"
CHANNELS       = "omni_p6"          # 콤마구분. 1차: 단일채널.

BG_WINDOW_DAYS = 7
K              = 7
ONSET_FLOOR    = 0.1
GATE_N         = 2                  # PRE_ALERT: 연속 2샘플(30분) -- S0.5 확정
ALERT_N        = 4                  # ALERT    : 연속 4샘플(45분) -- S0.5 확정

CADENCE_SEC    = 900                # 15분, 고정
REPLAY_SPEED   = 60.0
WARMUP_SPEED   = 7200.0
EVENT_TIME: str | None = None
START_DATE: str | None = None
END_DATE:   str | None = None

SNAPSHOT_PATH  = str(Path(RESULT_DIR) / "wp_poes_snapshot.json")
DRY_RUN        = False              # True 면 snapshot 저장(부작용)만 생략, 판정/발행은 그대로
IDLE           = False              # True 면 데이터 로드/타이머 생략 -- 노드만 기동(S5 시나리오 a')
STATS_PATH     = str(Path(RESULT_DIR) / "wp_stats.json")

Z_EPS  = 1.0
Z_CLIP = 10.0
RECENT_Z_LEN = 14                   # TCN 입력 윈도우(WINDOW=14)와 동일 -- 재부팅 후 워밍 없이 즉시 추론

WATCH_TRUE, WATCH_FALSE, WATCH_STALE, WATCH_ERROR = "TRUE", "FALSE", "STALE", "ERROR"

WDT: dict[str, dict] = {}           # main()에서 build_wdt() 로 채움


def build_wdt(channels: list[str], bg_window_days: int, k: float, onset_floor: float,
             gate_n: int, alert_n: int) -> dict:
    """WDT(Watchpoint Definition Table). 지금은 전 채널에 같은 값을 넣지만(1차 단일채널),
    테이블 구조라 채널별로 다른 값을 코드 수정 없이 줄 수 있다(cFS TBL 철학)."""
    return {
        ch: {"bg_window_days": bg_window_days, "k": k, "onset_floor": onset_floor,
             "gate_n": gate_n, "alert_n": alert_n}
        for ch in channels
    }


# ──────────────────────────────────────────────────────
# 데이터 로딩 (rclpy 불필요)
# ──────────────────────────────────────────────────────
def load_raw_channel(cache_dir: Path, channel: str) -> pd.Series:
    """1분 원시 count. 15분 리샘플은 여기서 하지 않는다 -- tick() 이 매 틱마다
    직접 슬라이스+평균해야 M1(resample_ms)이 의미가 있다."""
    tpl = metop03_io.fname_to_tuple(channel)
    df_count, _ = metop03_io.load(str(cache_dir), channels=[tpl])
    if df_count.index.tz is None:
        df_count.index = df_count.index.tz_localize("UTC")
    return df_count[tpl].dropna().sort_index()


def build_tick_schedule(raw: pd.Series) -> pd.DatetimeIndex:
    """15분 bin 경계를 pandas resample 자체로 뽑아 S0/S0.5 검증 파이프라인과 경계를 맞춘다
    (재사용). 실제 M1 측정은 tick() 이 매번 raw.loc[]로 다시 슬라이스해 수행 -- 여기선
    스케줄(타임스탬프 목록)만 얻는다."""
    return raw.resample("15min").mean().index


def resample_window(raw: pd.Series, bin_start: pd.Timestamp, cadence: pd.Timedelta) -> float:
    """1분 원시값 -> 15분 평균 1개 bin. M1 측정 대상 그 자체(온보드 1틱 리샘플 비용)."""
    end = bin_start + cadence - pd.Timedelta(microseconds=1)
    window = raw.loc[bin_start:end]
    return float(window.mean()) if len(window) else float("nan")


def compute_bg_stats(vals: np.ndarray) -> tuple[float, float]:
    """트레일링 버퍼(최대 bg_window_days*96개, 기본 7*96=672) -> (median, robust sigma).
    M4 측정 대상 그 자체. fsm_engine._sigma 재사용 -- 재구현 아님."""
    if len(vals) >= 2:
        return float(np.median(vals)), fsm_engine._sigma(vals)
    if len(vals) == 1:
        return float(vals[0]), float("nan")
    return float("nan"), float("nan")


def compute_z(count: float, bg_median: float, bg_std: float) -> float:
    """z = clip((count-bg_median)/max(bg_std,Z_EPS), -Z_CLIP,Z_CLIP). M2 측정 대상 그 자체.
    입력 중 하나라도 비유한이면 NaN(판정 임계와 별개 가드, ChannelState.eval_watchpoint 참고)."""
    if not (np.isfinite(count) and np.isfinite(bg_median) and np.isfinite(bg_std)):
        return float("nan")
    std_safe = max(bg_std, Z_EPS)
    return float(np.clip((count - bg_median) / std_safe, -Z_CLIP, Z_CLIP))


# ──────────────────────────────────────────────────────
# 채널 1개의 전체 온보드 상태 (배경 + run 카운터 + TCN 워밍용 최근 z)
# ──────────────────────────────────────────────────────
class ChannelState:
    """SUCHAI Status repository 단위(비휘발성 저장 대상). WDT 파라미터(k/onset_floor/
    gate_n/alert_n)는 여기 담지 않는다 -- 그건 설정(코드/CLI)이지 '상태'가 아니다."""

    def __init__(self, name: str, bg_window_days: int):
        self.name = name
        self.bg_window_days = bg_window_days
        self.buf: deque[tuple[pd.Timestamp, float]] = deque()   # trailing bg 윈도우 원시표본
        self.bg_median = float("nan")
        self.bg_std = float("nan")
        self.next_bg_update_ts: pd.Timestamp | None = None
        self.run_len = 0
        self.onset_ts: pd.Timestamp | None = None
        self.recent_z: deque[tuple[float, float]] = deque(maxlen=RECENT_Z_LEN)  # (unix_ts, z)

    def push_sample(self, ts: pd.Timestamp, value: float) -> None:
        """유한값만 배경 버퍼에 누적(오프라인 dropna()와 동일 인과) + 시간 기준 축출."""
        if np.isfinite(value):
            self.buf.append((ts, value))
        cutoff = ts - pd.Timedelta(days=self.bg_window_days)
        while self.buf and self.buf[0][0] < cutoff:
            self.buf.popleft()

    def maybe_update_bg(self, tick_ts: pd.Timestamp) -> float | None:
        """일 1회만 bg_median/bg_std 재계산(M4). 스케줄 도래 전이면 None(측정 없음, 로그에도
        생략). 앵커는 이 채널의 첫 tick 시각(오프라인 compute_rolling_bg 의 'first' 앵커와 동일,
        md 4절 정합 항목 ④).

        표본 부족(트레일링 버퍼가 비었거나 1개뿐)이면 그 필드는 갱신하지 않고 이전 유효값을
        그대로 둔다 -- compute_rolling_bg 가 전체 update-point 표를 만든 뒤 컬럼별로 개별
        ffill() 하는 것과 동일 인과(오프라인은 NaN 이 나온 갱신 지점을 건너뛰고 이전 유효
        업데이트 값을 계속 쓴다). 여기서 매번 NaN 으로 덮어쓰면 긴 데이터 공백(예: omni_p6
        2019-11-28~2020-01-01, 33일) 뒤에 이미 확립된 배경이 콜드스타트로 리셋돼 버려
        S0.5 tick-count 재현이 어긋난다(실측으로 발견, verify_wp_node.py 검증 2)."""
        if self.next_bg_update_ts is None:
            self.next_bg_update_ts = tick_ts
        if tick_ts < self.next_bg_update_ts:
            return None
        t0 = perf_counter()
        vals = np.array([v for _, v in self.buf], dtype=float)
        new_median, new_std = compute_bg_stats(vals)
        if np.isfinite(new_median):
            self.bg_median = new_median
        if np.isfinite(new_std):
            self.bg_std = new_std
        bg_update_ms = (perf_counter() - t0) * 1000.0
        self.next_bg_update_ts = tick_ts + pd.Timedelta(days=1)
        return bg_update_ms

    def eval_watchpoint(self, count: float, tick_ts: pd.Timestamp,
                        k: float, onset_floor: float, gate_n: int, alert_n: int) -> dict:
        """LC_watch WP 판정 1틱 (M2 측정 대상). 비교 연산자는 '>=' (fsm_engine.detect_segments
        와 동일 -- KSEM 의 '>' 를 그대로 옮기면 md 4절 함정 #1 에 걸린다).
        NaN=STALE(비교 안 함), inf=TRUE(FloatCompare 규칙, KSEM 승계)."""
        bg_median, bg_std = self.bg_median, self.bg_std

        if not np.isfinite(count):
            watch = WATCH_STALE
        elif math.isinf(count):
            watch = WATCH_TRUE
        elif not (np.isfinite(bg_median) and np.isfinite(bg_std)):
            watch = WATCH_STALE                              # cold-start / 배경 추정 불가
        else:
            threshold = bg_median + k * bg_std               # fsm_engine.build_threshold 와 동일 공식(원시 bg_std, 클립 없음)
            if onset_floor > 0:
                threshold = max(threshold, onset_floor)
            watch = WATCH_TRUE if count >= threshold else WATCH_FALSE   # 함정#1: '>=' (KSEM의 '>' 아님)

        # ── run_len: tick-count(연속 유효샘플 수). STALE 은 건너뜀(리셋도 증가도 안 함) ──
        # md 4절 정합 항목 ③ 유지: 결측이 연속 구간을 끊지 않는다(오프라인 dropna() 와 동일 인과).
        # 벽시계 경과 방식은 S0.5 실측으로 철회(모듈 docstring 참고).
        if watch == WATCH_STALE:
            pass
        elif watch == WATCH_TRUE:
            if self.run_len == 0:
                self.onset_ts = tick_ts                       # LastFalseToTrue
            self.run_len += 1
        else:
            self.run_len = 0
            self.onset_ts = None

        # ── z-score (TCN 입력용. WP 판정 임계와는 별개 가드 -- Z_EPS 로 하한, 판정 threshold 는 무클립) ──
        z = compute_z(count, bg_median, bg_std)
        if np.isfinite(z):
            self.recent_z.append((tick_ts.timestamp(), z))

        return {
            "channel": self.name, "watch": watch,
            "count": round(float(count), 3) if np.isfinite(count) else None,
            # z 는 반올림하지 않는다(그대로 float) -- round(z,4) 였을 때 최대 5e-5 오차가
            # /wp_results JSON 에 실려 AI 링버퍼까지 전달됐다. 학습은 float64 전정밀도
            # z 로 됐는데 배포에서만 4자리로 깎이면 train/serve skew 다. S5.2 다채널
            # 검증(목표 <1e-9)에서 실측으로 발견 -- count/bg_median/bg_std 는 진단·HK
            # 전용(모델 입력 아님)이라 그대로 반올림 유지.
            "z": float(z) if np.isfinite(z) else None,
            "bg_median": round(bg_median, 4) if np.isfinite(bg_median) else None,
            "bg_std": round(bg_std, 4) if np.isfinite(bg_std) else None,
            "run_len": self.run_len,                          # HK 텔레메트리 참고값(AP 판정 입력 아님, S2에서 확정)
            "onset_ts": self.onset_ts.timestamp() if self.onset_ts is not None else None,
            "gate_open": self.run_len >= gate_n,               # HK 참고용. AP 는 이 값이 아니라 자기 카운터로 판정한다
            "alert_open": self.run_len >= alert_n,             # (cFS 정통 순서: watch 불리언 -> RPN -> persistence, S2 docstring)
        }

    # ── 상태 저장/복구 (SUCHAI Status repository) ──
    def to_state_dict(self) -> dict:
        return {
            "bg_median": self.bg_median if np.isfinite(self.bg_median) else None,
            "bg_std": self.bg_std if np.isfinite(self.bg_std) else None,
            "next_bg_update_ts": (self.next_bg_update_ts.isoformat()
                                  if self.next_bg_update_ts is not None else None),
            "run_len": self.run_len,
            "onset_ts": self.onset_ts.isoformat() if self.onset_ts is not None else None,
            "recent_z": [[t, z] for t, z in self.recent_z],
        }

    def restore_from_state_dict(self, d: dict) -> None:
        self.bg_median = float(d["bg_median"]) if d.get("bg_median") is not None else float("nan")
        self.bg_std = float(d["bg_std"]) if d.get("bg_std") is not None else float("nan")
        nbu = d.get("next_bg_update_ts")
        self.next_bg_update_ts = pd.Timestamp(nbu) if nbu else None
        self.run_len = int(d.get("run_len", 0))
        ots = d.get("onset_ts")
        self.onset_ts = pd.Timestamp(ots) if ots else None
        self.recent_z = deque(((float(t), float(z)) for t, z in d.get("recent_z", [])),
                              maxlen=RECENT_Z_LEN)
        # 주의: buf(7일 원시 배경 버퍼)는 복구하지 않는다 -- 재부팅 즉시 bg_median/bg_std 는
        # 스냅샷 값 그대로 유효해(무능력 없음) 판정이 바로 재개되고, buf 는 이후 실시간 틱으로
        # 자연 재축적된다(모듈 docstring SUCHAI Status repository 항목, 논문 3.4절 취지).


def _canonical_json(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _checksum(payload: dict) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


class SystemState:
    """전 채널 상태 묶음 -- 단일 JSON 스냅샷(CS 체크섬 포함)으로 저장/복구."""

    def __init__(self, channels: dict[str, ChannelState]):
        self.channels = channels

    def snapshot(self, path: Path) -> None:
        payload = {
            "schema_version": 1,
            "saved_at": time.time(),
            "channels": {name: ch.to_state_dict() for name, ch in self.channels.items()},
        }
        checksum = _checksum(payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({**payload, "checksum": checksum}, f)
        tmp.replace(path)   # 원자적 교체 -- 쓰다 만 스냅샷이 restore 를 오염시키지 않게

    def restore(self, path: Path) -> tuple[bool, str]:
        """복구 성공 여부 + 사유. 실패해도 상태는 건드리지 않는다(호출부가 콜드스타트로 둔다)."""
        if not path.exists():
            return False, f"snapshot 파일 없음({path})"
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            return False, f"snapshot 파싱 실패({e})"
        checksum = raw.pop("checksum", None)
        expected = _checksum(raw)
        if checksum != expected:
            return False, "checksum 불일치(손상된 스냅샷)"
        try:
            for name, d in raw.get("channels", {}).items():
                if name in self.channels:
                    self.channels[name].restore_from_state_dict(d)
        except Exception as e:
            return False, f"snapshot 적용 실패({e})"
        return True, f"복구 완료(saved_at={raw.get('saved_at')})"


# ──────────────────────────────────────────────────────
# WpPoesNode -- rclpy 없으면 인스턴스화만 못 함(클래스 정의 자체는 항상 가능)
# ──────────────────────────────────────────────────────
class WpPoesNode(Node):
    def __init__(self):
        super().__init__("wp_poes_node")
        if String is None:
            raise RuntimeError("rclpy/std_msgs 없음 -- Jetson(ROS2 환경)에서 실행할 것")

        self.pub = self.create_publisher(String, "/wp_results", 10)

        if IDLE:
            # S5 시나리오 (a'): 데이터 로드/타이머 생략 -- rclpy/퍼블리셔만 그래프에 올라간
            # 채로 대기. tick() 이 절대 안 돌아 REPLAY_DONE 마커도 안 쓴다 -- 러너가 고정
            # 시간(sleep) 후 직접 종료시킨다(폴링 대상 아님).
            self.get_logger().info("[WP] --idle: 데이터 로드/타이머 생략, 노드만 기동 대기.")
            return

        self.get_logger().info(f"[WP] REPO={REPO}  cache={DATA_PATH}")
        self.get_logger().info(
            f"[WP] 게이트 확정 파라미터(S0.5 실측, 2026-08-08 확정): "
            f"PRE_ALERT gate_n={GATE_N}(연속 {GATE_N}샘플={GATE_N*15}min), "
            f"ALERT alert_n={ALERT_N}(연속 {ALERT_N}샘플={ALERT_N*15}min), tick-count 방식(벽시계 아님). "
            f"인과 실현 갭: 오프라인 배치 raw onset 224개 -> 온보드 tick-count 인과 재현 216개 "
            f"(배치는 duration을 'onset~첫 미달 샘플'로 재 공백이 채워주지만 인과 온보드는 못 그럼 -- "
            f"의도된 차이, 맞추지 않음)."
        )

        self.channels = [c.strip() for c in CHANNELS.split(",") if c.strip()]
        self.raw = {ch: load_raw_channel(DATA_PATH, ch) for ch in self.channels}
        for ch, s in self.raw.items():
            self.get_logger().info(
                f"[WP] {ch}: 원시 1분 표본 {len(s)}개 ({s.index[0]} ~ {s.index[-1]})")

        schedule = None
        for s in self.raw.values():
            sch = build_tick_schedule(s)
            schedule = sch if schedule is None else schedule.union(sch)
        if START_DATE is not None:
            schedule = schedule[schedule >= pd.Timestamp(START_DATE, tz="UTC")]
        if END_DATE is not None:
            schedule = schedule[schedule <= pd.Timestamp(END_DATE, tz="UTC")]
        self.schedule = schedule
        self.cursor = 0

        # ── 배경 워밍업 틱 카운트 (S5 "warm-up 제외" 프로토콜) ──
        # bg_median/bg_std 가 전 채널에서 처음 유효해지는 시각 이전 틱 수를 센다.
        # KSEM aggregate_resource.py 의 "타이머 간격 급변" 휴리스틱은 쓰지 않는다 --
        # 그건 배속 전환 시점 추정일 뿐 배경 워밍업과 무관하고, 여기선 실제 상태
        # (ChannelState.bg_median/bg_std 유한 여부)를 직접 본다.
        self.bg_warmup_done = False
        self.n_bg_warmup_ticks = 0
        self.bg_warmup_end_ts: float | None = None

        self.state = SystemState({
            ch: ChannelState(ch, WDT[ch]["bg_window_days"]) for ch in self.channels
        })
        self.snapshot_path = Path(SNAPSHOT_PATH)
        if not DRY_RUN:
            ok, reason = self.state.restore(self.snapshot_path)
            if ok:
                self.get_logger().info(f"[WP] snapshot restore 성공: {reason}")
            else:
                self.get_logger().warn(
                    f"[WP] snapshot restore 실패({reason}) -> 콜드스타트로 폴백")
        else:
            self.get_logger().info("[WP] --dry-run: snapshot restore 생략(콜드스타트로 진행)")

        if EVENT_TIME is not None:
            self.event_phys_ts = pd.Timestamp(EVENT_TIME, tz="UTC").timestamp()
            self.in_warmup = True
            start_speed = WARMUP_SPEED
        else:
            self.event_phys_ts = None
            self.in_warmup = False
            start_speed = REPLAY_SPEED
        timer_sec = CADENCE_SEC / start_speed
        self.timer = self.create_timer(timer_sec, self.tick)

        self.get_logger().info(
            f"[WP] ready. channels={self.channels} n_channels={len(self.channels)} "
            f"n_ticks={len(self.schedule)} timer={timer_sec:.4f}s "
            f"WDT={WDT}"
        )

    def tick(self):
        if self.cursor >= len(self.schedule):
            self.get_logger().info("[WP] Replay finished.")
            self.timer.cancel()
            if not DRY_RUN:
                self.state.snapshot(self.snapshot_path)
                self.get_logger().info(f"[WP] 종료 snapshot 저장 -> {self.snapshot_path}")
            try:
                stats = {
                    "n_ticks_total": len(self.schedule),
                    "n_bg_warmup_ticks": self.n_bg_warmup_ticks,
                    "bg_warmup_end_ts": self.bg_warmup_end_ts,
                    "channels": self.channels,
                }
                with open(STATS_PATH, "w", encoding="utf-8") as f:
                    json.dump(stats, f, indent=2)
                self.get_logger().info(f"[WP] stats 저장 -> {STATS_PATH}: {stats}")
            except Exception as e:
                self.get_logger().warn(f"[WP] stats 저장 실패({e})")
            try:
                marker = os.path.join(RESULT_DIR, "REPLAY_DONE")
                with open(marker, "w") as f:
                    f.write(f"{time.time()}\n")
            except Exception:
                pass
            rclpy.shutdown()
            return

        t_sample_rx = time.time()   # M10(e2e) 기준점 -- 이 샘플을 WP 가 처리하기 시작한 벽시계.
                                     # AP/AI 는 이 값을 변경 없이 그대로 전달만 한다(S4).
        bin_start = self.schedule[self.cursor]
        phys_ts = float(bin_start.timestamp())
        self.cursor += 1

        if self.in_warmup and self.event_phys_ts is not None and phys_ts >= self.event_phys_ts:
            self.in_warmup = False
            self.timer.cancel()
            self.timer = self.create_timer(CADENCE_SEC / REPLAY_SPEED, self.tick)
            self.get_logger().info(
                f"[WP] warm-up done at {bin_start.isoformat()} -> active 배속 전환")

        cadence = pd.Timedelta(seconds=CADENCE_SEC)

        # ── M1: 1분 원시 -> 15분 평균 (채널 합산) ──
        t0 = perf_counter()
        counts = {ch: resample_window(self.raw[ch], bin_start, cadence) for ch in self.channels}
        resample_ms = (perf_counter() - t0) * 1000.0

        # ── M4: 배경 갱신, 채널별로 일 1회만(스케줄 독립) ──
        bg_update_ms_total = 0.0
        any_bg_update = False
        for ch in self.channels:
            cs = self.state.channels[ch]
            cs.push_sample(bin_start, counts[ch])
            dt = cs.maybe_update_bg(bin_start)
            if dt is not None:
                bg_update_ms_total += dt
                any_bg_update = True
        bg_update_ms = bg_update_ms_total if any_bg_update else None

        # ── M2: WP 판정(z + threshold + run_len), 채널 합산 ──
        t1 = perf_counter()
        results = []
        for ch in self.channels:
            wdt = WDT[ch]
            res = self.state.channels[ch].eval_watchpoint(
                counts[ch], bin_start, wdt["k"], wdt["onset_floor"],
                wdt["gate_n"], wdt["alert_n"])
            results.append(res)
        z_eval_ms = (perf_counter() - t1) * 1000.0

        if not self.bg_warmup_done:
            all_warm = all(
                np.isfinite(self.state.channels[ch].bg_median)
                and np.isfinite(self.state.channels[ch].bg_std)
                for ch in self.channels
            )
            if all_warm:
                self.bg_warmup_done = True
                self.bg_warmup_end_ts = phys_ts
                self.get_logger().info(
                    f"[WP] 배경 워밍업 완료 @ {bin_start.isoformat()} "
                    f"({self.n_bg_warmup_ticks}틱 제외 대상)")
            else:
                self.n_bg_warmup_ticks += 1

        msg = String()
        msg.data = json.dumps({
            "ts": phys_ts,
            "t_sample_rx": t_sample_rx,
            "wp_pub_ts": time.time(),
            "n_channels": len(self.channels),
            "resample_ms": round(resample_ms, 4),
            "z_eval_ms": round(z_eval_ms, 4),
            "bg_update_ms": round(bg_update_ms, 4) if bg_update_ms is not None else None,
            "results": results,
        })
        self.pub.publish(msg)

        if any_bg_update and not DRY_RUN:
            self.state.snapshot(self.snapshot_path)

        summary = " | ".join(
            f"{r['channel']}={r['watch']}(run_len={r['run_len']},gate={r['gate_open']},alert={r['alert_open']})"
            for r in results
        )
        self.get_logger().info(
            f"[WP][{self.cursor}/{len(self.schedule)}] {bin_start.isoformat()} "
            f"n_ch={len(self.channels)} resample={resample_ms:.3f}ms z_eval={z_eval_ms:.3f}ms "
            f"bg_update={'-' if bg_update_ms is None else f'{bg_update_ms:.3f}ms'} | {summary}"
        )


def _parse_args(argv=None):
    import argparse
    p = argparse.ArgumentParser(
        description="wp_poes_node (WP) -- POES count 리플레이 + rolling MAD 임계 + tick-count 게이트, cFS LC_watch")
    p.add_argument("--channels", type=str, default=CHANNELS,
                   help="콤마구분 채널 목록. 1차: omni_p6 단일채널.")
    p.add_argument("--window", type=int, default=BG_WINDOW_DAYS, help="rolling 배경 윈도우(일)")
    p.add_argument("--k", type=float, default=K, help="threshold = bg_median + k*bg_std")
    p.add_argument("--onset-floor", type=float, default=ONSET_FLOOR, help="임계 하한")
    p.add_argument("--gate-n", type=int, default=GATE_N,
                   help="PRE_ALERT 게이트 지속성(연속 유효샘플 수). S0.5 확정값=2.")
    p.add_argument("--alert-n", type=int, default=ALERT_N,
                   help="ALERT 확정 지속성(연속 유효샘플 수). S0.5 확정값=4.")
    p.add_argument("--replay-speed", type=float, default=REPLAY_SPEED)
    p.add_argument("--warmup-speed", type=float, default=WARMUP_SPEED)
    p.add_argument("--event-time", type=str, default=None)
    p.add_argument("--data", type=str, default=str(DATA_PATH), help="POES count parquet 캐시 디렉토리")
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--snapshot-path", type=str, default=SNAPSHOT_PATH)
    p.add_argument("--stats-path", type=str, default=STATS_PATH)
    p.add_argument("--dry-run", action="store_true",
                   help="PASSIVE 등가 -- 판정/발행은 그대로, snapshot 저장(부작용)만 생략")
    p.add_argument("--idle", action="store_true",
                   help="데이터 로드/타이머 생략, 노드만 기동(S5 시나리오 a'). REPLAY_DONE 안 씀 -- "
                        "러너가 고정 시간 후 직접 종료시켜야 함.")
    return p.parse_known_args(argv)[0]   # ROS2 가 주입하는 args 무시


def main(argv=None):
    args = _parse_args(argv)

    global BG_WINDOW_DAYS, K, ONSET_FLOOR, GATE_N, ALERT_N, CHANNELS
    global REPLAY_SPEED, WARMUP_SPEED, EVENT_TIME, DATA_PATH, START_DATE, END_DATE
    global SNAPSHOT_PATH, DRY_RUN, IDLE, STATS_PATH, WDT
    CHANNELS       = args.channels
    BG_WINDOW_DAYS = args.window
    K              = args.k
    ONSET_FLOOR    = args.onset_floor
    GATE_N         = args.gate_n
    ALERT_N        = args.alert_n
    REPLAY_SPEED   = args.replay_speed
    WARMUP_SPEED   = args.warmup_speed
    EVENT_TIME     = args.event_time
    DATA_PATH      = Path(args.data)
    START_DATE     = args.start
    END_DATE       = args.end
    SNAPSHOT_PATH  = args.snapshot_path
    STATS_PATH     = args.stats_path
    DRY_RUN        = args.dry_run
    IDLE           = args.idle
    WDT = build_wdt([c.strip() for c in CHANNELS.split(",") if c.strip()],
                    BG_WINDOW_DAYS, K, ONSET_FLOOR, GATE_N, ALERT_N)

    if not _HAVE_RCLPY:
        raise SystemExit("[WP] rclpy 없음 -- 이 노드 실행은 Jetson(ROS2 환경)에서만 가능. "
                         "로직 단위 검증은 ChannelState/build_wdt/resample_window 를 "
                         "직접 import 해서 할 것(rclpy 불필요).")

    rclpy.init()
    node = WpPoesNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception:
        pass   # tick()의 rclpy.shutdown() 으로 spin 이 빠져나오며 예외 가능 -- 정상 종료 경로
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
