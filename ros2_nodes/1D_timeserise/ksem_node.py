"""
ksem_node.py
============
camera_node.py 의 최소 수정판.
GStreamer 카메라 대신 KSEM count parquet/CSV 를 15분 cadence 로 읽어
Float32MultiArray 토픽 '/ksem_count' 으로 발행한다.

변경 사항 (camera_node.py 대비):
  - cv2 / CvBridge / GStreamer 제거
  - 타이머 주기 0.016s(60fps) → 900s(15min cadence) 또는 REPLAY_SPEED 로 배속
  - 발행 토픽: /camera(Image) → /ksem_count(Float32MultiArray)
  - 메시지 헤더 timestamp 동일하게 삽입 (fsm_node 에서 ROS delay 측정용)

파라미터 (상단 블록만 수정):
  DATA_PATH      : ksem count parquet 또는 CSV 경로
  CHANNELS       : 사용할 채널 목록 (fsm_node 와 동일하게 맞출 것)
  CADENCE_SEC    : 실제 위성 cadence (15분 = 900)
  REPLAY_SPEED   : 1.0 = 실시간, 10.0 = 10배속 (테스트용)
"""

from __future__ import annotations
import sys
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import pandas as pd
import numpy as np

# ══════════════════════════════════════════════════════
# 파라미터 블록 — 여기만 수정
# ══════════════════════════════════════════════════════
DATA_PATH     = Path("ksem_cache_parquet")   # ksem_io parquet 디렉토리
                                              # CSV 라면 .csv 파일 경로로 교체
CHANNELS: list[tuple] = [                    # (pd_key, side, logic)
    ("PD3", "A", "OU"),
    ("PD3", "A", "OUT"),
    ("PD3", "A", "FTU"),
    ("PD3", "A", "FTUO"),
]
CADENCE_SEC   = 900          # 15분 cadence
REPLAY_SPEED  = 60.0         # 60배속 → 15분 데이터를 15초마다 발행 (테스트)
# ══════════════════════════════════════════════════════


def _load_data(data_path: Path) -> pd.DataFrame:
    """parquet 디렉토리 또는 CSV 를 로드해 15min resample."""
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
    return df.resample("15min").mean()


class KsemNode(Node):
    def __init__(self):
        super().__init__("ksem_node")

        self.pub = self.create_publisher(Float32MultiArray, "/ksem_count", 10)

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

        self.data: pd.DataFrame = pd.concat(cols, axis=1)
        self.data.columns = self.channel_names
        self.timestamps = self.data.index
        self.cursor = 0
        self.n_ch   = len(self.channel_names)

        timer_sec = CADENCE_SEC / REPLAY_SPEED
        self.timer = self.create_timer(timer_sec, self.publish_frame)

        self.get_logger().info(
            f"KsemNode ready. channels={self.channel_names} "
            f"n_pts={len(self.timestamps)} "
            f"timer={timer_sec:.1f}s (x{REPLAY_SPEED:.0f} speed)"
        )

    def publish_frame(self):
        if self.cursor >= len(self.timestamps):
            self.get_logger().info("Replay finished.")
            self.timer.cancel()
            return

        row    = self.data.iloc[self.cursor]
        values = row.fillna(0.0).values.astype("float32").tolist()
        ts     = self.timestamps[self.cursor]
        self.cursor += 1

        msg = Float32MultiArray()

        # 채널명을 dim.label 에, 물리 시각을 data_offset 에 실어 보냄
        dim        = MultiArrayDimension()
        dim.label  = ",".join(self.channel_names)
        dim.size   = self.n_ch
        dim.stride = self.n_ch
        msg.layout.dim.append(dim)
        msg.layout.data_offset = int(ts.timestamp())   # unix epoch(초)

        msg.data = values
        self.pub.publish(msg)

        self.get_logger().info(
            f"[{self.cursor}/{len(self.timestamps)}] {ts.isoformat()} | "
            + " | ".join(f"{n}={v:.2f}" for n, v in zip(self.channel_names, values))
        )


def main():
    rclpy.init()
    node = KsemNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
