import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import time
import cv2
import subprocess
import threading
import re
import csv
import json
from ultralytics import YOLO

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------
# tegrastats
# -------------------------------
class TegraMonitor:
    def __init__(self):
        self.data = {"gpu": 0, "cpu": 0, "power": 0, "ram": 0}
        self.running = True

    def start(self):
        def run():
            p = subprocess.Popen(["tegrastats"], stdout=subprocess.PIPE, text=True)
            while self.running:
                line = p.stdout.readline()

                gpu = re.search(r"GR3D_FREQ (\d+)%", line)
                if gpu:
                    self.data["gpu"] = int(gpu.group(1))

                cpu = re.search(r"CPU \[(.*?)\]", line)
                if cpu:
                    vals = cpu.group(1).split(',')
                    vals = [int(v.split('%')[0]) for v in vals]
                    self.data["cpu"] = sum(vals) / len(vals)

                ram = re.search(r"RAM (\d+)/", line)
                if ram:
                    self.data["ram"] = int(ram.group(1))

                power = re.search(r"VDD_IN (\d+)mW", line)
                if power:
                    self.data["power"] = int(power.group(1))

        threading.Thread(target=run, daemon=True).start()

    def stop(self):
        self.running = False


# -------------------------------
# YOLO Node (TensorRT FP16)
# -------------------------------
class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')

        self.sub = self.create_subscription(
            Image,
            '/camera',
            self.callback,
            10
        )

        # Publisher 1: 탐지된 bbox 크롭 이미지 (탐지 시만 발행)
        self.roi_pub = self.create_publisher(Image, '/target_roi', 10)

        # Publisher 2: BBox JSON (탐지 시만 발행)
        self.bbox_pub = self.create_publisher(String, '/target_bbox', 10)

        self.bridge = CvBridge()

        self.get_logger().info("Loading TensorRT FP16 model...")
        self.model = YOLO(os.path.join(BASE_DIR, "../models/yolo26n.engine"))
        self.get_logger().info("Model loaded.")

        self.monitor = TegraMonitor()
        self.monitor.start()

        self.csv = open("log.csv", "w", newline="")
        self.writer = csv.writer(self.csv)
        self.writer.writerow([
            "timestamp",
            "ros_delay_ms",
            "yolo_infer_ms",
            "node_latency_ms",
            "roi_kb",       # 탐지 없으면 0
            "detections",
            "gpu_%",
            "cpu_%",
            "power_mW",
            "ram_MB"
        ])

    def callback(self, msg):
        t0 = time.perf_counter()

        # ── ROS end-to-end delay ──
        now = self.get_clock().now()
        sent = rclpy.time.Time.from_msg(msg.header.stamp)
        ros_delay = (now - sent).nanoseconds / 1e6  # ms

        # ── 프레임 변환 ──
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # ── YOLO 추론 (전체 프레임) ──
        t_infer_start = time.perf_counter()
        results = self.model(frame, verbose=False)
        yolo_infer_ms = (time.perf_counter() - t_infer_start) * 1000

        # ── 탐지 결과 파싱 ──
        detections = []
        roi_kb = 0.0

        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                conf = float(box.conf[0])
                cls  = int(box.cls[0])
                detections.append({
                    "cls":  cls,
                    "conf": round(conf, 3),
                    "bbox": [x1, y1, x2, y2]
                })

            # ── 신뢰도 가장 높은 bbox로 ROI 크롭 후 발행 ──
            best = max(detections, key=lambda d: d["conf"])
            x1, y1, x2, y2 = best["bbox"]

            # 프레임 경계 클램핑 (bbox가 프레임 밖으로 나가는 경우 방어)
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            roi = frame[y1:y2, x1:x2]

            if roi.size > 0:
                roi_msg = self.bridge.cv2_to_imgmsg(roi, encoding='bgr8')
                roi_msg.header.stamp = msg.header.stamp
                self.roi_pub.publish(roi_msg)

                _, buffer = cv2.imencode('.jpg', roi, [cv2.IMWRITE_JPEG_QUALITY, 80])
                roi_kb = len(buffer) / 1024

            # ── BBox JSON 발행 ──
            bbox_msg = String()
            bbox_msg.data = json.dumps(detections)
            self.bbox_pub.publish(bbox_msg)

        # ── 전체 콜백 지연 ──
        node_latency_ms = (time.perf_counter() - t0) * 1000

        # ── 리소스 기록 ──
        stats = self.monitor.data
        self.writer.writerow([
            time.time(),
            round(ros_delay, 2),
            round(yolo_infer_ms, 2),
            round(node_latency_ms, 2),
            round(roi_kb, 1),
            len(detections),
            stats["gpu"],
            round(stats["cpu"], 1),
            stats["power"],
            stats["ram"]
        ])

        print(
            f"ROS={ros_delay:.1f}ms | "
            f"YOLO={yolo_infer_ms:.1f}ms | "
            f"total={node_latency_ms:.1f}ms | "
            f"ROI={roi_kb:.1f}KB | "
            f"det={len(detections)} | "
            f"GPU={stats['gpu']}% | "
            f"pwr={stats['power']}mW"
        )

    def destroy_node(self):
        self.monitor.stop()
        self.csv.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = YoloNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()