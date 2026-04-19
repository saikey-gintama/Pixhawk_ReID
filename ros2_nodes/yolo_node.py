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
import os
from ultralytics import YOLO

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.environ.get("RESULT_DIR", ".")

# ═══════════════════════════════════════════════════════
# ▶ 필터링 파라미터  ← 여기만 바꾸면 됩니다
# ═══════════════════════════════════════════════════════
CONF_THRESHOLD = 0.45   # 이 값 미만 bbox 무시
TARGET_CLASS   = 0      # COCO class 0 = person  (-1 이면 전체 허용)

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
# ROI (중앙 50%) - 대역폭 비교용 고정 크롭
# -------------------------------
def get_roi(frame):
    h, w, _ = frame.shape
    return frame[h // 4:3 * h // 4, w // 4:3 * w // 4]


# -------------------------------
# YOLO Node (TensorRT FP16)
# -------------------------------
class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')

        self.sub      = self.create_subscription(Image,  '/camera',       self.callback, 10)
        self.roi_pub  = self.create_publisher(   Image,  '/target_roi',   10)
        self.bbox_pub = self.create_publisher(   String, '/target_bbox',  10)

        self.bridge = CvBridge()

        self.get_logger().info("Loading TensorRT FP16 model...")
        self.model = YOLO(os.path.join(BASE_DIR, "../models/yolo26n.engine"))
        self.get_logger().info("Model loaded.")

        self.monitor = TegraMonitor()
        self.monitor.start()

        # 성능 로그
        log_path = os.path.join(RESULT_DIR, "log.csv")
        self.csv    = open(log_path, "w", newline="")
        self.writer = csv.writer(self.csv)
        self.writer.writerow([
            "timestamp","ros_delay_ms","yolo_infer_ms","node_latency_ms",
            "roi_kb","detections","gpu_%","cpu_%","power_mW","ram_MB"
        ])

        # 이벤트 로그
        det_path = os.path.join(RESULT_DIR, "detection_log.csv")
        self.det_csv = open(det_path, "w", newline="")
        self.det_writer = csv.writer(self.det_csv)
        self.det_writer.writerow([
            "timestamp","event","num_detections","best_conf","best_bbox"
        ])

        self.prev_detected = False

        self.get_logger().info(f"Logging to {log_path}")
        self.get_logger().info(f"Detection events → {det_path}")

    def callback(self, msg):
        t0 = time.perf_counter()

        now   = self.get_clock().now()
        sent  = rclpy.time.Time.from_msg(msg.header.stamp)
        ros_delay = (now - sent).nanoseconds / 1e6

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        t_infer_start = time.perf_counter()
        results = self.model(frame, verbose=False)
        yolo_infer_ms = (time.perf_counter() - t_infer_start) * 1000

        detections = []
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls  = int(box.cls[0])
                # conf + class 필터
                if conf < CONF_THRESHOLD:
                    continue
                if TARGET_CLASS != -1 and cls != TARGET_CLASS:
                    continue
                detections.append({
                    "cls":  cls,
                    "conf": round(conf, 3),
                    "bbox": [round(x1), round(y1), round(x2), round(y2)]
                })
        has_detection = len(detections) > 0

        # 이벤트 기반 로깅
        if has_detection and not self.prev_detected:
            best = max(detections, key=lambda d: d["conf"])
            self.det_writer.writerow([
                time.time(),
                "DETECTION_START",
                len(detections),
                best["conf"],
                best["bbox"]
            ])
            self.det_csv.flush()
            print(f"[START] {best}")

        elif not has_detection and self.prev_detected:
            self.det_writer.writerow([
                time.time(),
                "DETECTION_END",
                0,
                0,
                []
            ])
            self.det_csv.flush()
            print("[END] detection lost")

        self.prev_detected = has_detection

        # bbox publish
        if has_detection:
            bbox_msg      = String()
            bbox_msg.data = json.dumps(detections)
            self.bbox_pub.publish(bbox_msg)

        # ROI publish
        roi = get_roi(frame)
        roi_msg = self.bridge.cv2_to_imgmsg(roi, encoding='bgr8')
        roi_msg.header.stamp = msg.header.stamp
        self.roi_pub.publish(roi_msg)

        _, buffer = cv2.imencode('.jpg', roi, [cv2.IMWRITE_JPEG_QUALITY, 80])
        roi_kb = len(buffer) / 1024

        node_latency_ms = (time.perf_counter() - t0) * 1000

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
        self.det_csv.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = YoloNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()