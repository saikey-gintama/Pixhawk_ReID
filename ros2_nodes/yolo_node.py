import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time
import cv2
import subprocess
import threading
import re
import csv

# -------------------------------
# tegrastats
# -------------------------------
class TegraMonitor:
    def __init__(self):
        self.data = {"gpu":0, "cpu":0, "power":0, "ram":0}
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
                    self.data["cpu"] = sum(vals)/len(vals)

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
# ROI
# -------------------------------
def get_roi(frame):
    h, w, _ = frame.shape
    return frame[h//4:3*h//4, w//4:3*w//4]


# -------------------------------
# YOLO Node
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

        self.bridge = CvBridge()

        self.monitor = TegraMonitor()
        self.monitor.start()

        self.prev_time = time.time()
        self.frame_count = 0

        self.csv = open("log.csv", "w", newline="")
        self.writer = csv.writer(self.csv)
        self.writer.writerow([
            "timestamp",
            "ros_delay_ms",
            "node_latency_ms",
            "roi_kb",
            "gpu",
            "cpu",
            "power",
            "ram"
        ])

    def callback(self, msg):
        start = time.time()

        now = self.get_clock().now()
        sent = rclpy.time.Time.from_msg(msg.header.stamp)

        # 🔥 ROS end-to-end delay
        ros_delay = (now - sent).nanoseconds / 1e6

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # YOLO 자리
        # results = model(frame)

        # ROI
        roi = get_roi(frame)
        _, buffer = cv2.imencode('.jpg', roi)
        roi_kb = len(buffer) / 1024

        node_latency = (time.time() - start) * 1000

        stats = self.monitor.data

        self.writer.writerow([
            time.time(),
            ros_delay,
            node_latency,
            roi_kb,
            stats["gpu"],
            stats["cpu"],
            stats["power"],
            stats["ram"]
        ])

        print(f"ROS={ros_delay:.2f}ms | node={node_latency:.2f}ms | ROI={roi_kb:.1f}KB")

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