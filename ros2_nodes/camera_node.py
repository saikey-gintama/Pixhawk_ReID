import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def gstreamer_pipeline():
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=1920, height=1080, framerate=60/1 ! "
        "nvvidconv ! "
        "video/x-raw, width=640, height=640, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! appsink drop=true max-buffers=1"
    )

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        self.pub = self.create_publisher(Image, '/camera', 10)
        self.bridge = CvBridge()

        self.cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        if not self.cap.isOpened(): 
            self.get_logger().error("Failed to open camera")

        self.timer = self.create_timer(0.016, self.publish_frame)  # ~60 FPS

    def publish_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')

        # 🔥 핵심: timestamp 삽입
        msg.header.stamp = self.get_clock().now().to_msg()

        self.pub.publish(msg)

def main():
    rclpy.init()
    node = CameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()