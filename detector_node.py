import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory # Added
import pyrealsense2 as rs
import numpy as np
import cv2
import os
from ultralytics import YOLO

class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector_node')
        
        # 1. Initialize ROS 2 Publishers
        self.image_pub = self.create_publisher(Image, 'yolo/annotated_frame', 10)
        self.coord_pub = self.create_publisher(Point, 'yolo/pepper_coord', 10)
        self.bridge = CvBridge()

        # 2. Data Collection Settings (Path updated to Package Folder)
        self.save_dir = '/root/colcon_ws/src/yolo_pepper_detector/pepper_data'
            
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        self.get_logger().info(f"Data will be saved in: {self.save_dir}")
        
        self.img_count = 0
        self.frame_count = 0
        self.max_images = 500
        self.save_interval = 5 # 1 image every 10 frames

        # 3. Load YOLOv8 Nano Model
        self.model = YOLO('yolov8n.pt')
        self.get_logger().info("YOLOv8 Nano Model loaded on GPU")

        # 4. Configure Intel RealSense
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) # Set to 30Hz for stability
        self.pipeline.start(config)
        self.get_logger().info("RealSense started at 30Hz. Auto-collection active.")
        
        # 5. Timer (approx 30 FPS)
        self.timer = self.create_timer(0.033, self.timer_callback)

    def timer_callback(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return

        img = np.asanyarray(color_frame.get_data())
        self.frame_count += 1

        # --- Automatic Data Collection ---
        if self.img_count < self.max_images:
            if self.frame_count % self.save_interval == 0:
                img_name = os.path.join(self.save_dir, f'pepper_{self.img_count:04d}.jpg')
                cv2.imwrite(img_name, img)
                self.img_count += 1
                if self.img_count % 50 == 0:
                    self.get_logger().info(f"Progress: {self.img_count}/{self.max_images}")
        
        # --- Visualization & Pub ---
        results = self.model(img, device=0, verbose=False)
        for result in results:
            annotated_frame = result.plot()
            if len(result.boxes) > 0:
                box = result.boxes[0].xywh[0].cpu().numpy()
                msg = Point()
                msg.x, msg.y = float(box[0]), float(box[1])
                self.coord_pub.publish(msg)

            img_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
            self.image_pub.publish(img_msg)

    def destroy_node(self):
        self.pipeline.stop()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()