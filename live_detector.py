import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import os

class UnifiedDetector(Node):
    def __init__(self):
        super().__init__('unified_detector')
        
        # 1. 통합 모델 경로 설정 (지민님의 v26 학습 결과물)
        # 경로에 /detect/detect/ 중복 여부를 실제 폴더 구조 보고 꼭 확인하세요!
        model_path = '/root/colcon_ws/src/yolo_pepper_detector/runs1/detect/detect/pepper_train_0309/weights/best.pt'
        
        if not os.path.exists(model_path):
            self.get_logger().error(f'❌ 모델 파일을 찾을 수 없습니다: {model_path}')
            return

        self.model = YOLO(model_path)
        self.bridge = CvBridge()
        
        # 2. 토픽 설정
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10)
        
        self.publisher = self.create_publisher(Image, '/detected_all_view', 10)
        self.get_logger().info('✅ YOLOv26 통합 디텍터가 준비되었습니다!')

    def image_callback(self, msg):
        try:
            # ROS Image -> OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # 3. 통합 모델 추론 (Pepper + Head 한 번에!)
            results = self.model.predict(cv_image, conf=0.5, verbose=False)
            
            # 4. 시각화 및 결과 전송
            annotated_frame = results[0].plot()
            out_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
            self.publisher.publish(out_msg)

        except Exception as e:
            self.get_logger().error(f'Error in image_callback: {e}')

# 🚀 지민님이 강조하신 그 'Main' 부분!
def main(args=None):
    rclpy.init(args=args)
    
    node = UnifiedDetector()
    
    try:
        # 노드를 계속 실행 상태로 유지 (이게 있어야 메시지를 계속 받습니다)
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('🛑 사용자에 의해 노드가 종료되었습니다.')
    finally:
        # 깔끔하게 종료 처리
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()