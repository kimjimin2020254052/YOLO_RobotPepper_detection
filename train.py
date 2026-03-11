import os
from ultralytics import YOLO

def main():
    # 1. 경로 설정 (절대 경로로 지정하는 것이 가장 안전합니다)
    # 현재 스크립트 위치를 기준으로 경로를 잡거나 직접 입력하세요.
    #base_path = "/root/colcon_ws/src/yolo_pepper_detector/runs_pepper/detect/train3/weights"
    
    #model_path = os.path.join(base_path, "best.pt")      # 아까 옮긴 v5 가중치
    data_yaml_path = "/root/colcon_ws/src/yolo_pepper_detector/train_detection_pepper_v1/data.yaml"       # v8 데이터셋 yaml 경로 (수정 필요!)
    
    # 2. YOLOv26 모델 로드 (지민님의 기존 지식 이식)
    model = YOLO('yolo26n.pt')

    # 3. 학습 시작
    print("🎬 YOLOv26 전이 학습을 시작합니다...")
    results = model.train(
        data=data_yaml_path,
        epochs=50,
        imgsz=160,
        device=0,               # GPU 사용 (NVIDIA)
        project = '/root/colcon_ws/src/yolo_pepper_detector/my_results',
        name="pepper_train_0311", # 결과 저장 폴더 이름
        plots=True,             # 학습 결과 그래프 생성
        exist_ok=True           # 같은 이름 폴더 있어도 덮어쓰기/이어서 하기
    )

    print("✅ 학습이 완료되었습니다!")

if __name__ == "__main__":
    main()