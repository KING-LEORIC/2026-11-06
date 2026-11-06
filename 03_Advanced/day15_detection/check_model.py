from ultralytics import YOLO
from pathlib import Path
# 현재 사용한 모델 불러오기
BASE=Path(__file__).resolve().parent
model = YOLO(BASE/'data/demo.pt')

# 모델의 작업 종류와 클래스 이름 확인
print("모델 작업:", model.task)
print("모델 클래스:", model.names)

# 검증 이미지 한 장으로 낮은 신뢰도 기준에서도 탐지되는지 확인
for threshold in (0.25, 0.01):
    result = model.predict(
        source=BASE/"data/images/val/standard_val_000.png",
        conf=threshold,
        imgsz=160,
        device="cpu",
        verbose=False
    )[0]

    print("\n신뢰도 기준:", threshold)
    print("탐지 객체 수:", len(result.boxes))

    if len(result.boxes) > 0:
        print("클래스 번호:", result.boxes.cls.cpu().tolist())
        print("신뢰도:", result.boxes.conf.cpu().tolist())