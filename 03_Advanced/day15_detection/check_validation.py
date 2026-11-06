
from pathlib import Path
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent

model = YOLO(str(BASE / "data" / "demo.pt"))

# 모델 체크포인트에 저장된 학습 정보 확인
ckpt = model.ckpt

print("===== 모델 학습 기록 =====")
print("모델 클래스:", model.names)

if ckpt is None:
    print("체크포인트 학습 정보가 없습니다.")
else:
    print("학습 epoch:", ckpt.get("epoch"))
    print("최고 fitness:", ckpt.get("best_fitness"))
    print("저장 날짜:", ckpt.get("date"))
    print("Ultralytics 버전:", ckpt.get("version"))

    print("\n===== 학습 설정 =====")
    train_args = ckpt.get("train_args")

    if train_args:
        for key in [
            "data", "model", "epochs", "imgsz",
            "batch", "device", "optimizer"
        ]:
            print(f"{key}:", train_args.get(key))
    else:
        print("저장된 train_args가 없습니다.")
