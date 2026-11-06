"""실습 04. imgsz 한 조건만 바꾼 1 epoch 연결 점검입니다."""
# pathlib.Path
# - 운영체제와 무관한 경로 객체 제공
# - 데이터셋, 출력 폴더 경로를 안전하게 관리
from pathlib import Path
# time.perf_counter
# - 코드 실행 시간을 고해상도로 측정
# - 학습 소요 시간(seconds) 계산에 사용
from time import perf_counter
# yaml
# - dataset_active.yaml 생성용
# - YOLO 학습에 사용할 데이터셋 설정 파일 저장
import yaml,torch
# pandas
# - 실험 결과를 DataFrame으로 정리
# - CSV 저장 및 표 출력에 사용
import pandas as pd
# Ultralytics YOLO
# - 모델 생성 및 학습 수행
from ultralytics import YOLO
# 현재 파이썬 파일이 위치한 폴더 경로
BASE=Path(__file__).resolve().parent
if __name__=="__main__":
    # PyTorch CPU 스레드 수를 1개로 제한
    # - 실행 환경 차이로 인한 속도 변동 감소
    torch.set_num_threads(1)
    # 출력 폴더 생성
    # - 실험 결과(CSV, 모델 파일 등) 저장 위치
    out=BASE/"outputs";out.mkdir(exist_ok=True);data=BASE.parent/"data"
    # 현재 실험용 데이터셋 설정 파일 경로
    p=out/"dataset_active.yaml"
    # YOLO 데이터셋 구성 정보 저장
    # - train : 학습 이미지 경로
    # - val : 검증 이미지 경로
    # - names : 클래스 이름 목록
    p.write_text(yaml.safe_dump({"train":str(data/"images/train"),"val":str(data/"images/val"),"names":["square","circle","triangle"]}),encoding="utf-8")
    # 실험 결과 저장용 리스트
    rows=[]
    # 입력 이미지 크기(imgsz) 비교 실험
    # - 160
    # - 192
    for size in [160,192]:
        # 난수 고정
        # - 동일 조건 실험을 위한 재현성 확보
        # yolo26n 구조의 새 모델 생성
        # 실험 시작 시간 기록
        torch.manual_seed(42);model=YOLO("yolo26n.yaml");start=perf_counter()
        # >>> C04
        # 요구: 현재 size만 imgsz로 바꿔 학습하세요. 다른 조건은 고정합니다.
        # 이 짧은 실험은 속도·파일 생성 확인이며 최적 모델 순위를 결론 내리지 않습니다.
        # pass  # TODO C04: 위 요구사항의 핵심 코드를 작성하세요.
        
        # YOLO 모델 학습 실행
        # - data=str(p) : 방금 생성한 데이터셋 설정 파일 사용
        # - epochs=1 : 연결 점검용 최소 학습
        # - imgsz=size : 현재 반복문의 이미지 크기(160 또는 192)
        model.train(data=str(p), epochs=1, imgsz=size)
        # <<< C04
        # 현재 실험 결과 기록
        # - imgsz : 입력 이미지 크기
        # - parameters : 전체 학습 파라미터 수
        # - seconds : 학습 소요 시간
        # - map50 : IoU=0.5 기준 mAP
        # - map50_95 : COCO 방식 mAP(0.5~0.95 평균)
        rows.append(dict(imgsz=size,parameters=sum(t.numel() for t in model.model.parameters()),seconds=perf_counter()-start,map50=float(model.metrics.box.map50),map50_95=float(model.metrics.box.map)))
    # 실험 결과 CSV 저장
    pd.DataFrame(rows).to_csv(out/"size_comparison.csv",index=False)
    # 결과 표 출력
    print(pd.DataFrame(rows).to_string(index=False))
