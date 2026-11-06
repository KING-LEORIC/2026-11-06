"""실습 08. 새로운 프로세스에서 같은 모델·입력·설정의 검출을 대조합니다."""
# 파일 및 폴더 경로를 처리하기 위한 라이브러리
from pathlib import Path
# PyTorch : 딥러닝 모델 실행 및 CPU 스레드 설정
import torch
# Pandas : CSV 파일 읽기 및 데이터 비교
import pandas as pd
# NumPy : 수치 계산 및 오차 비교
import numpy as np
# vision.py의 이미지 탐지 함수
from vision import detect_images
# 현재 실행 중인 restore.py 파일이 들어 있는 폴더의 절대 경로
BASE=Path(__file__).resolve().parent
# 이 파일을 직접 실행했을 때만 아래 코드를 실행
if __name__=='__main__':
    # PyTorch의 CPU 연산 스레드 수를 1개로 제한
    # 결과 파일이 저장된 outputs 폴더 경로
    torch.set_num_threads(1);out=BASE/'outputs'
    # 복사해 둔 모델을 사용하여 동일한 검증 이미지에 대해 다시 추론
    # - out/'model.pt' : main.py에서 복사한 모델
    # - BASE/'data/images/val' : 원래 사용했던 검증 이미지 폴더
    # - out/'restored_images' : 재추론 결과 이미지 저장 폴더
    # - restored : 새 프로세스에서 얻은 이미지 탐지 결과
    # - counts : 새 프로세스에서 이미지별 탐지 객체 수
    restored,counts=detect_images(out/'model.pt',BASE/'data/images/val',out/'restored_images')
    # main.py에서 저장한 최초 탐지 결과 CSV 파일 읽기
    # 기존 결과와 재추론 결과를 비교하기 위한 정렬 기준
    # - image : 이미지 파일명
    # - class_id : 탐지된 객체의 클래스 번호
    # - confidence : 탐지 신뢰도
    # - x1, y1 : 바운딩 박스 왼쪽 위 좌표
    old=pd.read_csv(out/'predictions.csv');sort=['image','class_id','confidence','x1','y1']
    # 최초 결과와 재추론 결과를 동일한 기준으로 정렬
    # reset_index(drop=True) : 기존 행 번호를 버리고 0부터 다시 부여
    old=old.sort_values(sort).reset_index(drop=True);restored=restored.sort_values(sort).reset_index(drop=True)
    # >>> C10
    # 요구: 행 수·이미지·클래스가 같은지와 점수·좌표 오차를 함께 비교하세요.
    # 탐지가 0건인 경우에도 처리 이미지 수를 image_counts.csv와 대조합니다.
    # pass  # TODO C10: 위 요구사항의 핵심 코드를 작성하세요.
    
    # 최초 실행과 재실행의 탐지 결과가 동일한지 종합적으로 판정
    # - len(old)==len(restored) : 최초 결과와 재실행 결과의 탐지 행 수 비교
    # - old[['image','class_id']] : 최초 탐지 결과의 이미지 이름과 클래스 번호
    # - restored[['image','class_id']] : 재실행 결과의 이미지 이름과 클래스 번호
    # - equals() : 두 DataFrame의 값과 순서가 정확히 일치하는지 확인
    # - np.allclose() : 두 숫자 배열의 값이 허용 오차 범위 내에서 가까운지 확인
    # - confidence : 객체 탐지 신뢰도 점수
    # - x1, y1, x2, y2 : 바운딩 박스의 좌표
    # - atol=1e-4 : 절대 오차 허용 기준 0.0001
    # - rtol=1e-5 : 상대 오차 허용 기준 0.00001
    # - and : 세 가지 조건이 모두 True일 때만 최종 결과를 True로 판정
    # - same : 최초 실행과 재실행의 탐지 결과 일치 여부
    same=len(old)==len(restored) and old[['image','class_id']].equals(restored[['image','class_id']]) and np.allclose(old[['confidence','x1','y1','x2','y2']].astype(float),restored[['confidence','x1','y1','x2','y2']].astype(float),atol=1e-4,rtol=1e-5)
    # <<< C10
    # 최초 실행에서 저장한 이미지별 탐지 건수 CSV 파일 읽기
    expected=pd.read_csv(out/'image_counts.csv')
    # 최종 예측 결과 일치 여부와 이미지 처리 결과 일치 여부 출력
    # - counts.equals(expected) : 두 DataFrame의 데이터와 순서 및 자료형까지 비교
    print('예측 일치:',same,'이미지 처리 일치:',counts.equals(expected))
    # 이번 검증은 기존 모델의 재현성을 확인하는 과정 //test 데이터로 새로운 모델을 선택하거나 학습하는 과정이 아님
    print('동일 모델의 재현이며 test로 모델을 다시 선택한 것이 아닙니다.')
