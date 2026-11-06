"""실습 07. 데이터 감사와 실제 YOLO 추론·추적을 연결합니다."""
# 파일 및 폴더 경로를 운영체제에 맞게 처리하기 위한 라이브러리
from pathlib import Path
# JSON 파일 생성, SHA-256 해시 계산, 파일 복사를 위한 라이브러리
import json,hashlib,shutil
# PyTorch : 딥러닝 모델 실행 및 CPU 스레드 설정
import torch
# vision.py의 이미지 탐지 함수와 동영상 추적 함수
from vision import detect_images,track_video
# 현재 실행 중인 main.py 파일이 들어 있는 폴더의 절대 경로
# __file__ : 현재 Python 파일의 경로
# resolve() : 절대 경로로 변환
# parent : 해당 파일이 들어 있는 상위 폴더
BASE=Path(__file__).resolve().parent
# 이 파일을 직접 실행했을 때만 아래 코드를 실행 //다른 Python 파일에서 main.py를 import하는 경우에는 실행되지 않음
if __name__=='__main__':
    # PyTorch가 사용할 CPU 연산 스레드 수를 1개로 제한
    # CPU 자원 사용량과 실행 환경을 일정하게 관리하기 위한 설정
    torch.set_num_threads(1)
    # 결과물을 저장할 outputs 폴더 경로 생성
    # outputs 폴더가 없으면 생성하고, 이미 있으면 그대로 사용
    out=BASE/'outputs';out.mkdir(exist_ok=True)
    # YOLO 모델 가중치 파일 경로
    # demo.pt : 사전에 준비된 모델 파일
    weight=BASE/'data/demo.pt'
    # 이미지 탐지 실행
    # - detect_images(모델 가중치, 입력 이미지 폴더, 결과 저장 폴더)
    # - predictions : 이미지별 탐지 결과
    # - counts : 이미지별 탐지 객체 수
    predictions,counts=detect_images(weight,BASE/'data/images/val',out/'images')
    # 동영상 객체 추적 실행
    # - track_video(모델 가중치, 입력 동영상, 결과 저장 폴더)
    # - tracks : 객체별 추적 결과 및 좌표
    # - frames : 프레임별 탐지 객체 수
    tracks,frames=track_video(weight,BASE/'data/demo.avi',out)
    # 이미지별 탐지 결과를 CSV 파일로 저장
    # - index=False : DataFrame의 행 번호를 별도 열로 저장하지 않음
    # 이미지별 탐지 객체 수를 CSV 파일로 저장
    predictions.to_csv(out/'predictions.csv',index=False);counts.to_csv(out/'image_counts.csv',index=False)
    # 동영상의 객체 추적 결과를 CSV 파일로 저장
    # 프레임별 탐지 객체 수를 CSV 파일로 저장
    tracks.to_csv(out/'tracks.csv',index=False);frames.to_csv(out/'frames.csv',index=False)
    # >>> C09
    # 요구: 실제 처리한 이미지 수·프레임 수·모델 해시를 기록하세요.
    # pass  # TODO C09: 위 요구사항의 핵심 코드를 작성하세요.
    
    # 실험 실행 결과를 요약하는 딕셔너리 생성
    # - dict() : 키와 값으로 구성된 Python 딕셔너리 생성
    # - images=len(counts) : 실제 처리한 이미지 수
    # - frames=len(frames) : 실제 처리한 동영상 프레임 수
    # - detections=len(predictions) : 전체 이미지에서 탐지한 객체 수
    # - track_rows=len(tracks) : 동영상에서 추적 결과로 기록한 행 수
    # - model_sha256 : 사용한 모델 파일의 SHA-256 해시값
    # - hashlib.sha256(weight.read_bytes()) : 모델 파일의 바이너리 데이터로 해시 계산
    # - hexdigest() : 해시 결과를 16진수 문자열로 변환
    # - split='val' : 검증 데이터(val)를 사용했음을 기록
    # - test_used=False : test 데이터는 사용하지 않았음을 기록
    summary=dict(images=len(counts),frames=len(frames),detections=len(predictions),track_rows=len(tracks),model_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),split='val',test_used=False)
    # <<< C09
    # 실행 요약 정보를 JSON 파일로 저장
    # - json.dumps() : Python 딕셔너리를 JSON 문자열로 변환
    # - indent=2 : JSON 데이터를 2칸 들여쓰기로 정리
    # - encoding='utf-8' : UTF-8 문자 인코딩 사용
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    # 사용한 모델 가중치 파일을 outputs 폴더에 복사
    # - copyfile(원본, 복사본) : 파일 내용 복사
    # - 나중에 동일한 모델을 사용해 결과를 재검증할 수 있도록 보관
    shutil.copyfile(weight,out/'model.pt')  # 신뢰하는 자체 준비 모델을 함께 보관합니다.
    # 실행 요약 정보를 화면에 출력
    # - ensure_ascii=False : 한글 등이 유니코드 이스케이프 형태로 바뀌지 않도록 설정
    print(json.dumps(summary,ensure_ascii=False,indent=2))
