"""모델 호출 담당. 결과를 표 형태로 바꾸며 test를 입력하지 않습니다."""
# 파일 및 폴더 경로를 처리하기 위한 라이브러리
from pathlib import Path
# OpenCV : 이미지 및 동영상 입출력, 영상 프레임 처리
import cv2
# Pandas : 탐지 결과를 표(DataFrame) 형태로 정리
import pandas as pd
# Ultralytics : YOLO 모델을 불러오고 객체 탐지 및 추적 수행
from ultralytics import YOLO

# 이미지 탐지 함수
# - weights : YOLO 모델 가중치 파일 경로
# - folder : 탐지할 이미지가 저장된 폴더
# - output : 탐지 결과 이미지를 저장할 폴더
# - conf : 객체 탐지 신뢰도 기준 (기본값 0.25)
def detect_images(weights,folder,output,conf=.25):
    # YOLO 모델 불러오기
    # - str(weights) : Path 객체를 문자열 경로로 변환
    # 결과 저장 폴더 경로 설정
    # 결과 저장 폴더가 없으면 생성
    model=YOLO(str(weights));out=Path(output);out.mkdir(parents=True,exist_ok=True)
    # 이미지별 객체 탐지 결과를 저장할 리스트 //각 행에는 이미지 이름, 클래스 번호, 신뢰도, 바운딩 박스 좌표가 들어감
    # 이미지별 탐지 객체 수를 저장할 리스트 //탐지 객체가 없는 이미지도 기록
    rows=[];counts=[]
    # >>> C07
    # 요구: folder를 stream=True로 추론합니다. 결과 객체를 전부 메모리에 모으지 않습니다.
    # pass  # TODO C07: 위 요구사항의 핵심 코드를 작성하세요.
    
    # YOLO 이미지 객체 탐지 실행
    # - model.predict() : YOLO 모델의 이미지 객체 탐지 수행
    # - source=str(folder) : 탐지할 이미지 폴더 경로를 문자열로 전달
    # - stream=True : 전체 이미지 결과를 한꺼번에 메모리에 저장하지 않고 순차적으로 처리
    # - conf=conf : 객체 탐지 결과로 인정할 최소 신뢰도 기준 (기본값 0.25)
    # - imgsz=160 : 모델 추론에 사용할 입력 이미지 크기 지정
    # - device="cpu" : CPU를 사용하여 추론 실행
    # - verbose=False : 이미지별 상세 추론 로그 출력 비활성화
    # - results : 이미지별 YOLO 탐지 결과를 순차적으로 제공하는 객체
    results=model.predict(source=str(folder),stream=True,conf=conf,imgsz=160,device="cpu",verbose=False)
    # <<< C07
    # 이미지별 탐지 결과를 하나씩 처리 //stream=True이므로 이미지 전체 결과를 한꺼번에 메모리에 적재하지 않음
    for r in results:
        # 현재 처리 중인 이미지 파일명 추출
        # - r.path : YOLO 결과 객체에 포함된 원본 이미지 경로
        # - Path(r.path).name : 경로를 제외한 파일명만 추출
        # 현재 이미지에서 탐지한 객체 수 기록
        # - r.boxes : 현재 이미지의 바운딩 박스 탐지 결과
        # - len(r.boxes) : 탐지된 객체의 개수
        # - 탐지 결과가 0개여도 이미지 이름과 0을 기록
        image=Path(r.path).name;counts.append(dict(image=image,detections=len(r.boxes)))
        # 현재 이미지에 탐지 결과를 시각화한 이미지 생성 및 저장
        # - r.plot() : 바운딩 박스, 클래스 이름, 신뢰도 등을 이미지에 표시
        # - cv2.imwrite() : 이미지를 파일로 저장
        # - out/image : 결과 이미지의 저장 경로
        cv2.imwrite(str(out/image),r.plot())
        # 현재 이미지의 탐지 객체 정보를 하나씩 추출
        # - r.boxes.xyxy : 객체의 바운딩 박스 좌표 [x1, y1, x2, y2]
        # - r.boxes.cls : 탐지된 객체의 클래스 번호
        # - r.boxes.conf : 탐지된 객체의 신뢰도 점수
        # - cpu() : GPU 등에 있는 텐서를 CPU 메모리로 이동
        # - tolist() : 텐서 또는 배열을 Python 리스트로 변환
        # - zip() : 좌표, 클래스 번호, 신뢰도를 같은 객체끼리 묶음
        # 탐지 객체 한 개의 결과를 rows에 추가
        # - image : 원본 이미지 파일명
        # - int(cls) : 클래스 번호를 정수로 변환
        # - score : 객체 탐지 신뢰도
        # - *box : 바운딩 박스 좌표 4개를 개별 값으로 펼침
        # - 최종 형태 : [이미지명, 클래스번호, 신뢰도, x1, y1, x2, y2]
        for box,cls,score in zip(r.boxes.xyxy.cpu().tolist(),r.boxes.cls.cpu().tolist(),r.boxes.conf.cpu().tolist()):rows.append([image,int(cls),score,*box])
    # 탐지 결과를 Pandas DataFrame으로 변환
    # - columns : 각 열의 이름 지정
    # - 탐지 결과가 0건이어도 지정된 열 구조를 가진 DataFrame 반환
    # - 반환값 1 : 이미지별 객체 탐지 상세 결과
    # - 반환값 2 : 이미지별 탐지 객체 수
    return pd.DataFrame(rows,columns=["image","class_id","confidence","x1","y1","x2","y2"]),pd.DataFrame(counts)

# 동영상 객체 추적 함수
# - weights : YOLO 모델 가중치 파일 경로
# - video : 입력 동영상 파일 경로
# - output : 추적 결과 동영상을 저장할 폴더
# - tracker_name : 객체 추적 알고리즘 설정 파일명 (기본값 bytetrack.yaml)
def track_video(weights,video,output,tracker_name='bytetrack.yaml'):
    # YOLO 모델 불러오기
    # 결과 저장 폴더 경로 설정
    # 결과 저장 폴더가 없으면 생성
    model=YOLO(str(weights));out=Path(output);out.mkdir(parents=True,exist_ok=True)
    # OpenCV로 입력 동영상 파일 열기
    # - VideoCapture() : 동영상 파일을 읽기 위한 객체 
    cap=cv2.VideoCapture(str(video))
    # 동영상의 FPS 확인
    # 동영상의 가로 및 세로 크기 확인
    # - CAP_PROP_FRAME_WIDTH : 프레임 가로 크기
    # - CAP_PROP_FRAME_HEIGHT : 프레임 세로 크기
    # - int() : 영상 크기를 정수로 변환
    # 동영상 정보 확인이 끝났으므로 입력 동영상 객체 해제
    fps=cap.get(cv2.CAP_PROP_FPS);size=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)));cap.release()
    # 추적 결과를 저장할 동영상 파일 생성
    # - tracked.avi : 저장할 동영상 파일명
    # - MJPG : AVI 동영상에서 사용할 비디오 코덱
    # - fps : 원본 동영상의 초당 프레임 수
    # - size : 원본 동영상의 가로 및 세로 크기
    writer=cv2.VideoWriter(str(out/'tracked.avi'),cv2.VideoWriter_fourcc(*'MJPG'),fps,size)
    # 객체별 추적 결과를 저장할 리스트 //각 행에는 프레임 번호, 추적 ID, 클래스 번호, 바운딩 박스 좌표가 들어감
    # 프레임별 탐지 객체 수를 저장할 리스트
    rows=[];frames=[]
    # >>> C08
    # 요구: 새 영상용 모델에 tracker_name을 명시해 추적합니다. persist=False로 시작합니다.
    # pass  # TODO C08: 위 요구사항의 핵심 코드를 작성하세요.
    
    # YOLO 객체 탐지 및 동영상 객체 추적 실행
    # - model.track() : 객체 탐지와 객체 추적을 함께 수행
    # - source=str(video) : 추적할 동영상 파일 경로를 문자열로 전달
    # - tracker=tracker_name : 사용할 객체 추적 알고리즘 설정 파일 지정
    #   - 기본값은 bytetrack.yaml
    # - stream=True : 동영상 전체 프레임 결과를 한꺼번에 메모리에 저장하지 않고 순차적으로 처리
    # - persist=False : 이전 영상의 추적 상태를 이어받지 않고 새로운 추적 실행 시작
    # - conf=.25 : 객체 탐지 결과로 인정할 최소 신뢰도 기준 0.25
    # - imgsz=160 : 모델 추론에 사용할 입력 이미지 크기 지정
    # - device='cpu' : CPU를 사용하여 추론 실행
    # - verbose=False : 프레임별 상세 추론 로그 출력 비활성화
    # - results : 프레임별 객체 탐지 및 추적 결과를 순차적으로 제공하는 객체
    results=model.track(source=str(video),tracker=tracker_name,stream=True,persist=False,conf=.25,imgsz=160,device='cpu',verbose=False)
    # <<< C08
    # 동영상의 프레임을 순서대로 처리
    # - enumerate() : 반복할 때 프레임 번호와 결과를 함께 제공
    # - frame : 현재 프레임 번호 (0부터 시작)
    # - r : 현재 프레임의 YOLO 추적 결과
    for frame,r in enumerate(results):
        # 현재 프레임의 객체 탐지 및 추적 결과를 영상에 표시
        # - r.plot() : 바운딩 박스, 클래스 정보, 추적 ID 등을 시각화
        # - writer.write() : 시각화한 프레임을 출력 동영상에 기록
        # 현재 프레임 번호와 탐지 객체 수를 기록
        # - len(r.boxes) : 현재 프레임에서 탐지된 객체 수
        # - 추적 ID가 없는 객체도 탐지 객체 수에 포함
        writer.write(r.plot());frames.append([frame,len(r.boxes)])
        # 현재 프레임에서 추적 ID가 존재하는지 확인
        # - r.boxes.id : 객체별 추적 ID
        # - is not None : 추적 ID가 존재하는 경우에만 상세 추적 결과 기록
        # - 추적 ID가 없는 객체는 rows에 기록하지 않음
        if r.boxes.id is not None:
            # 추적 ID가 존재하는 객체의 정보를 하나씩 추출
            # - r.boxes.xyxy : 바운딩 박스 좌표
            # - r.boxes.cls : 객체의 클래스 번호
            # - r.boxes.id : 객체의 추적 ID
            # - cpu().tolist() : CPU 메모리로 이동한 후 Python 리스트로 변환
            # - zip() : 동일한 객체의 좌표, 클래스, 추적 ID를 묶음
            # 객체 한 개의 추적 결과를 rows에 추가
            # - frame : 현재 프레임 번호
            # - int(tid) : 객체의 추적 ID
            # - int(cls) : 객체의 클래스 번호
            # - *box : 바운딩 박스 좌표 4개를 개별 값으로 펼침
            # - 최종 형태 : [프레임번호, 추적ID, 클래스번호, x1, y1, x2, y2]
            for box,cls,tid in zip(r.boxes.xyxy.cpu().tolist(),r.boxes.cls.cpu().tolist(),r.boxes.id.cpu().tolist()):rows.append([frame,int(tid),int(cls),*box])
    # 모든 프레임 처리가 끝났으므로 출력 동영상 파일 기록 종료
    # - release() : VideoWriter가 사용한 자원 해제 및 파일 저장 마무리
    writer.release()
    # 추적 결과와 프레임별 탐지 결과를 각각 DataFrame으로 변환하여 반환
    # - 반환값 1 : 객체별 추적 결과
    # - 반환값 2 : 프레임별 탐지 객체 수
    return pd.DataFrame(rows,columns=['frame','track_id','class_id','x1','y1','x2','y2']),pd.DataFrame(frames,columns=['frame','detections'])
