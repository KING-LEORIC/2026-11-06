"""실습 03. 같은 그룹의 이미지는 분할 경계를 넘지 않도록 배정합니다."""
# Path : 현재 파일 위치를 기준으로 경로를 다루기 위한 모듈
from pathlib import Path
# pandas : CSV 파일을 읽고 데이터프레임으로 처리하기 위한 라이브러리
import pandas as pd
# 현재 파이썬 파일이 있는 디렉터리 경로
BASE=Path(__file__).resolve().parent
# outputs 폴더 생성 (이미 존재하면 그대로 사용)
out=BASE/"outputs";out.mkdir(exist_ok=True)
# manifest.csv 로드
# - 이미지 메타데이터를 읽어 데이터프레임 생성
m=pd.read_csv(BASE.parent/"data/manifest.csv")
# 기존 test는 보관합니다. train+val의 개발 그룹만 새 분할을 설명합니다.
# test 데이터를 제외하고 train/val 대상 개발 데이터만 추출
# 개발 데이터에 포함된 모든 그룹 ID를 정렬하여 추출
dev=m[m.split!="test"].copy();groups=sorted(dev.group_id.unique())

# 전체 그룹의 75% 지점을 train/val 분리 경계로 계산
cut=int(len(groups)*.75)
# >>> C03
# 요구: 앞 75%의 group을 train, 나머지를 val로 배정하세요. 그룹별로 하나의 값만 갖게 합니다.
# pass  # TODO C03: 위 요구사항의 핵심 코드를 작성하세요.

# 앞쪽 75% 그룹을 train_groups로 지정
train_groups=set(groups[:cut])
# group_id 기준으로 train/val 부여
# - 같은 group_id는 반드시 동일한 split 값을 가짐
# - 그룹 단위 데이터 누수를 방지
dev["new_split"]=dev["group_id"].apply(lambda g: "train" if g in train_groups else "val")
# <<< C03
# 이미지 순서 기준으로 train/val을 번갈아 부여한 예시 //그룹 정보를 고려하지 않는 잘못된 분할 방식
naive=["train" if i%2==0 else "val" for i in range(len(dev))]
# naive 방식 결과 저장
dev["image_split"]=naive
# 그룹별로 new_split 값이 여러 개 존재하는 그룹 수 계산 //정상 구현 시 결과는 0
print("그룹 분할의 중복 그룹:",int((dev.groupby("group_id").new_split.nunique()>1).sum()))
# 그룹별로 image_split 값이 여러 개 존재하는 그룹 수 계산 //그룹을 무시했기 때문에 보통 0보다 큰 값이 나옴
print("이미지 번갈아 분할의 중복 그룹:",int((dev.groupby("group_id").image_split.nunique()>1).sum()))
# 결과를 CSV 파일로 저장
dev.to_csv(out/"split_demo.csv",index=False)
