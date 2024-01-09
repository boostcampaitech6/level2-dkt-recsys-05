# Baseline2: LightGCN

## Setup
```bash
cd /opt/ml/code/lightgcn
conda init
(base) . ~/.bashrc
(base) conda create -n gcn python=3.10 -y
(base) conda activate gcn
(gcn) pip install -r requirements.txt
(gcn) python train.py
(gcn) python inference.py
```

## Files
`code/lightgcn`
* `train.py`: 학습코드입니다.
* `inference.py`: 추론 후 `submissions.csv` 파일을 만들어주는 소스코드입니다.
* `requirements.txt`: 모델 학습에 필요한 라이브러리들이 정리되어 있습니다.

`code/lightgcn/lightgcn`
* `args.py`: `argparse`를 통해 학습에 활용되는 여러 argument들을 받아줍니다.
* `datasets.py`: 학습 데이터를 불러 GCN 입력에 맞게 변환해줍니다.
* `trainer.py`: 훈련에 사용되는 함수들을 포함합니다.
* `utils.py`: 학습에 필요한 부수적인 함수들을 포함합니다.
