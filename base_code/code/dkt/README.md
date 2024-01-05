# Baseline1: Deep Knowledge Tracing

## Setup
```bash
cd /opt/ml/code/dkt
conda init
(base) . ~/.bashrc
(base) conda create -n dkt python=3.10 -y
(base) conda activate dkt
(dkt) pip install -r requirements.txt
(dkt) python train.py
(dkt) python inference.py
```

## Files
`code/dkt`
* `train.py`: 학습코드입니다.
* `inference.py`: 추론 후 `submissions.csv` 파일을 만들어주는 소스코드입니다.
* `requirements.txt`: 모델 학습에 필요한 라이브러리들이 정리되어 있습니다.

`code/dkt/dkt`
* `args.py`: `argparse`를 통해 학습에 활용되는 여러 argument들을 받아줍니다.
* `criterion.py`: Loss를 포함합니다.
* `datloader.py`: dataloader를 불러옵니다.
* `metric.py`: metric 계산하는 함수를 포함합니다.
* `model.py`: 여러 모델 소스 코드를 포함합니다. `LSTM`, `LSTMATTN`, `BERT`를 가지고 있습니다.
* `optimizer.py`: optimizer를 instantiate할 수 있는 소스코드를 포함합니다.
* `scheduler.py`: scheduler 소스코드를 포함합니다.
* `trainer.py`: 훈련에 사용되는 함수들을 포함합니다.
* `utils.py`: 학습에 필요한 부수적인 함수들을 포함합니다.
