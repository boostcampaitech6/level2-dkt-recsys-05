# Baseline LightGCN을 본따서 Transformer를 만들었습니다.

## Setup
```potery install
python train.py --data_dir ../data/
python inference.py --data_dir ../data/
```

## Files
`code/transformer`
* `train.py`: 학습코드입니다.
* `inference.py`: 추론 후 `submissions.csv` 파일을 만들어주는 소스코드입니다.
* `config.json`: 환경변수들을 이 파일에서 수정할 수 있습니다.

`code/transformer/transformer`
* `args.py`: `argparse`를 통해 학습에 활용되는 여러 argument들을 받아줍니다.
* `datasets.py`: 학습 데이터를 불러 transformer 입력에 맞게 변환해줍니다.
* `trainer.py`: 훈련에 사용되는 함수들을 포함합니다.
* `utils.py`: 학습에 필요한 부수적인 함수들을 포함합니다.
