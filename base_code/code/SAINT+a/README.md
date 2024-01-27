# SAINT+$\alpha$
- Kaggle Riiid Competition에서 등장한 SAINT 모델에서 추가적인 정보를 반영한 모델
- 인코더에 추가적인 정보를 반영한 모델 (+ $\alpha$)

## Setup
```potery install
python main.py
```

## Files
`code/SAINT+α`
* `main.py` : 데이터 전처리 및 학습, 추론까지 할 수 있는 코드입니다.
* `sweep.py` : Wandb Sweep을 수행하는 코드입니다.
* `config.yaml` : 모델 학습에 활용되는 여러 argument들이 저장되어있는 코드입니다.
* `sweep.yaml` : Wandb sweep에 활용되는 parameter들이 저장되어있는 코드입니다.

`code/SAINT+α/SaintPlusAlpha`
* `dataloader.py` : dataloader를 불러오는 코드입니다.
* `inference.py` : Test Data에 대해 추론 후 예측 결과를 파일로 만들어주는 코드입니다.
* `model.py` : SAINT+α 모델이 포함되어 있는 코드입니다.
* `preprocess.py` : 학습을 위한 데이터를 전처리하는 코드입니다.
* `train.py` : SAINT+α 모델을 학습하는 코드입니다.
* `utils.py` : 학습에 필요한 부수적인 함수들을 포함하는 코드입니다.
