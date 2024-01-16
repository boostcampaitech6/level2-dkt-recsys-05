import argparse
from pydantic import BaseModel
from typing import Optional

import yaml


class Config(BaseModel):
    seed = 42
    data_dir: str = "/data/ephemeral/data/"
    data_preprocessed_parquet: str = "data_preprocessed.parquet"
    wandb_key: Optional[str] = None
    wandb_team: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # model config
    xgb: Optional["XGBoostConfig"]
    
    best_params: Optional[dict] = {
        "booster": "dart",
        "max_depth": 13,
        "learning_rate": 0.05,
        "min_child_weight": 6,
        "gamma": 1,
        "colsample_bytree": 0.5,
        "lambda": 10,
        "alpha": 1,
        "subsample": 1.0,
        "max_delta_step": 5,
    }

    # feature config
    use_columns: list[str] = []


class XGBoostConfig(BaseModel):
    n_estimators: int = 5000
    objective: str = "binary:logistic"
    eval_metric: str = "auc"
    nthread: int = -1
    device: str = "cpu"
    tree_method: str = "hist"
    early_stopping_rounds: int = 100
    random_state: int = 42
    updater: str = "grow_gpu_hist"


Config.update_forward_refs()


def parse_config() -> Config:
    """
    config 파일을 파싱합니다.
    --config 옵션으로 config 파일을 지정할 수 있습니다.
    ex) python src/main.py --config src/train.yaml

    Returns:
        Config: 모델 학습을 위한 config
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config", default="src/train.yaml", type=str, help="config file path"
    )

    args = parser.parse_args()

    return Config.parse_obj(yaml.load(open(args.config), Loader=yaml.FullLoader))
