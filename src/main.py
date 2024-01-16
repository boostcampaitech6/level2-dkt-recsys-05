import os
import random
import string
from typing import Optional

import numpy as np
from config import parse_config
from dataset import get_data
import wandb

from models.xgboost import XGBoost


def generate_exp_code() -> str:
    possible_characters = string.ascii_letters + string.digits
    random_string = "".join(random.choices(possible_characters, k=6))

    return random_string.upper()


def wandb_login(
    team_name: Optional[str],
    project_name: Optional[str],
    run_name: Optional[str],
    key: Optional[str],
) -> None:
    if key is None:
        wandb.login()
    else:
        wandb.login(key=key)

    if project_name is None:
        project_name = input("Please input project name:")

    if team_name is None:
        team_name = input("Please input team name:")

    wandb.init(project=project_name, name=run_name, entity=team_name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


if __name__ == "__main__":
    exp_code = generate_exp_code()

    config = parse_config()

    wandb_login(
        config.wandb_team, config.wandb_project, config.wandb_run_name, config.wandb_key
    )

    seed_everything(config.seed)

    X_train, y_train, X_valid, y_valid, test_GB = get_data(config)

    xgb = XGBoost(
        config.xgb,
        config.best_params,
        config.use_columns,
        X_train,
        y_train,
        X_valid,
        y_valid,
        test_GB,
        exp_code,
    )

    xgb.hpo_start()
    xgb.train_start()

    print(f"{exp_code} Done!")
