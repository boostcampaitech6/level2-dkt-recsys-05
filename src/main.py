from config import parse_config
from dataset import get_data
from utils.wandb import init as init_wandb
from utils.common import seed_everything, generate_exp_code
import warnings

from models.xgboost import XGBoost

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    exp_code = generate_exp_code()

    config = parse_config()

    init_wandb(
        team_name=config.wandb_team,
        project_name=config.wandb_project,
        run_name=f"xgboost_f{len(config.use_columns)}_{exp_code}",
        key=config.wandb_key,
    )

    seed_everything(config.seed)

    X_train, y_train, X_valid, y_valid, test_GB = get_data(config)

    xgb = XGBoost(
        config,
        X_train,
        y_train,
        X_valid,
        y_valid,
        test_GB,
        exp_code,
    )

    if not config.hpo.skip:
        xgb.hpo_start()

    xgb.train_start()

    print(f"{exp_code} Done!")
