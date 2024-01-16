import random
import string
from config import parse_config
from dataset import get_data


from models.xgboost import XGBoost


def generate_exp_code() -> str:
    possible_characters = string.ascii_letters + string.digits
    random_string = "".join(random.choices(possible_characters, k=6))

    return random_string.upper()


if __name__ == "__main__":
    config = parse_config()

    exp_code = generate_exp_code()

    X_train, y_train, X_valid, y_valid = get_data(config)

    xgb = XGBoost(
        config.xgb, config.use_columns, X_train, y_train, X_valid, y_valid, exp_code
    )
    xgb.hpo_start()
    xgb.train_start()
