import json
import os
import random
import string

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def generate_exp_code() -> str:
    possible_characters = string.ascii_letters + string.digits
    random_string = "".join(random.choices(possible_characters, k=6))

    base = "exp/"
    dir_list = os.listdir(base)
    last_exp_seq = 0

    for dir in dir_list:
        if dir.split("_")[0].isdigit():
            last_exp_seq = max(last_exp_seq, int(dir.split("_")[0]))

    return f"{last_exp_seq+1}_{random_string.upper()}"


def get_path(dir, path):
    if not os.path.exists(dir):
        os.makedirs(dir)

    return os.path.join(dir, path)


def new_experiment(exp_code: str, exp: dict):
    write_path = get_path(
        f"exp/{exp_code}/",
        f"exp_{exp_code}.json",
    )

    with open(write_path, "w", encoding="utf8") as w:
        json.dump(exp, w, ensure_ascii=False, indent=2)


def update_experiment(exp_code: str, exp: dict):
    write_path = get_path(
        f"exp/{exp_code}/",
        f"exp_{exp_code}.json",
    )

    with open(write_path, "r", encoding="utf8") as r:
        origin_exp = json.load(r)

    for key, value in exp.items():
        origin_exp[key] = value

    with open(write_path, "w", encoding="utf8") as w:
        json.dump(exp, w, ensure_ascii=False, indent=2)
