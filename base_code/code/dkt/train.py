import os

import numpy as np
import torch
import wandb

from dkt import trainer
from dkt.configs import load_config
from dkt.dataloader import Preprocess
from dkt.utils import get_logger, set_seeds, logging_conf


logger = get_logger(logging_conf)


def main(cfg):
    wandb.login()
    wandb.init(cfg['project'], config=cfg)
    set_seeds(cfg['seed'])
    cfg['device'] = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Preparing data ...")
    preprocess = Preprocess(cfg)
    preprocess.load_train_data(file_name=cfg['file_name'])
    train_data: np.ndarray = preprocess.get_train_data()
    train_data, valid_data = preprocess.split_data(data=train_data)

    logger.info("Building Model ...")
    model: torch.nn.Module = trainer.get_model(cfg=cfg).to(cfg['device'])

    logger.info("Start Training ...")
    trainer.run(cfg=cfg, train_data=train_data, valid_data=valid_data, model=model)


if __name__ == "__main__":
    cfg = load_config("config.yaml")
    os.makedirs(cfg['model_dir'], exist_ok=True)
    main(cfg)
