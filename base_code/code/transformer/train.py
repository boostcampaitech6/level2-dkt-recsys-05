import os
import argparse

import torch
import wandb

from transformer.args import parse_args
from transformer.datasets import TransformerDataset, PrepareData
from transformer import trainer
from transformer.utils import get_logger, set_seeds, logging_conf, CFG


logger = get_logger(logging_conf)


def main(cfg: CFG):
    wandb.login()
    wandb.init(config=vars(cfg))
    set_seeds(cfg.seed)

    cfg.train = True

    logger.info("Preparing data ...")
    train_data = PrepareData(cfg).get_data()
    train_data = TransformerDataset(train_data, cfg)

    logger.info("Building Model ...")
    model = trainer.build(cfg)
    
    logger.info("Start Training ...")
    trainer.run(model=model, train_data=train_data, cfg=cfg)


if __name__ == "__main__":
    args = parse_args()
    cfg = CFG('config.json')

    for key, value in vars(args).items():
        if value is not None:  # 명령줄에서 제공된 인자만 업데이트
            setattr(cfg, key, value)

    main(cfg=cfg)
