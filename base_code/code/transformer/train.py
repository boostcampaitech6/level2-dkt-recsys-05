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

    use_cuda: bool = torch.cuda.is_available() and cfg.use_cuda_if_available
    device = torch.device("cuda" if use_cuda else "cpu")

    logger.info("Preparing data ...")

    cfg.train = True
    train_data = PrepareData(cfg).get_data()
    train_data = TransformerDataset(train_data, cfg, device, max_seq_len=cfg.seq_len)

    train_size = int(0.8 * len(train_data))
    valid_size = len(train_data) - train_size
    train_data, valid_data = torch.utils.data.random_split(train_data, [train_size, valid_size])

    logger.info("Building Model ...")
    model = trainer.build(cfg)
    model = model.to(device)
    
    logger.info("Start Training ...")
    trainer.run(
        model=model,
        train_data=train_data,
        valid_data=valid_data,
        cfg=cfg,
        n_epochs=cfg.n_epochs,
        learning_rate=cfg.lr,
        model_dir=cfg.model_dir,
    )


if __name__ == "__main__":
    args = parse_args()
    cfg = CFG('config.json')

    for key, value in vars(args).items():
        if value is not None:  # 명령줄에서 제공된 인자만 업데이트
            setattr(cfg, key, value)

    os.makedirs(name=cfg.model_dir, exist_ok=True)

    main(cfg=cfg)
