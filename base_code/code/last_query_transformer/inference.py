import os

import torch

from last_query_transformer.args import parse_args
from last_query_transformer.datasets import TransformerDataset, PrepareData
from last_query_transformer import trainer
from last_query_transformer.utils import get_logger, logging_conf, set_seeds, CFG


logger = get_logger(logging_conf)


def main(cfg):
    set_seeds(cfg.seed)

    cfg.cate_col_size = len(cfg.cate_cols)
    cfg.cont_col_size = len(cfg.cont_cols)
    cfg.train = False 

    logger.info("Preparing data ...")
    test_data = PrepareData(cfg).get_data()
    test_data = TransformerDataset(test_data, cfg)

    logger.info("Loading Model ...")
    model: torch.nn.Module = trainer.build(cfg)

    logger.info("Make Predictions & Save Submission ...")
    trainer.inference(cfg, model=model, data=test_data, output_dir=cfg.output_dir)

if __name__ == "__main__":
    args = parse_args()
    cfg = CFG('config.json')

    for key, value in vars(args).items():
        if value is not None:  # 명령줄에서 제공된 인자만 업데이트
            setattr(cfg, key, value)

    main(cfg=cfg)

