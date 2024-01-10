import os

import torch

from transformer.args import parse_args
from transformer.datasets import TransformerDataset, PrepareData
from transformer import trainer
from transformer.utils import get_logger, logging_conf, set_seeds, CFG


logger = get_logger(logging_conf)


def main(cfg):
    set_seeds(cfg.seed)

    cfg.cate_col_size = len(cfg.cate_cols)
    cfg.cont_col_size = len(cfg.cont_cols)

    logger.info("Preparing data ...")
    test_data = PrepareData(cfg).get_data()
    test_data = TransformerDataset(test_data, cfg)

    logger.info("Loading Model ...")
    weight: str = os.path.join(cfg.model_dir, cfg.model_name)
    model: torch.nn.Module = trainer.build(cfg)
    model = model.to(cfg.device)

    logger.info("Make Predictions & Save Submission ...")
    trainer.inference(cfg, model=model, data=test_data, output_dir=cfg.output_dir)

if __name__ == "__main__":
    args = parse_args()
    cfg = CFG('config.json')

    for key, value in vars(args).items():
        if value is not None:  # 명령줄에서 제공된 인자만 업데이트
            setattr(cfg, key, value)

    main(cfg=cfg)

