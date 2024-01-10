import os

import torch

from transformer.args import parse_args
from transformer.datasets import InferenceDataset, PrepareForIF
from transformer import trainer
from transformer.utils import get_logger, logging_conf, set_seeds, CFG


logger = get_logger(logging_conf)


def main(cfg):
    set_seeds(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Preparing data ...")
    test_data = PrepareForIF(cfg).get_data()['test']
    test_data = InferenceDataset(test_data, cfg, device, max_seq_len=cfg.seq_len)

    logger.info("Loading Model ...")
    weight: str = os.path.join(cfg.model_dir, cfg.model_name)
    model: torch.nn.Module = trainer.build(cfg)
    model = model.to(device)

    logger.info("Make Predictions & Save Submission ...")
    trainer.inference(cfg, model=model, data=test_data, output_dir=cfg.output_dir)

if __name__ == "__main__":
    args = parse_args()

    cfg = CFG('config.json')
    cfg.cate_col_size = len(cfg.cate_cols)
    cfg.cont_col_size = len(cfg.cont_cols)

    for key, value in vars(args).items():
        if value is not None:  # 명령줄에서 제공된 인자만 업데이트
            setattr(cfg, key, value)

    os.makedirs(name=cfg.model_dir, exist_ok=True)

    main(cfg=cfg)

