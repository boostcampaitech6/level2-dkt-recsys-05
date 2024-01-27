import os
import argparse

import numpy as np
import torch

from dkt import trainer
from dkt.configs import load_config
from dkt.dataloader import Preprocess
from dkt.utils import get_logger, logging_conf


logger = get_logger(logging_conf)


def main(cfg):
    cfg['device'] = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Preparing data ...")
    preprocess = Preprocess(cfg=cfg)
    preprocess.load_test_data(file_name=cfg['test_file_name'])
    test_data: np.ndarray = preprocess.get_test_data()

    logger.info("Loading Model ...")
    model: torch.nn.Module = trainer.load_model(cfg=cfg).to(cfg['device'])

    logger.info("Make Predictions & Save Submission ...")
    trainer.inference(cfg=cfg, test_data=test_data, model=model)


if __name__ == "__main__":
    cfg = load_config("config.yaml")
    os.makedirs(cfg['model_dir'], exist_ok=True)
    main(cfg)
