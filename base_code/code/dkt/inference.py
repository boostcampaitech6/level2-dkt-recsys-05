import os
import argparse

import numpy as np
import torch

from dkt import trainer
from dkt.args import parse_args
from dkt.dataloader import Preprocess
from dkt.utils import get_logger, logging_conf


logger = get_logger(logging_conf)


def main(args: argparse.Namespace):
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info("Preparing data ...")
    preprocess = Preprocess(args=args)
    preprocess.load_test_data(file_name=args.test_file_name)
    test_data: np.ndarray = preprocess.get_test_data()
    
    logger.info("Loading Model ...")
    model: torch.nn.Module = trainer.load_model(args=args).to(args.device)
    
    logger.info("Make Predictions & Save Submission ...")
    trainer.inference(args=args, test_data=test_data, model=model)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
