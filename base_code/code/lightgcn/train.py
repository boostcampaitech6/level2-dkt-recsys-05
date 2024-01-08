import os
import argparse

import torch
import wandb

from lightgcn.args import parse_args
from lightgcn.datasets import prepare_dataset
from lightgcn import trainer
from lightgcn.utils import get_logger, set_seeds, logging_conf


logger = get_logger(logging_conf)


def main(args: argparse.Namespace):
    wandb.login()
    wandb.init(config=vars(args))
    set_seeds(args.seed)
    
    use_cuda: bool = torch.cuda.is_available() and args.use_cuda_if_available
    device = torch.device("cuda" if use_cuda else "cpu")

    logger.info("Preparing data ...")
    train_data, test_data, n_node = prepare_dataset(device=device, data_dir=args.data_dir)

    logger.info("Building Model ...")
    model = trainer.build(
        n_node=n_node,
        embedding_dim=args.hidden_dim,
        num_layers=args.n_layers,
        alpha=args.alpha,
    )
    model = model.to(device)
    
    logger.info("Start Training ...")
    trainer.run(
        model=model,
        train_data=train_data,
        n_epochs=args.n_epochs,
        learning_rate=args.lr,
        model_dir=args.model_dir,
    )


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(name=args.model_dir, exist_ok=True)
    main(args=args)
