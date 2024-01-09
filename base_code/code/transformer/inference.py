import os

import torch

from lightgcn.args import parse_args
from lightgcn.datasets import prepare_dataset
from lightgcn import trainer
from lightgcn.utils import get_logger, logging_conf, set_seeds


logger = get_logger(logging_conf)


def main(args):
    set_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Preparing data ...")
    train_data, test_data, n_node = prepare_dataset(device=device, data_dir=args.data_dir)

    logger.info("Loading Model ...")
    weight: str = os.path.join(args.model_dir, args.model_name)
    model: torch.nn.Module = trainer.build(
        n_node=n_node,
        embedding_dim=args.hidden_dim,
        num_layers=args.n_layers,
        alpha=args.alpha,
        weight=weight,
    )
    model = model.to(device)

    logger.info("Make Predictions & Save Submission ...")
    trainer.inference(model=model, data=test_data, output_dir=args.output_dir)

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(name=args.model_dir, exist_ok=True)
    main(args=args)
