import time
from .datasets import NGCFDataset
from .logger import get_logger, logging_conf
from .args import parse_args
from torch.utils.data import DataLoader
import torch
from .model import NGCF

logger = get_logger(logging_conf)

if __name__ == "__main__":
    start_time = time.time()

    logger.info("NGCF Training Start at")

    args = parse_args()

    device = torch.device("cuda" if args.use_cuda_if_available else "cpu")
    dataset = NGCFDataset(args)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    model = NGCF(
        args=args,
        device=device,
        n_users=dataset.get_n_users(),
        n_items=dataset.get_n_items(),
        norm_adj=dataset.get_cached_adj_matrix(),
    )

    optimizer = torch.optim.NAdam(
        params=model.parameters(),
        lr=args.lr,
    )

    for epoch in range(args.n_epochs):
        model.train()
        n_batch = dataset.__len__() // args.batch_size + 1
        running_loss = 0.0
        for _ in range(n_batch):
            batch = next(iter(dataloader))
            batch = [tensor.to(device) for tensor in batch]
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            

    logger.info(f"NGCF Training End at {time.time() - start_time:.2f} seconds")
