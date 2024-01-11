import math

import numpy as np
from torch import nn
import wandb

from .dataloader import get_loaders
from .optimizer import get_optimizer
from .scheduler import get_scheduler
from .utils import get_logger, logging_conf
from .trainer import train, validate

logger = get_logger(logger_conf=logging_conf)

def run(args,
        train_data: np.ndarray,
        valid_data: np.ndarray,
        model: nn.Module):
    train_loader, valid_loader = get_loaders(args=args, train=train_data, valid=valid_data)

    # For warmup scheduler which uses step interval
    args.total_steps = int(math.ceil(len(train_loader.dataset) / args.batch_size)) * (
        args.n_epochs
    )
    args.warmup_steps = args.total_steps // 10

    optimizer = get_optimizer(model=model, args=args)
    scheduler = get_scheduler(optimizer=optimizer, args=args)

    best_auc = -1
    early_stopping_counter = 0
    for epoch in range(args.n_epochs):
        logger.info("Start Training: Epoch %s", epoch + 1)

        # TRAIN
        train_auc, train_acc, train_loss = train(train_loader=train_loader,
                                                 model=model, optimizer=optimizer,
                                                 scheduler=scheduler, args=args)

        # VALID
        auc, acc = validate(valid_loader=valid_loader, model=model, args=args)

        if auc > best_auc:
            best_auc = auc
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.patience:
                logger.info(
                    "EarlyStopping counter: %s out of %s",
                    early_stopping_counter, args.patience
                )
                break
        wandb.log(dict(epoch=epoch,
                       train_loss_epoch=train_loss,
                       train_auc_epoch=train_auc,
                       train_acc_epoch=train_acc,
                       valid_auc_epoch=auc,
                       valid_acc_epoch=acc,
                       best_auc = best_auc))        
        # scheduler
        if args.scheduler == "plateau":
            scheduler.step(best_auc)