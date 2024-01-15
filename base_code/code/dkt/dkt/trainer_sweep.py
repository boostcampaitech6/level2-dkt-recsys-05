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

def run(cfg,
        train_data: np.ndarray,
        valid_data: np.ndarray,
        model: nn.Module):
    train_loader, valid_loader = get_loaders(cfg=cfg, train=train_data, valid=valid_data)

    # For warmup scheduler which uses step interval
    cfg['total_steps'] = int(math.ceil(len(train_loader.dataset) / cfg['batch_size'])) * (
        cfg['n_epochs']
    )
    cfg['warmup_steps'] = cfg['total_steps'] // 10

    optimizer = get_optimizer(model=model, cfg=cfg)
    scheduler = get_scheduler(optimizer=optimizer, cfg=cfg)

    best_auc = -1
    early_stopping_counter = 0
    for epoch in range(cfg['n_epochs']):
        logger.info("Start Training: Epoch %s", epoch + 1)

        # TRAIN
        train_auc, train_acc, train_loss = train(train_loader=train_loader,
                                                 model=model, optimizer=optimizer,
                                                 scheduler=scheduler, cfg=cfg)

        # VALID
        auc, acc = validate(valid_loader=valid_loader, model=model, cfg=cfg)

        if auc > best_auc:
            best_auc = auc
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= cfg['patience']:
                logger.info(
                    "EarlyStopping counter: %s out of %s",
                    early_stopping_counter, cfg['patience']
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
        if cfg["scheduler"] == "plateau":
            scheduler.step(best_auc)