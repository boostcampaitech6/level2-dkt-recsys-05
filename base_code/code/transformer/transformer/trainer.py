import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
from torch import nn
from .model import TransformerModel
from .datasets import TransformerDataset
from torch.utils.data import DataLoader
import wandb

from .utils import get_logger, logging_conf


logger = get_logger(logger_conf=logging_conf)


def build(cfg):
    model = TransformerModel(cfg)
    return model


def run(
    model: nn.Module,
    train_data,
    valid_data,
    cfg,
    n_epochs: int = 100,
    learning_rate: float = 0.01,
    model_dir: str = None,
):
    model.train()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    # loss함수
    loss_fun = nn.BCELoss()

    os.makedirs(name=model_dir, exist_ok=True)

    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=cfg.batch_size, shuffle=True)

    logger.info(f"Training Started : n_epochs={n_epochs}")
    best_auc, best_epoch = 0, -1
    for e in range(n_epochs):
        logger.info("Epoch: %s", e)
        # TRAIN
        train_auc, train_acc, train_loss = train(train_loader=train_loader, model=model, optimizer=optimizer, loss_fun=loss_fun)
    
        # VALID
        auc, acc = validate(model=model, valid_loader=valid_loader)

        wandb.log(dict(train_loss_epoch=train_loss,
                       train_acc_epoch=train_acc,
                       train_auc_epoch=train_auc,
                       valid_acc_epoch=acc,
                       valid_auc_epoch=auc))

        if auc > best_auc:
            logger.info("Best model updated AUC from %.4f to %.4f", best_auc, auc)
            best_auc, best_epoch = auc, e
            torch.save(obj= {"model": model.state_dict(), "epoch": e + 1},
                       f=os.path.join(model_dir, f"best_model.pt"))
            
    torch.save(obj={"model": model.state_dict(), "epoch": e + 1},
               f=os.path.join(model_dir, f"last_model.pt"))
    
    logger.info(f"Best Weight Confirmed : {best_epoch+1}'th epoch")


def train(model: nn.Module, train_loader, optimizer: torch.optim.Optimizer, loss_fun):
    model.train()
    for cate_x, cont_x, mask, target in train_loader:
        optimizer.zero_grad()
        output = model(cate_x, cont_x, mask)

        acc = accuracy_score(y_true=target, y_pred=output > 0.5)
        auc = roc_auc_score(y_true=target, y_score=output)
        
        loss = loss_fun(output, target)
        loss.backward()
        optimizer.step()
    
    logger.info("TRAIN AUC : %.4f ACC : %.4f LOSS : %.4f", auc, acc, loss.item())
    return auc, acc, loss


def validate(model: nn.Module, valid_loader):
    model.eval()
    with torch.no_grad():
        for cate_x, cont_x, mask, target in valid_loader:
            output = model(cate_x, cont_x, mask)

    acc = accuracy_score(y_true=target, y_pred=output > 0.5)
    auc = roc_auc_score(y_true=target, y_score=output)

    logger.info("VALID AUC : %.4f ACC : %.4f", auc, acc)
    return auc, acc


def inference(model: nn.Module, data: dict, output_dir: str):
    model.eval()
    with torch.no_grad():
        pred = model.predict_link(edge_index=data["edge"], prob=True)
        
    logger.info("Saving Result ...")
    pred = pred.detach().cpu().numpy()
    os.makedirs(name=output_dir, exist_ok=True)
    write_path = os.path.join(output_dir, "submission.csv")
    pd.DataFrame({"prediction": pred}).to_csv(path_or_buf=write_path, index_label="id")
    logger.info("Successfully saved submission as %s", write_path)
