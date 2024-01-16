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
from tqdm import tqdm 

from .utils import get_logger, logging_conf


logger = get_logger(logger_conf=logging_conf)


def build(cfg):
    model = TransformerModel(cfg)

    if cfg.train:
        pass

    else:
        model_path = os.path.join(cfg.model_dir, cfg.model_name)
        model_state = torch.load(model_path)
        model.load_state_dict(model_state["model"])

    model = model.to(cfg.device)
    return model


def run(model: nn.Module, prepared, cfg):
    model.train()
    model_dir=cfg.model_dir
    os.makedirs(name=model_dir, exist_ok=True)
    
    train_data = prepared['train_data']
    valid_data = prepared['valid_data']
    train_cfg = prepared['train_cfg']
    valid_cfg = prepared['valid_cfg']

    train_data = TransformerDataset(train_data, train_cfg)
    valid_data = TransformerDataset(valid_data, valid_cfg)

    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=cfg.batch_size, shuffle=True)

    n_epochs=cfg.n_epochs
    learning_rate=cfg.lr
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    loss_fun = nn.BCEWithLogitsLoss()

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
    total_loss = 0.0
    target_list = []
    output_list = []

    for cate_x, cont_x, mask, target in tqdm(train_loader, mininterval=1):
        
        optimizer.zero_grad()
        output = model(cate_x, cont_x, mask)

        loss = loss_fun(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        target_list.append(target.detach())
        output_list.append(output.detach())

    target_list = torch.concat(target_list).cpu().numpy()
    output_list = torch.concat(output_list).cpu().numpy()

    acc = accuracy_score(y_true=target_list, y_pred=output_list > 0.5)
    auc = roc_auc_score(y_true=target_list, y_score=output_list)

    average_loss = total_loss / len(train_loader)
    logger.info("TRAIN AUC : %.4f ACC : %.4f LOSS : %.4f", auc, acc, average_loss)

    return auc, acc, average_loss


def validate(model: nn.Module, valid_loader):
    model.eval()
    with torch.no_grad():
        target_list = []
        output_list = []
        for cate_x, cont_x, mask, target in tqdm(valid_loader, mininterval=1):
            output = model(cate_x, cont_x, mask)
            target_list.append(target.detach())
            output_list.append(output.detach())
        
    target_list = torch.concat(target_list).cpu().numpy()
    output_list = torch.concat(output_list).cpu().numpy()

    acc = accuracy_score(y_true=target_list, y_pred=output_list > 0.5)
    auc = roc_auc_score(y_true=target_list, y_score=output_list)

    logger.info("VALID AUC : %.4f ACC : %.4f", auc, acc)
    return auc, acc


def inference(cfg, model: nn.Module, prepared, output_dir: str):
    test_data = prepared['test_data']
    test_cfg = prepared['test_cfg']

    test_data = TransformerDataset(test_data, test_cfg)
    test_loader = DataLoader(test_data, batch_size=cfg.batch_size, shuffle=False)

    model.eval()
    with torch.no_grad():
        output_list = []
        for cate_x, cont_x, mask, target in tqdm(test_loader, mininterval=1):
            output = model(cate_x, cont_x, mask)
            output_list.append(output.cpu().detach().numpy())
        
    output_list = np.concatenate(output_list).flatten()

    logger.info("Saving Result ...")
    os.makedirs(name=output_dir, exist_ok=True)
    write_path = os.path.join(output_dir, "submission.csv")
    pd.DataFrame({"prediction": output_list}).to_csv(path_or_buf=write_path, index_label="id")

    logger.info("Successfully saved submission as %s", write_path)
