import math
import os

import numpy as np
import torch
from torch import nn
from torch.nn.functional import sigmoid
import wandb

from .criterion import get_criterion
from .dataloader import get_loaders
from .metric import get_metric
from .model import LSTM, LSTMATTN, BERT, GRU, GRUATTN
from .optimizer import get_optimizer
from .scheduler import get_scheduler
from .utils import get_logger, logging_conf


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

        wandb.log(dict(epoch=epoch,
                       train_loss_epoch=train_loss,
                       train_auc_epoch=train_auc,
                       train_acc_epoch=train_acc,
                       valid_auc_epoch=auc,
                       valid_acc_epoch=acc))

        if auc > best_auc:
            best_auc = auc
            # nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
            model_to_save = model.module if hasattr(model, "module") else model
            save_checkpoint(state={"epoch": epoch + 1,
                                   "state_dict": model_to_save.state_dict()},
                            model_dir=cfg['model_dir'],
                            model_filename="best_model.pt")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= cfg['patience']:
                logger.info(
                    "EarlyStopping counter: %s out of %s",
                    early_stopping_counter, cfg['patience']
                )
                break

        # scheduler
        if cfg['scheduler'] == "plateau":
            scheduler.step(best_auc)


def train(train_loader: torch.utils.data.DataLoader,
          model: nn.Module,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler._LRScheduler,
          cfg):
    model.train()

    total_preds = []
    total_targets = []
    losses = []
    for step, batch in enumerate(train_loader):
        batch = {k: v.to(cfg['device']) for k, v in batch.items()}
        preds = model(**batch)
        targets = batch["correct"]

        loss = compute_loss(preds=preds, targets=targets)
        update_params(loss=loss, model=model, optimizer=optimizer,
                      scheduler=scheduler, cfg=cfg)

        if step % cfg['log_steps'] == 0:
            logger.info("Training steps: %s Loss: %.4f", step, loss.item())

        # predictions
        preds = sigmoid(preds[:, -1])
        targets = targets[:, -1]

        total_preds.append(preds.detach())
        total_targets.append(targets.detach())
        losses.append(loss)

    total_preds = torch.concat(total_preds).cpu().numpy()
    total_targets = torch.concat(total_targets).cpu().numpy()

    # Train AUC / ACC
    auc, acc = get_metric(targets=total_targets, preds=total_preds)
    loss_avg = sum(losses) / len(losses)
    logger.info("TRAIN AUC : %.4f ACC : %.4f", auc, acc)
    return auc, acc, loss_avg


def validate(valid_loader: nn.Module, model: nn.Module, cfg):
    model.eval()

    total_preds = []
    total_targets = []
    for step, batch in enumerate(valid_loader):
        batch = {k: v.to(cfg['device']) for k, v in batch.items()}
        preds = model(**batch)
        targets = batch["correct"]

        # predictions
        preds = sigmoid(preds[:, -1])
        targets = targets[:, -1]

        total_preds.append(preds.detach())
        total_targets.append(targets.detach())

    total_preds = torch.concat(total_preds).cpu().numpy()
    total_targets = torch.concat(total_targets).cpu().numpy()

    # Train AUC / ACC
    auc, acc = get_metric(targets=total_targets, preds=total_preds)
    logger.info("VALID AUC : %.4f ACC : %.4f", auc, acc)
    return auc, acc


def inference(cfg, test_data: np.ndarray, model: nn.Module) -> None:
    model.eval()
    _, test_loader = get_loaders(cfg=cfg, train=None, valid=test_data)

    total_preds = []
    for step, batch in enumerate(test_loader):
        batch = {k: v.to(cfg['device']) for k, v in batch.items()}
        preds = model(**batch)

        # predictions
        preds = sigmoid(preds[:, -1])
        preds = preds.cpu().detach().numpy()
        total_preds += list(preds)

    write_path = os.path.join(cfg['output_dir'], "submission.csv")
    os.makedirs(name=cfg['output_dir'], exist_ok=True)
    with open(write_path, "w", encoding="utf8") as w:
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write("{},{}\n".format(id, p))
    logger.info("Successfully saved submission as %s", write_path)


def get_model(cfg) -> nn.Module:
    model_cfg = dict(
        hidden_dim=cfg['hidden_dim'],
        n_layers=cfg['n_layers'],
        n_tests=cfg['n_tests'],
        n_questions=cfg['n_questions'],
        n_tags=cfg['n_tags'],
        n_heads=cfg['n_heads'],
        drop_out=cfg['drop_out'],
        max_seq_len=cfg['max_seq_len'],
    )
    try:
        model_name = cfg['model'].lower()
        model = {
            "lstm": LSTM,
            "lstmattn": LSTMATTN,
            "bert": BERT,
            "gru" : GRU,
            "gruattn" : GRUATTN,
        }.get(model_name)(**model_cfg)
    except KeyError:
        logger.warn("No model name %s found", model_name)
    except Exception as e:
        logger.warn("Error while loading %s with cfg: %s", model_name, model_cfg)
        raise e
    return model

def compute_loss(preds: torch.Tensor, targets: torch.Tensor):
    """
    loss계산하고 parameter update
    cfg :
        preds   : (batch_size, max_seq_len)
        targets : (batch_size, max_seq_len)

    """
    loss = get_criterion(pred=preds, target=targets.float())

    # 마지막 시퀀드에 대한 값만 loss 계산
    loss = loss[:, -1]
    loss = torch.mean(loss)
    return loss


def update_params(loss: torch.Tensor,
                  model: nn.Module,
                  optimizer: torch.optim.Optimizer,
                  scheduler: torch.optim.lr_scheduler._LRScheduler,
                  cfg):
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad'])
    if cfg['scheduler'] == "linear_warmup":
        scheduler.step()
    optimizer.step()
    optimizer.zero_grad()


def save_checkpoint(state: dict, model_dir: str, model_filename: str) -> None:
    """ Saves checkpoint to a given directory. """
    save_path = os.path.join(model_dir, model_filename)
    logger.info("saving model as %s...", save_path)
    os.makedirs(model_dir, exist_ok=True)
    torch.save(state, save_path)


def load_model(cfg):
    model_path = os.path.join(cfg['model_dir'], cfg['model_name'])
    logger.info("Loading Model from: %s", model_path)
    load_state = torch.load(model_path)
    model = get_model(cfg)

    # load model state
    model.load_state_dict(load_state["state_dict"], strict=True)
    logger.info("Successfully loaded model state from: %s", model_path)
    return model
