import torch
from torch.optim import Adam, AdamW, NAdam


def get_optimizer(model: torch.nn.Module, cfg):
    if cfg['optimizer'] == "adam":
        optimizer = Adam(model.parameters(), lr=cfg['lr'], weight_decay=0.01)

    elif cfg['optimizer'] == "adamW":
        optimizer = AdamW(model.parameters(), lr=cfg['lr'], weight_decay=0.01)

    elif cfg['optimizer'] == "NAdam":
        optimizer = NAdam(model.parameters(), lr=cfg['lr'], weight_decay=0.01)
    # 모든 parameter들의 grad값을 0으로 초기화
    optimizer.zero_grad()
    return optimizer
