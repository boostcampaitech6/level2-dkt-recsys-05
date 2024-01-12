import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_linear_schedule_with_warmup

['optimizer']
def get_scheduler(optimizer: torch.optim.Optimizer, cfg):
    if cfg['scheduler'] == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5, mode="max", verbose=True
        )
    elif cfg['scheduler'] == "linear_warmup":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg['warmup_steps'],
            num_training_steps=cfg['total_steps'],
        )
    return scheduler
