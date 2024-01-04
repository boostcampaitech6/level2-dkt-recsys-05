import torch


def get_criterion(pred: torch.Tensor, target: torch.Tensor):
    loss = torch.nn.BCEWithLogitsLoss(reduction="none")
    return loss(pred, target)
