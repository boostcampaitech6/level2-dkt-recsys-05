from typing import Tuple

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def get_metric(targets: np.ndarray, preds: np.ndarray) -> Tuple[float]:
    auc = roc_auc_score(y_true=targets, y_score=preds)
    acc = accuracy_score(y_true=targets, y_pred=np.where(preds >= 0.5, 1, 0))
    return auc, acc
