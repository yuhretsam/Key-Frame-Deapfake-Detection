from typing import Dict

import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve


def compute_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0

    eer = 0.0
    if len(set(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        try:
            eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
        except ValueError:
            eer = float(np.nan)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "eer": float(eer),
    }
