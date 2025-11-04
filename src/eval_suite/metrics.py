# --- file: src/eval_suite/metrics.py ---
from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix


def basic_metrics(y_true, y_prob):
    y_pred = np.argmax(y_prob, axis=1)
    res = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro')),
    }
    # 容错：当某些类在 y_true 中缺失时，sklearn 的 multi-class AUROC 需要处理
    try:
        res['macro_auroc'] = float(roc_auc_score(y_true, y_prob, multi_class='ovo', average='macro'))
    except Exception:
        res['macro_auroc'] = float('nan')
    try:
        res['macro_auprc'] = float(average_precision_score(_one_hot(y_true, y_prob.shape[1]), y_prob, average='macro'))
    except Exception:
        res['macro_auprc'] = float('nan')
    res['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    return res


def _one_hot(y: np.ndarray, K:int):
    oh = np.zeros((y.shape[0], K), dtype=float)
    oh[np.arange(y.shape[0]), y.astype(int)] = 1.0
    return oh
