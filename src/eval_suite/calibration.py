# --- file: src/eval_suite/calibration.py ---
from __future__ import annotations
import numpy as np


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15):
    """ECE for multi-class: 以 max prob 的 bin 进行统计。"""
    y_true = y_true.astype(int)
    y_pred = y_prob.argmax(axis=1)
    conf = y_prob.max(axis=1)
    acc = (y_pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    bin_stats = []
    for i in range(n_bins):
        m, M = bins[i], bins[i+1]
        sel = (conf >= m) & (conf < M) if i < n_bins-1 else (conf >= m) & (conf <= M)
        if sel.sum() == 0:
            bin_stats.append((0, 0.0, 0.0))
            continue
        acc_bin = acc[sel].mean()
        conf_bin = conf[sel].mean()
        w = sel.mean()
        ece += w * abs(acc_bin - conf_bin)
        bin_stats.append((int(sel.sum()), float(acc_bin), float(conf_bin)))
    return float(ece), bin_stats
