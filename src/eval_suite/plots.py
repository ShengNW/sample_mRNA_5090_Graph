# --- file: src/eval_suite/plots.py ---
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt


def save_confusion_matrix(fig_path, cm, class_names=None):
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.figure()
    plt.imshow(cm)
    plt.colorbar()
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(fig_path, bbox_inches='tight', dpi=180)
    plt.close()


def save_reliability_diagram(fig_path, bin_stats):
    # bin_stats: [(count, acc, conf), ...]
    acc = [b[1] for b in bin_stats]
    conf = [b[2] for b in bin_stats]
    plt.figure()
    plt.plot([0,1],[0,1])
    plt.step(conf, acc, where='mid')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram')
    plt.savefig(fig_path, bbox_inches='tight', dpi=180)
    plt.close()
