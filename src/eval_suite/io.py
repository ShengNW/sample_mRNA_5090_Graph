# --- file: src/eval_suite/io.py ---
from __future__ import annotations
import os, glob, json
import numpy as np
import pandas as pd


def save_preds_table(path: str, y_true, y_prob):
    df = pd.DataFrame({'y_true': y_true})
    df = df.join(pd.DataFrame(y_prob, columns=[f'p_{i}' for i in range(y_prob.shape[1])]))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
