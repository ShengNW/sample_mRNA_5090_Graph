# src/dataset_builder/splitter.py
from typing import Tuple, Dict
import numpy as np
import pandas as pd
from collections import defaultdict

def stratified_split(
    df: pd.DataFrame, label_col: str, ratios=(0.8, 0.1, 0.1), seed=2024
) -> pd.Series:
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for i, y in enumerate(df[label_col].values):
        groups[y].append(i)
    assign = np.array([""] * len(df), dtype=object)
    for y, idxs in groups.items():
        idxs = np.array(idxs, dtype=int)
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(round(ratios[0] * n))
        n_val = int(round(ratios[1] * n))
        train_idx = idxs[:n_train]
        val_idx = idxs[n_train:n_train+n_val]
        test_idx = idxs[n_train+n_val:]
        assign[train_idx] = "train"
        assign[val_idx] = "val"
        assign[test_idx] = "test"
    return pd.Series(assign, index=df.index, name="split")

def materialize_split(df: pd.DataFrame, col_split: str, label_col: str, ratios, seed) -> pd.Series:
    if col_split and col_split in df.columns:
        # Normalize to train/val/test
        s = df[col_split].astype(str).str.lower()
        s = s.replace({"dev": "val", "valid": "val", "validation": "val"})
        return s
    return stratified_split(df, label_col, ratios=ratios, seed=seed)
