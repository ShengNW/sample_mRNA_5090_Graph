import pandas as pd
import numpy as np

def read_csv_required(path, required_cols):
    df = pd.read_csv(path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df

def save_merged_csv(df, path):
    df.to_csv(path, index=False)

def compute_basic_stats(series):
    return {
        "mean": float(np.mean(series)),
        "std": float(np.std(series)),
        "min": float(np.min(series)),
        "max": float(np.max(series)),
        "n": int(series.shape[0])
    }
