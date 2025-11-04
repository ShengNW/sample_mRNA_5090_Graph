# -*- coding: utf-8 -*-
from pathlib import Path
"""
from pathlib import Path
Compute per-task (54-dim) mean/std for regression targets.
from pathlib import Path
Saves to <dataset_dir>/y_stats.json so the trainer can standardize targets.
from pathlib import Path

from pathlib import Path
Usage:
from pathlib import Path
  python compute_y_stats.py --dataset_dir data/processed/seq_cnn_v1_reg
from pathlib import Path
"""
from pathlib import Path
import os, glob, json, argparse
from pathlib import Path
import torch
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    args = ap.parse_args()

    pts = sorted(glob.glob(os.path.join(args.dataset_dir, "data.part-*.pt")))
    if not pts:
        raise FileNotFoundError(f"No shards found under {args.dataset_dir}")
    mu = None
    s2 = None
    n = 0
    for p in pts:
        obj = torch.load(p, map_location="cpu")
        y = obj["y"].float()  # (n_i, 54)
        y_np = y.numpy()
        if mu is None:
            mu = y_np.mean(axis=0)
            s2 = y_np.var(axis=0)
            n = y_np.shape[0]
        else:
            n_i = y_np.shape[0]
            mu_new = (mu * n + y_np.sum(axis=0)) / (n + n_i)
            # Merge variances (two-pass safe merge)
            s2 = (s2 * n + y_np.var(axis=0) * n_i + (n * n_i) / (n + n_i) * (mu - mu_new)**2) / (n + n_i)
            mu = mu_new
            n += n_i

    std = (s2 ** 0.5).clip(min=1e-8)
    out = {"mean": mu.tolist(), "std": std.tolist(), "N": int(n)}
    with open(Path(args.dataset_dir) / "y_stats.json", "w", encoding="utf-8") as f:
        import json
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {args.dataset_dir}/y_stats.json with N={n}")

if __name__ == "__main__":
    main()
