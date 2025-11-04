# -*- coding: utf-8 -*-
"""
Convert an existing *classification* dataset (seq_cnn_v1) into a *(N, 54) multi-task regression* dataset.

Usage
-----
python -m src.dataset_builder.build_regression_from_classification \
  --config configs/dataset.yaml \
  --src_dir data/processed/seq_cnn_v1 \
  --out_dir data/processed/seq_cnn_v1_reg \
  --ycol tpm \
  --transform log1p \
  --agg mean

Notes
-----
- Keeps the same sample order as the classification build (N unchanged).
  For each row (duplicate across tissues), we attach the full 54-dim vector for its UTR pair.
- Copies `index/` from src_dir to out_dir, so train/val/test split remains identical.
- Only rewrites shard `.pt` files to replace `y` with float32 of shape (n_i, 54).
- Organ order is taken from `{src_dir}/label_mapping.json`. If missing, falls back to sorted unique organ_id in the parquet.
- The 54 number is not hard-coded; it uses the size of the label mapping it finds (expected 54).

Requires
--------
pandas, pyarrow (to read parquet).

"""
import os, sys, json, math, shutil, glob, argparse
from typing import Dict, Any, List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

# Reuse your dataset builder utilities
from .config import load_config
from .preprocess import clean_seq
from .truncation import truncate_utr5, truncate_utr3

def _read_label_mapping(src_dir: str, df: pd.DataFrame, col_organ: str) -> Tuple[Dict[str,int], List[str]]:
    lm_path = os.path.join(src_dir, "label_mapping.json")
    if os.path.exists(lm_path):
        with open(lm_path, "r", encoding="utf-8") as f:
            mp = json.load(f)
        # ensure deterministic order by class index
        inv = sorted([(v, k) for k, v in mp.items()], key=lambda x: x[0])
        classes = [k for _, k in inv]
        return mp, classes
    # fallback: build from df
    classes = sorted(list(map(str, pd.unique(df[col_organ]))))
    mp = {c:i for i,c in enumerate(classes)}
    return mp, classes

def _guess_y_column(df: pd.DataFrame, preferred: str = None) -> str:
    if preferred and preferred in df.columns:
        return preferred
    candidates = ["tpm", "TPM", "tpm_mean", "TPM_mean", "expr", "expression", "Expression", "y_reg", "y", "value"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find expression column. Available: {list(df.columns)}. "
                     f"Pass --ycol explicitly.")

def _apply_transform(x: np.ndarray, name: str) -> np.ndarray:
    name = (name or "log1p").lower()
    if name in ("none","identity","id"):
        return x.astype(np.float32)
    if name in ("log1p","log+1","log1"):
        return np.log1p(x).astype(np.float32)
    if name in ("sqrt","root"):
        return np.sqrt(x).astype(np.float32)
    raise ValueError(f"Unknown transform: {name}")

def _agg_update(curr: float, new: float, how: str) -> float:
    # For simplicity we implement mean and max in a streaming fashion.
    if how == "max":
        if curr is None:
            return new
        return max(curr, new)
    # default mean
    if curr is None:
        return new
    # we don't keep counts per cell to keep code short; assume no heavy duplicates
    # if you expect many duplicates per (key,tissue), change to running-mean with count.
    return 0.5 * (curr + new)

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _load_pt_sizes(pt_paths: List[str]) -> List[int]:
    sizes = []
    for p in pt_paths:
        obj = torch.load(p, map_location="cpu")
        n = int(obj["x5"].shape[0])
        sizes.append(n)
    return sizes

def _write_manifest(out_dir: str, shapes: Dict[str, Tuple[int,int]], counts: Dict[str,int], pt_paths: List[str], cfg_raw: Dict[str,Any]) -> None:
    man = dict(
        version="0.1.0",
        alphabet=list("ACGTN"),
        encoding="onehot",
        shapes=shapes,  # {"utr5": (C,L), "utr3": (C,L)}
        counts=counts,  # {"total": N, "train": n1, ...}
        shards=[{"path": os.path.relpath(p, out_dir), "size": None} for p in pt_paths],
        rng_seed=int(cfg_raw.get("split",{}).get("seed", 2024)),
        index_format="parquet(csv-fallback)",
        storage_format="pt",
        config=cfg_raw,
    )
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)

def _read_split_counts(src_dir: str) -> Dict[str,int]:
    # Try to read train/val/test index sizes
    def _nrows(pq):
        try:
            import pandas as _pd
            return int(_pd.read_parquet(pq).shape[0])
        except Exception:
            import pandas as _pd
            return int(_pd.read_csv(pq.replace(".parquet",".csv")).shape[0])
    root = os.path.join(src_dir, "index")
    out = {}
    for sp in ["train","val","test"]:
        ip = os.path.join(root, sp, "index.parquet")
        if os.path.exists(ip):
            out[sp] = _nrows(ip)
        else:
            cp = os.path.join(root, sp, "index.csv")
            out[sp] = _nrows(cp) if os.path.exists(cp) else 0
    out["total"] = sum(out.values())
    return out

def build():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dataset.yaml")
    ap.add_argument("--src_dir", default="data/processed/seq_cnn_v1")  # classification dataset directory
    ap.add_argument("--out_dir", default="data/processed/seq_cnn_v1_reg")  # regression output directory
    ap.add_argument("--ycol", default=None, help="expression column name in parquet; if omitted, auto-guess")
    ap.add_argument("--transform", default="log1p", choices=["log1p","none","sqrt"])
    ap.add_argument("--agg", default="mean", choices=["mean","max"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    pq_path = cfg.input_parquet
    col_organ = cfg.col_organ
    col_u5 = cfg.col_utr5
    col_u3 = cfg.col_utr3
    # Read parquet
    df = pd.read_parquet(pq_path)
    ycol = _guess_y_column(df, preferred=cfg.raw.get("colmap",{}).get("y_reg") or args.ycol)
    print(f"[info] using ycol={ycol}")
    # Label mapping / class order
    lab_map, classes = _read_label_mapping(args.src_dir, df, col_organ)
    n_cls = len(classes)
    print(f"[info] n_classes (expected 54) = {n_cls}")
    # Clean + truncate to get the SAME UTR strings as features were built from
    L5 = cfg.trunc_utr5.length
    L3 = cfg.trunc_utr3.length
    s5 = df[col_u5].astype(str).map(clean_seq).tolist()
    s3 = df[col_u3].astype(str).map(clean_seq).tolist()
    t5, t3, keys = [], [], []
    for a,b in zip(s5, s3):
        u5, _ = truncate_utr5(a, L5, cfg.trunc_utr5.strategy)
        u3, _ = truncate_utr3(b, L3, cfg.trunc_utr3.strategy)
        t5.append(u5); t3.append(u3); keys.append(u5 + "|" + u3)
    # Build per-key 54-dim vector by aggregation
    print("[info] building key -> vector map ...")
    # Initialize dict[key] = np.full(n_cls, np.nan) and fill
    vecs: Dict[str, np.ndarray] = {}
    for i in range(len(df)):
        k = keys[i]
        organ = str(df[col_organ].iloc[i])
        cls_idx = lab_map[organ]
        val = float(df[ycol].iloc[i])
        if k not in vecs:
            vecs[k] = np.full((n_cls,), np.nan, dtype=np.float32)
        curr = vecs[k][cls_idx]
        if np.isnan(curr):
            vecs[k][cls_idx] = val
        else:
            if args.agg == "max":
                vecs[k][cls_idx] = max(curr, val)
            else:
                vecs[k][cls_idx] = 0.5 * (curr + val)  # simple mean of duplicates
    # Replace NaN (missing tissues) with 0 before transform
    for k in vecs:
        v = vecs[k]
        v[np.isnan(v)] = 0.0
        vecs[k] = _apply_transform(v, args.transform)
    # Per-row labels (replicate the vector for each row order)
    print("[info] assembling per-row labels with preserved order ...")
    Y = np.zeros((len(df), n_cls), dtype=np.float32)
    for i, k in enumerate(keys):
        Y[i] = vecs[k]
    # Ready to rewrite shards
    src_dir = args.src_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    # Copy index/ directory as-is
    src_index_dir = os.path.join(src_dir, "index")
    out_index_dir = os.path.join(out_dir, "index")
    if os.path.exists(out_index_dir):
        shutil.rmtree(out_index_dir)
    if os.path.exists(src_index_dir):
        shutil.copytree(src_index_dir, out_index_dir)
    else:
        print("[warn] source index/ not found; regression dataset will miss split indices.")
    # Enumerate shard files
    pt_paths = sorted(glob.glob(os.path.join(src_dir, "data.part-*.pt")))
    if not pt_paths:
        raise FileNotFoundError(f"No PT shards under {src_dir}. Expected files like data.part-00000.pt")
    sizes = [int(torch.load(p, map_location="cpu")["x5"].shape[0]) for p in pt_paths]
    cum = np.cumsum([0] + sizes)
    print(f"[info] found {len(pt_paths)} parts; total N={cum[-1]}")
    if cum[-1] != len(df):
        print(f"[warn] shard sizes ({cum[-1]}) != parquet rows ({len(df)}). Proceeding, but verify your inputs.")
    # Write new shards
    out_pt_paths: List[str] = []
    for pi, p in enumerate(pt_paths):
        s, e = int(cum[pi]), int(cum[pi+1])
        obj = torch.load(p, map_location="cpu")
        x5 = obj["x5"]
        x3 = obj["x3"]
        y = torch.from_numpy(Y[s:e])  # float32 (n_i, n_cls)
        new_obj = {"x5": x5, "x3": x3, "y": y}
        out_p = os.path.join(out_dir, os.path.basename(p))
        torch.save(new_obj, out_p)
        out_pt_paths.append(out_p)
    # Manifest
    # Infer shapes from first shard
    fobj = torch.load(out_pt_paths[0], map_location="cpu")
    C5, L5 = int(fobj["x5"].shape[1]), int(fobj["x5"].shape[2])
    C3, L3 = int(fobj["x3"].shape[1]), int(fobj["x3"].shape[2])
    shapes = {"utr5": (C5, L5), "utr3": (C3, L3)}
    counts = _read_split_counts(src_dir)
    _write_manifest(out_dir, shapes, counts, out_pt_paths, cfg.raw)
    # Also write the label mapping we used (so downstream knows the tissue order)
    with open(os.path.join(out_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump({c:i for i,c in enumerate(classes)}, f, ensure_ascii=False, indent=2)
    print(f"[OK] Regression dataset written to: {out_dir}")

if __name__ == "__main__":
    build()
