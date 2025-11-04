# src/dataset_builder/shard_writer.py
import os, math, json
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch  # 新增：用于保存 .pt 分片

from .io_reader import ensure_dir


def _part_name(i: int) -> str:
    return f"part-{i:05d}"


def write_index(df_idx: pd.DataFrame, out_dir: str) -> str:
    ensure_dir(out_dir)
    path_parquet = os.path.join(out_dir, "index.parquet")
    try:
        df_idx.to_parquet(path_parquet, index=False)
        return path_parquet
    except Exception:
        # fallback to csv
        path_csv = os.path.join(out_dir, "index.csv")
        df_idx.to_csv(path_csv, index=False)
        return path_csv


def write_split_indices(df_all: pd.DataFrame, out_root: str):
    # write train/val/test index tables separately
    for sp in ["train", "val", "test"]:
        sub = df_all[df_all["split"] == sp].copy()
        write_index(sub, os.path.join(out_root, "index", sp))


def write_shards(
    x5: np.ndarray,
    x3: np.ndarray,
    y: np.ndarray,
    out_dir: str,
    shard_size: int,
    fmt: str = "npz",
) -> Dict[str, List[str]]:
    """
    将数据切分并写出分片。

    参数
    ----
    x5, x3 : (N, C, L) 的 float32 one-hot 数组
    y      : (N,) 的标签数组（会按需要转为 int32 / int64）
    out_dir: 输出目录
    shard_size: 每片样本数
    fmt    : "npz"（默认，三文件/片：utr5/utr3/labels）
             或 "pt"（单文件/片：打包 {"x5","x3","y"}）

    返回
    ----
    当 fmt="npz"：{"utr5": [...], "utr3": [...], "labels": [...]}
    当 fmt="pt" ：{"pt": [...]}
    """
    ensure_dir(out_dir)
    N = x5.shape[0]
    n_parts = math.ceil(N / shard_size)

    if fmt == "npz":
        paths: Dict[str, List[str]] = {"utr5": [], "utr3": [], "labels": []}
    elif fmt == "pt":
        paths = {"pt": []}
    else:
        raise ValueError(f"Unsupported shard format: {fmt}")

    for p in range(n_parts):
        s = p * shard_size
        e = min((p + 1) * shard_size, N)
        pfx = _part_name(p)

        if fmt == "npz":
            p5 = os.path.join(out_dir, f"features_utr5.{pfx}.npz")
            p3 = os.path.join(out_dir, f"features_utr3.{pfx}.npz")
            py = os.path.join(out_dir, f"labels.{pfx}.npz")
            np.savez_compressed(p5, x=x5[s:e])
            np.savez_compressed(p3, x=x3[s:e])
            np.savez_compressed(py, y=y[s:e].astype(np.int32))
            paths["utr5"].append(p5)
            paths["utr3"].append(p3)
            paths["labels"].append(py)

        elif fmt == "pt":
            p_all = os.path.join(out_dir, f"data.{pfx}.pt")
            # 用 torch.save 打包三个张量，避免 NPZ 首迭代解压热点
            obj = {
                "x5": torch.from_numpy(x5[s:e]),  # float32 one-hot
                "x3": torch.from_numpy(x3[s:e]),
                "y": torch.from_numpy(y[s:e].astype(np.int64)),  # 分类索引用 int64
            }
            torch.save(obj, p_all)
            paths["pt"].append(p_all)

    return paths


def write_manifest(
    out_dir: str,
    config_raw: Dict,
    alphabet: List[str],
    shapes: Dict[str, Tuple[int, int]],
    counts: Dict[str, int],
    shard_paths: Dict[str, List[str]],
    seed: int,
    index_fmt: str,
):
    ensure_dir(out_dir)
    man = dict(
        version="0.1.0",
        alphabet=alphabet,
        encoding="onehot",
        shapes=shapes,  # {"utr5": (C,L), "utr3": (C,L)}
        counts=counts,  # {"total": N, "train": n1, ...}
        shards=shard_paths,
        rng_seed=seed,
        index_format=index_fmt,
        storage_format=("pt" if "pt" in shard_paths else "npz"),  # 新增：标注分片存储格式
        config=config_raw,
    )
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
