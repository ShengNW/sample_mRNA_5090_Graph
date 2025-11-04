# src/dataset_builder/cli.py
import argparse, os, sys, json
import numpy as np
import pandas as pd

from .config import load_config
from .io_reader import read_parquet_with_snapshot, ensure_dir, dump_json
from .validators import basic_scan
from .preprocess import clean_seq
from .truncation import truncate_utr5, truncate_utr3
from .encoding import build_alphabet_map, onehot_encode
from .splitter import materialize_split
from .label_mapping import build_label_mapping, dump_label_mapping
from .shard_writer import write_shards, write_manifest, write_split_indices
from .report import write_report_md

def _parse_args():
    p = argparse.ArgumentParser(prog="dataset_builder")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build dataset shards and indices")
    b.add_argument("--config", required=True, help="Path to configs/dataset.yaml")
    return p.parse_args()

def main():
    args = _parse_args()
    if args.cmd == "build":
        cfg = load_config(args.config)
        ensure_dir(cfg.output_dir)

        # 1) Read parquet + snapshot
        df, snapshot = read_parquet_with_snapshot(cfg.input_parquet)

        # 2) Basic scan for report
        scan = basic_scan(df, cfg.col_utr5, cfg.col_utr3, cfg.col_organ)

        # 3) Split materialization
        df["split"] = materialize_split(
            df, cfg.col_split, cfg.stratify_by, cfg.split_ratios, cfg.seed
        )

        # 4) Label mapping (frozen by data)
        mapping = build_label_mapping(df, cfg.col_organ)
        if len(mapping) != 54:
            print(f"[WARN] Detected n_classes={len(mapping)} != 54 (locked). Proceeding.", file=sys.stderr)
        label_map_path = os.path.join(cfg.output_dir, "label_mapping.json")
        dump_label_mapping(label_map_path, mapping)

        # 5) Preprocess + truncate + encode
        alpha_map = build_alphabet_map(cfg.alphabet)
        C = len(alpha_map)
        L5 = cfg.trunc_utr5.length
        L3 = cfg.trunc_utr3.length

        utr5_list, utr3_list, y_list = [], [], []
        pol5, pol3 = [], []
        organ_ids = df[cfg.col_organ].astype(str).tolist()
        y = np.array([mapping[str(k)] for k in organ_ids], dtype=np.int32)

        seq5_raw = df[cfg.col_utr5].astype(str).tolist()
        seq3_raw = df[cfg.col_utr3].astype(str).tolist()

        for s5, s3 in zip(seq5_raw, seq3_raw):
            cs5 = clean_seq(s5)
            cs3 = clean_seq(s3)
            t5, p5 = truncate_utr5(cs5, L5, cfg.trunc_utr5.strategy)
            t3, p3 = truncate_utr3(cs3, L3, cfg.trunc_utr3.strategy)
            x5 = onehot_encode(t5, alpha_map, C)  # (C, L5)
            x3 = onehot_encode(t3, alpha_map, C)  # (C, L3)
            utr5_list.append(x5)
            utr3_list.append(x3)
            pol5.append(p5)
            pol3.append(p3)

        X5 = np.stack(utr5_list, axis=0)  # (N, C, L5)
        X3 = np.stack(utr3_list, axis=0)  # (N, C, L3)

        # 6) Write shards (pass through cfg.output_format)
        shard_dir = os.path.join(cfg.output_dir, "shards")
        paths = write_shards(
            X5, X3, y, shard_dir,
            shard_size=cfg.shard_size,
            fmt=cfg.output_format
        )

        # 7) Build index table (包含 part/local，避免依赖回退路径) 并写出 split indices
        N = len(df)
        sample_id = np.arange(N, dtype=np.int64)
        part_id   = (sample_id // cfg.shard_size).astype(np.int64)
        local_idx = (sample_id %  cfg.shard_size).astype(np.int64)
        idx_df = pd.DataFrame({
            "sample_id": sample_id,
            "part_id": part_id,
            "local_idx": local_idx,
            "organ_id": df[cfg.col_organ].astype(str).values,
            "class_idx": y,
            "split": df["split"].values,
            "utr5_policy": pol5,
            "utr3_policy": pol3,
            "utr5_len": L5,
            "utr3_len": L3,
        })
        write_split_indices(idx_df, cfg.output_dir)

        # 8) Manifest + Report
        counts = {
            "total": int(len(df)),
            "train": int((idx_df["split"]=="train").sum()),
            "val": int((idx_df["split"]=="val").sum()),
            "test": int((idx_df["split"]=="test").sum()),
        }
        shapes = {"utr5": (int(X5.shape[1]), int(X5.shape[2])), "utr3": (int(X3.shape[1]), int(X3.shape[2]))}
        index_fmt = "parquet(csv-fallback)"
        write_manifest(cfg.output_dir, cfg.raw, list(alpha_map.keys()), shapes, counts, paths, cfg.seed, index_fmt)

        write_report_md(cfg.output_dir, snapshot, scan, shapes, counts, label_n_expected=54)
        print(f"[OK] Dataset built at: {cfg.output_dir}")

if __name__ == "__main__":
    main()
