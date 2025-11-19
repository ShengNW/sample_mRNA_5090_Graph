#!/usr/bin/env python
"""
Prepare unified feature tables for Real / M1 / M2 / M3 datasets.

Adds/normalizes columns:
  - seq_id (generated for files that lack one)
  - score_pred (renamed from dataset-specific columns)
  - gc (overall GC content of utr5+utr3)
  - novelty (reuse if present, otherwise mark 1 when sequence not seen in predict_eval)
  - organ_name (resolved from manifest when possible)
Merges MFE values (mfe_utr5/mfe_utr3) either from existing derived CSVs or by
running scripts/calc_mfe.py for the new generator outputs.

Outputs:
  data/derived/multi_model/<slug>_features.csv
  data/derived/multi_model/all_models_features.csv
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import yaml


def gc_content(seq: str) -> float:
    seq = (seq or "").upper()
    if not seq:
        return 0.0
    gc = sum(1 for ch in seq if ch in ("G", "C"))
    return gc / len(seq)


def load_manifest_mapping(root: Path) -> Dict[int, str]:
    cfg_path = root / "configs" / "gen_predict.yaml"
    if not cfg_path.exists():
        return {}
    cfg = yaml.safe_load(cfg_path.read_text())
    dataset_dir = cfg.get("dataset_dir")
    if not dataset_dir:
        return {}
    manifest = Path(dataset_dir)
    if not manifest.is_absolute():
        manifest = root / manifest
    manifest = manifest / "manifest.json"
    if not manifest.exists():
        return {}
    data = json.loads(manifest.read_text())
    vocab = data.get("organ_vocab", {})
    return {int(k): v for k, v in vocab.items()}


def ensure_seq_id(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if "seq_id" in df.columns:
        return df
    df = df.copy()
    df.insert(
        0,
        "seq_id",
        [f"{prefix}_{i:06d}" for i in range(len(df))],
    )
    return df


def run_calc_mfe(input_csv: Path, output_csv: Path, num_workers: int | None) -> None:
    if output_csv.exists():
        print(f"[INFO] Reusing existing {output_csv}")
        return
    cmd = [
        "python",
        "scripts/calc_mfe.py",
        "--input",
        str(input_csv),
        "--output",
        str(output_csv),
    ]
    if num_workers is not None:
        cmd += ["--num-workers", str(num_workers)]
    print(f"[INFO] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def compute_novelty(df: pd.DataFrame, known_pairs: set[tuple[str, str]]) -> List[int]:
    if "novelty" in df.columns:
        return df["novelty"].tolist()
    return [
        0 if (u5, u3) in known_pairs else 1
        for u5, u3 in zip(df["utr5"].astype(str), df["utr3"].astype(str))
    ]


def normalize_dataset(
    df: pd.DataFrame,
    *,
    slug: str,
    label: str,
    score_col: str,
    organ_col: str,
    organ_vocab: Dict[int, str],
    real_pairs: set[tuple[str, str]],
) -> pd.DataFrame:
    df = ensure_seq_id(df, slug)
    if score_col not in df.columns:
        raise ValueError(f"{slug}: missing score column '{score_col}'")
    if organ_col not in df.columns:
        raise ValueError(f"{slug}: missing organ column '{organ_col}'")

    df = df.copy()
    df["score_pred"] = pd.to_numeric(df[score_col], errors="coerce")
    if organ_col != "organ_id":
        df["organ_id"] = df[organ_col]
    else:
        df["organ_id"] = df["organ_id"]

    if "gc" not in df.columns:
        df["gc"] = [
            gc_content(u5 + u3) for u5, u3 in zip(df["utr5"], df["utr3"])
        ]

    df["novelty"] = compute_novelty(df, real_pairs)
    df["source"] = label

    def resolve_name(val):
        if isinstance(val, str):
            return val
        try:
            idx = int(val)
        except (ValueError, TypeError):
            return str(val)
        return organ_vocab.get(idx, str(val))

    df["organ_name"] = df["organ_id"].apply(resolve_name)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Worker processes for calc_mfe (default: calc_mfe auto selects)",
    )
    ap.add_argument(
        "--out-dir",
        default="data/derived/multi_model",
        help="Directory to store processed feature tables",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        {
            "slug": "real",
            "label": "Real (eval)",
            "path": root / "data" / "raw" / "predict_eval.csv",
            "score_col": "y_pred",
            "organ_col": "organ_id",
            "mfe_csv": root / "data" / "derived" / "mfe_predict.csv",
        },
        {
            "slug": "m1_topk",
            "label": "M1 top-k",
            "path": root / "data" / "raw" / "generated_topk.csv",
            "score_col": "score_pred",
            "organ_col": "organ_id_target",
            "mfe_csv": root / "data" / "derived" / "mfe_generated.csv",
        },
        {
            "slug": "m2_cvae",
            "label": "M2 cVAE",
            "path": root / "data" / "raw" / "m2_cvae_scored.csv",
            "score_col": "pred",
            "organ_col": "organ_id",
            "mfe_csv": None,
        },
        {
            "slug": "m2_cgan",
            "label": "M2 cGAN",
            "path": root / "data" / "raw" / "m2_cgan_scored.csv",
            "score_col": "pred",
            "organ_col": "organ_id",
            "mfe_csv": None,
        },
        {
            "slug": "m3_rl",
            "label": "M3 RL",
            "path": root / "data" / "raw" / "m3_rl_scored.csv",
            "score_col": "pred",
            "organ_col": "organ_id",
            "mfe_csv": None,
        },
    ]

    real_df = pd.read_csv(root / "data" / "raw" / "predict_eval.csv")
    real_pairs = set(zip(real_df["utr5"], real_df["utr3"]))
    organ_vocab = load_manifest_mapping(root)

    combined = []
    for cfg in datasets:
        path = cfg["path"]
        if not path.exists():
            raise FileNotFoundError(f"{cfg['slug']}: missing file {path}")

        df = pd.read_csv(path)
        df = normalize_dataset(
            df,
            slug=cfg["slug"],
            label=cfg["label"],
            score_col=cfg["score_col"],
            organ_col=cfg["organ_col"],
            organ_vocab=organ_vocab,
            real_pairs=real_pairs,
        )
        tmp_csv = out_dir / f"{cfg['slug']}_base.csv"
        df.to_csv(tmp_csv, index=False)

        if cfg["mfe_csv"]:
            mfe_df = pd.read_csv(cfg["mfe_csv"])
        else:
            mfe_out = out_dir / f"{cfg['slug']}_mfe.csv"
            run_calc_mfe(tmp_csv, mfe_out, args.num_workers)
            mfe_df = pd.read_csv(mfe_out)

        df = df.merge(mfe_df, on="seq_id", how="left")
        feature_path = out_dir / f"{cfg['slug']}_features.csv"
        df.to_csv(feature_path, index=False)
        combined.append(df)
        print(f"[OK] Wrote {feature_path} ({len(df)} rows)")

    all_df = pd.concat(combined, ignore_index=True)
    combined_path = out_dir / "all_models_features.csv"
    all_df.to_csv(combined_path, index=False)
    print(f"[OK] Wrote combined table {combined_path} with {len(all_df)} rows")


if __name__ == "__main__":
    main()
