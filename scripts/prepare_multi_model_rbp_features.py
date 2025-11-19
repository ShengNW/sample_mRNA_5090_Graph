#!/usr/bin/env python3
"""
Aggregate RBP feature tables for real and multi-model generators.
Outputs one CSV with unified columns: seq_id, source, organ_id, score_pred,
rbp_hits_total, rbp_hits_per_kb.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


def ensure_seq_id(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if "seq_id" in df.columns:
        return df
    df = df.copy()
    df.insert(0, "seq_id", [f"{prefix}_{i:06d}" for i in range(len(df))])
    return df


def load_base_features(cfg: Dict[str, str], root: Path) -> pd.DataFrame:
    data_path = root / cfg["data_csv"]
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data CSV: {data_path}")
    df = pd.read_csv(data_path)
    df = ensure_seq_id(df, cfg.get("seq_prefix") or data_path.stem)
    score_col = cfg["score_col"]
    if score_col not in df.columns:
        raise ValueError(f"{data_path} missing score column '{score_col}'")
    base = pd.DataFrame({
        "seq_id": df["seq_id"],
        "score_pred": pd.to_numeric(df[score_col], errors="coerce"),
    })
    organ_col = cfg.get("organ_col")
    if organ_col and organ_col in df.columns:
        base["organ_id"] = df[organ_col]
    elif "organ_id" in df.columns:
        base["organ_id"] = df["organ_id"]
    else:
        base["organ_id"] = None
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="data/derived/all_models_rbp_features.csv",
        help="Output CSV path",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    datasets: List[Dict[str, str]] = [
        {
            "source": "real",
            "label": "Real",
            "data_csv": "data/raw/predict_eval.csv",
            "rbp_csv": "data/derived/rbp_predict.csv",
            "score_col": "y_pred",
            "organ_col": "organ_id",
        },
        {
            "source": "m1",
            "label": "M1 topk",
            "data_csv": "data/raw/generated_topk.csv",
            "rbp_csv": "data/derived/rbp_generated.csv",
            "score_col": "score_pred",
            "organ_col": "organ_id_target",
        },
        {
            "source": "m2_cvae",
            "label": "M2 cVAE",
            "data_csv": "data/raw/m2_cvae_scored.csv",
            "rbp_csv": "data/derived/rbp_m2_cvae.csv",
            "score_col": "pred",
            "organ_col": "organ_id",
        },
        {
            "source": "m2_cgan",
            "label": "M2 cGAN",
            "data_csv": "data/raw/m2_cgan_scored.csv",
            "rbp_csv": "data/derived/rbp_m2_cgan.csv",
            "score_col": "pred",
            "organ_col": "organ_id",
        },
        {
            "source": "m3_rl",
            "label": "M3 RL",
            "data_csv": "data/raw/m3_rl_scored.csv",
            "rbp_csv": "data/derived/rbp_m3_rl.csv",
            "score_col": "pred",
            "organ_col": "organ_id",
        },
    ]

    frames = []
    for cfg in datasets:
        base = load_base_features(cfg, root)
        rbp_path = root / cfg["rbp_csv"]
        if not rbp_path.exists():
            raise FileNotFoundError(f"Missing RBP CSV: {rbp_path}")
        rbp = pd.read_csv(rbp_path)
        merged = rbp.merge(base, on="seq_id", how="left")
        merged.insert(0, "source", cfg["source"])
        merged["source_label"] = cfg["label"]
        frames.append(merged)
        print(f"[INFO] merged {cfg['source']} ({len(merged)} rows)")

    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_df = pd.concat(frames, ignore_index=True)
    cols = [
        "source",
        "source_label",
        "seq_id",
        "organ_id",
        "score_pred",
        "rbp_hits_total",
        "rbp_hits_per_kb",
    ]
    all_df = all_df[cols]
    all_df.to_csv(out_path, index=False)
    print(f"[INFO] wrote {out_path} ({len(all_df)} rows)")


if __name__ == "__main__":
    main()
