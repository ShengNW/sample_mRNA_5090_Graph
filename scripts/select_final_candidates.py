#!/usr/bin/env python
"""
Select top candidates per generator using GC/MFE/novelty filters.
Outputs CSVs under data/derived/multi_model/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_SOURCES = {
    "m1_topk": "M1 top-k",
    "m2_cvae": "M2 cVAE",
    "m2_cgan": "M2 cGAN",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--features",
        default="data/derived/multi_model/all_models_features.csv",
        help="Path to the combined feature table.",
    )
    ap.add_argument("--gc-min", type=float, default=0.4)
    ap.add_argument("--gc-max", type=float, default=0.65)
    ap.add_argument("--mfe5-min", type=float, default=-250.0)
    ap.add_argument("--mfe5-max", type=float, default=-20.0)
    ap.add_argument("--mfe3-min", type=float, default=-1000.0)
    ap.add_argument("--mfe3-max", type=float, default=-50.0)
    ap.add_argument("--novelty-min", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument(
        "--out-dir",
        default="data/derived/multi_model",
        help="Output directory for the candidate CSVs.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / args.features)

    mask = (
        df["gc"].between(args.gc_min, args.gc_max)
        & df["mfe_utr5"].between(args.mfe5_min, args.mfe5_max)
        & df["mfe_utr3"].between(args.mfe3_min, args.mfe3_max)
        & (df["novelty"] >= args.novelty_min)
    )
    filtered = df[mask].copy()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_path = out_dir / "filtered_candidates_all.csv"
    filtered.to_csv(all_path, index=False)
    print(f"[OK] Wrote filtered set ({len(filtered)} rows) to {all_path}")

    for slug, label in DEFAULT_SOURCES.items():
        sub = (
            filtered[filtered["source"] == label]
            .sort_values("score_pred", ascending=False)
            .head(args.top_k)
        )
        out_path = out_dir / f"final_candidates_{slug}.csv"
        sub.to_csv(out_path, index=False)
        print(f"[{label}] selected {len(sub)} rows -> {out_path}")


if __name__ == "__main__":
    main()
