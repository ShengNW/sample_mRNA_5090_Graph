#!/usr/bin/env python3
"""
Plot upgraded Fig2 focusing on RBP motif statistics across models.
Reads data/derived/all_models_rbp_features.csv and draws:
  1. Histogram of rbp_hits_total per source.
  2. Histogram of rbp_hits_per_kb per source.
  3. Scatter of rbp_hits_total vs score_pred.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORDER: List[Tuple[str, str, str]] = [
    ("real", "Real", "#1f77b4"),
    ("m1", "M1 top-k", "#ff7f0e"),
    ("m2_cvae", "M2 cVAE", "#2ca02c"),
    ("m2_cgan", "M2 cGAN", "#d62728"),
    ("m3_rl", "M3 RL", "#9467bd"),
]


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def plot_hist(df: pd.DataFrame, feature: str, xlabel: str, out_path: Path, bins: int = 80) -> None:
    sub = df[["source", feature]].dropna()
    if sub.empty:
        print(f"[WARN] no data for {feature}")
        return
    overall = sub[feature]
    rng = (overall.min(), overall.max())
    plt.figure(figsize=(7, 4))
    for key, label, color in ORDER:
        vals = sub[sub["source"] == key][feature]
        if vals.empty:
            continue
        plt.hist(
            vals,
            bins=bins,
            range=rng,
            density=True,
            histtype="step",
            linewidth=1.5,
            label=label,
            color=color,
        )
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"[OK] wrote {out_path}")


def plot_score_scatter(df: pd.DataFrame, out_path: Path, max_points: int = 2000) -> None:
    plt.figure(figsize=(6.5, 5))
    for key, label, color in ORDER:
        sub = df[df["source"] == key][["rbp_hits_total", "score_pred"]].dropna()
        if sub.empty:
            continue
        if len(sub) > max_points:
            sub = sub.sample(max_points, random_state=0)
        plt.scatter(
            sub["rbp_hits_total"],
            sub["score_pred"],
            s=10,
            alpha=0.5,
            label=label,
            color=color,
        )
    plt.xlabel("RBP hits total")
    plt.ylabel("score_pred")
    plt.legend(markerscale=1.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"[OK] wrote {out_path}")


def summarize(df: pd.DataFrame, feature: str) -> None:
    print(f"\n=== {feature} ===")
    for key, label, _ in ORDER:
        vals = df[df["source"] == key][feature].dropna()
        if vals.empty:
            print(f"{label}: n=0")
            continue
        print(
            f"{label}: n={len(vals)}, mean={vals.mean():.2f}, stdev={vals.std():.2f}, "
            f"q10={vals.quantile(0.1):.2f}, q50={vals.quantile(0.5):.2f}, q90={vals.quantile(0.9):.2f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="data/derived/all_models_rbp_features.csv",
        help="Aggregated RBP feature table",
    )
    ap.add_argument("--outdir", default="figs/outputs", help="Directory for plots")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    df = load_data(root / args.input)
    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    plot_hist(df, "rbp_hits_total", "RBP hits total", outdir / "fig2_rbp_hits_total.png")
    plot_hist(df, "rbp_hits_per_kb", "RBP hits per kb", outdir / "fig2_rbp_hits_per_kb.png")
    plot_score_scatter(df, outdir / "fig2_rbp_hits_vs_score.png")

    summarize(df, "rbp_hits_total")
    summarize(df, "rbp_hits_per_kb")


if __name__ == "__main__":
    main()
