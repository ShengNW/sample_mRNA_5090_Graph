"""
Fig3：生成序列性质分布与新颖性-表现关系。
输入：generated_topk.csv + 派生特征（mfe/rbp）。
"""
import argparse, os, numpy as np, pandas as pd, matplotlib.pyplot as plt

def hist_compare(ax, arrays, labels, title, bins=40):
    for arr, lb in zip(arrays, labels):
        ax.hist(arr, bins=bins, alpha=0.5, label=lb, histtype='step')
    ax.set_title(title)
    ax.legend()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--mfe", default=None)
    ap.add_argument("--rbp", default=None)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.gen)
    if args.mfe and os.path.exists(args.mfe):
        df = df.merge(pd.read_csv(args.mfe), on="seq_id", how="left")
    if args.rbp and os.path.exists(args.rbp):
        df = df.merge(pd.read_csv(args.rbp), on="seq_id", how="left")

    # A) 性质分布
    for col in ["gc","mfe_utr5","mfe_utr3","rbp_hits_total"]:
        if col in df.columns:
            fig = plt.figure(figsize=(5,4))
            ax = fig.add_subplot(111)
            ax.hist(df[col].dropna().values, bins=40)
            ax.set_title(f"Distribution of {col}")
            fig.savefig(os.path.join(args.outdir, f"fig3A_{col}.png"), dpi=300, bbox_inches="tight")
            fig.savefig(os.path.join(args.outdir, f"fig3A_{col}.svg"), bbox_inches="tight")
            plt.close(fig)

    # B) 新颖性 vs 预测得分
    if "novelty" in df.columns and "score_pred" in df.columns:
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        ax.scatter(df["novelty"].values, df["score_pred"].values, s=6, alpha=0.6)
        ax.set_xlabel("Novelty")
        ax.set_ylabel("Predicted Score")
        ax.set_title("Novelty vs Predicted")
        fig.savefig(os.path.join(args.outdir, "fig3B_novelty_vs_pred.png"), dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(args.outdir, "fig3B_novelty_vs_pred.svg"), bbox_inches="tight")
        plt.close(fig)

    # C) 目标 vs 非目标（如果有 off-target 列）
    if "score_off_target" in df.columns:
        fig = plt.figure(figsize=(5,4))
        ax = fig.add_subplot(111)
        ax.hist(df["score_pred"].dropna().values, bins=40, histtype='step', label="target")
        ax.hist(df["score_off_target"].dropna().values, bins=40, histtype='step', label="off-target")
        ax.set_title("Target vs Off-target")
        ax.legend()
        fig.savefig(os.path.join(args.outdir, "fig3C_target_vs_off.png"), dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(args.outdir, "fig3C_target_vs_off.svg"), bbox_inches="tight")
        plt.close(fig)

if __name__ == "__main__":
    main()
