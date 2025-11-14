"""
Fig2：预测器表现与可解释性图（散点、残差直方、可选 UMAP）。
仅依赖 pandas/matplotlib/sklearn；SHAP 另行可选。
"""
import argparse, os, json, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def scatter_true_pred(ax, y_true, y_pred):
    ax.scatter(y_true, y_pred, s=8, alpha=0.6)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title("Prediction vs Truth")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.input)
    y_true = df["y_true"].values
    y_pred = df["y_pred"].values

    # 1) 散点
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    scatter_true_pred(ax, y_true, y_pred)
    fig.savefig(os.path.join(args.outdir, "fig2A_scatter.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(args.outdir, "fig2A_scatter.svg"), bbox_inches="tight")
    plt.close(fig)

    # 2) 残差直方
    resid = y_pred - y_true
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    ax.hist(resid, bins=40)
    ax.set_title("Residuals")
    fig.savefig(os.path.join(args.outdir, "fig2B_residual_hist.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(args.outdir, "fig2B_residual_hist.svg"), bbox_inches="tight")
    plt.close(fig)

    # 3) 可选：简单嵌入（PCA 代替 UMAP 以去依赖）
    # 若存在一些序列学特征列，演示嵌入
    feature_cols = [c for c in df.columns if c.startswith("kmer_") or c in ["gc","mfe_utr5","mfe_utr3","rbp_hits_total"]]
    if len(feature_cols) >= 2:
        X = df[feature_cols].fillna(0).values
        X = StandardScaler().fit_transform(X)
        emb = PCA(n_components=2).fit_transform(X)
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        ax.scatter(emb[:,0], emb[:,1], s=6, alpha=0.6)
        ax.set_title("Embedding (PCA)")
        fig.savefig(os.path.join(args.outdir, "fig2C_embedding.png"), dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(args.outdir, "fig2C_embedding.svg"), bbox_inches="tight")
        plt.close(fig)

if __name__ == "__main__":
    main()
