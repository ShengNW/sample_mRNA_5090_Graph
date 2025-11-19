#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def get_score(series_df: pd.DataFrame, name: str) -> pd.Series:
    """
    尝试在 DataFrame 里找到预测得分列，并转成 float。
    优先级：score_pred > y_pred > score
    """
    for col in ["score_pred", "y_pred", "score"]:
        if col in series_df.columns:
            s = pd.to_numeric(series_df[col], errors="coerce")
            if s.notna().any():
                print(f"[{name}] 使用列 '{col}' 作为 score_pred")
                return s
    raise RuntimeError(
        f"{name} 表里找不到 score 列（期望列名之一：score_pred / y_pred / score）。"
        f" 实际列: {list(series_df.columns)}"
    )


def main():
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw"
    out_fig_dir = root / "figs" / "outputs"
    out_fig_dir.mkdir(parents=True, exist_ok=True)
    out_csv_dir = root / "outputs" / "phase2" / "m1"
    out_csv_dir.mkdir(parents=True, exist_ok=True)

    # 1）读两张表
    df_real = pd.read_csv(raw_dir / "predict_eval.csv")
    df_gen = pd.read_csv(raw_dir / "generated_topk.csv")

    real_scores = get_score(df_real, "predict_eval")
    gen_scores = get_score(df_gen, "generated_topk")

    # 2）画真实 vs 生成的 score_pred 直方图
    plt.figure()
    bins = 40

    plt.hist(
        real_scores.dropna(),
        bins=bins,
        density=True,
        alpha=0.5,
        label="Real (predict_eval)",
    )
    plt.hist(
        gen_scores.dropna(),
        bins=bins,
        density=True,
        alpha=0.5,
        label="Generated (generated_topk)",
    )

    plt.xlabel("score_pred")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    fig_path = out_fig_dir / "fig_score_pred_real_vs_generated.png"
    plt.savefig(fig_path, dpi=200)
    print(f"[OK] 保存图像到 {fig_path}")

    # 顺便在终端打印一点统计量，方便你写总结
    def summarize(name, s: pd.Series):
        s = s.dropna()
        print(
            f"[{name}] n={len(s)}, "
            f"mean={s.mean():.4f}, std={s.std():.4f}, "
            f"q10={s.quantile(0.1):.4f}, "
            f"q50={s.quantile(0.5):.4f}, "
            f"q90={s.quantile(0.9):.4f}"
        )

    summarize("Real", real_scores)
    summarize("Generated", gen_scores)

    # 3）从 generated_topk 里导出 top-N
    N = 20
    df_gen_sorted = df_gen.copy()
    df_gen_sorted["_score_for_sort"] = gen_scores
    df_gen_sorted = df_gen_sorted.sort_values("_score_for_sort", ascending=False)
    topN = df_gen_sorted.head(N).drop(columns=["_score_for_sort"])

    top_path = out_csv_dir / "generated_top20_single_organ.csv"
    topN.to_csv(top_path, index=False)
    print(f"[OK] 保存 top-{N} 结果到 {top_path}")


if __name__ == "__main__":
    main()

