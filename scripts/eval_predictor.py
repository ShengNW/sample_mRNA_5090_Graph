"""
评估预测器：读入 predict_eval.csv，计算回归/二分类指标，并保存一个简要报告。
"""
import argparse, json, numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score, average_precision_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--mfe", default=None)
    ap.add_argument("--rbp", default=None)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.pred)
    y_true = df["y_true"].values
    y_pred = df["y_pred"].values

    report = {}
    # 回归指标
    report["r2"] = float(r2_score(y_true, y_pred))
    report["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # 若存在二分类标签（0/1），计算 AUC/PR
    if set(np.unique(y_true)).issubset({0,1}):
        report["auc_roc"] = float(roc_auc_score(y_true, y_pred))
        report["avg_precision"] = float(average_precision_score(y_true, y_pred))

    # 合并外部特征用于后续可视化
    if args.mfe and os.path.exists(args.mfe):
        df_mfe = pd.read_csv(args.mfe)
        df = df.merge(df_mfe, on="seq_id", how="left")
    if args.rbp and os.path.exists(args.rbp):
        df_rbp = pd.read_csv(args.rbp)
        df = df.merge(df_rbp, on="seq_id", how="left")

    df.to_csv(os.path.join(args.outdir, "predict_eval_merged.csv"), index=False)
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("Saved metrics:", report)

if __name__ == "__main__":
    import os
    main()
