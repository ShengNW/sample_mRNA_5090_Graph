"""
计算 UTR 的最小自由能（MFE）。
需要系统已安装 `RNAfold`（ViennaRNA）。将从输入 CSV 中读取 `utr5` 和/或 `utr3` 列，分别计算并合并。
输出列：seq_id, mfe_utr5, mfe_utr3
"""
import argparse, subprocess, tempfile, os, pandas as pd
from utils import read_csv_required, save_merged_csv

def run_rnafold(seq_list):
    # 使用临时文件批量调用 RNAfold
    with tempfile.TemporaryDirectory() as td:
        in_f = os.path.join(td, "in.fa")
        out_f = os.path.join(td, "out.txt")
        with open(in_f, "w") as f:
            for i, s in enumerate(seq_list):
                f.write(f">s{i}\n{s}\n")
        # -d2：对 dangling ends 的处理更接近常用设置；--noPS：不生成ps图片
        cmd = ["RNAfold", "--noPS", "-d2"]
        with open(in_f, "r") as fin, open(out_f, "w") as fout:
            subprocess.run(cmd, stdin=fin, stdout=fout, check=True)
        # 解析输出：每两行一条，第二行括号里包含能量
        mfes = []
        with open(out_f) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        for i in range(0, len(lines), 2):
            line2 = lines[i+1]
            # 末尾类似  "....((...))... (-12.30)"
            energy = float(line2[line2.rfind("(")+1: line2.rfind(")")])
            mfes.append(energy)
        return mfes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = read_csv_required(args.input, ["seq_id"])
    out = df[["seq_id"]].copy()

    if "utr5" in df.columns:
        out["mfe_utr5"] = run_rnafold(df["utr5"].astype(str).tolist())
    if "utr3" in df.columns:
        out["mfe_utr3"] = run_rnafold(df["utr3"].astype(str).tolist())

    save_merged_csv(out, args.output)
    print(f"Saved {args.output} with columns: {out.columns.tolist()}")

if __name__ == "__main__":
    main()
