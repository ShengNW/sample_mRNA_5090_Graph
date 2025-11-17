"""
使用 FIMO（MEME Suite）扫描 RBP PWM 位点。
需要本机已安装 `fimo`，并提供 `--motifs`（MEME 格式）文件。
输出列：seq_id, rbp_hits_total, rbp_hits_per_kb
"""
import argparse, os, pandas as pd, subprocess, tempfile
from utils import read_csv_required, save_merged_csv

def write_fasta(df, fasta_path):
    with open(fasta_path, "w") as f:
        for _, row in df.iterrows():
            seq = (str(row.get("utr5","")) + str(row.get("utr3",""))).replace("U","T")
            f.write(f">{row['seq_id']}\n{seq}\n")

def run_fimo(meme_motifs, fasta_path, out_dir):
    cmd = ["fimo", "--verbosity", "1", "--text", meme_motifs, fasta_path]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # 解析 TSV 风格输出（文本模式）
    rows = []
    for ln in res.stdout.splitlines():
        if ln.startswith("#") or not ln.strip(): 
            continue
        parts = ln.split('\t')
        if len(parts) >= 9:
            motif_id = parts[0]
            sequence_id = parts[1]
            try:
                pval = float(parts[7])
            except ValueError:
                continue
            qval = None
            if parts[8] not in ("", "NA"):
                try:
                    qval = float(parts[8])
                except ValueError:
                    qval = None
            rows.append((sequence_id, motif_id, pval, qval))
    df = pd.DataFrame(rows, columns=["seq_id","motif","pval","qval"])
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--motifs", required=True, help="MEME/PWM 文件路径")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = read_csv_required(args.input, ["seq_id"])
    with tempfile.TemporaryDirectory() as td:
        fa = os.path.join(td, "seqs.fa")
        write_fasta(df, fa)
        hits = run_fimo(args.motifs, fa, td)

    # 聚合
    agg = hits.groupby("seq_id").size().rename("rbp_hits_total").reset_index()
    merged = df[["seq_id"]].merge(agg, on="seq_id", how="left").fillna({"rbp_hits_total":0})
    merged["rbp_hits_per_kb"] = merged["rbp_hits_total"] / ((df.get("utr5","").str.len().fillna(0) + df.get("utr3","").str.len().fillna(0)).replace(0,1)/1000)

    save_merged_csv(merged, args.output)
    print(f"Saved {args.output}")

if __name__ == "__main__":
    main()
