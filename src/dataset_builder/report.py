# src/dataset_builder/report.py
import os
from datetime import datetime
from typing import Dict, Any

def write_report_md(out_dir: str, snapshot: Dict[str, Any], scan: Dict[str, Any], shapes, counts, label_n_expected: int = 54):
    rep_dir = os.path.join(out_dir, "reports")
    os.makedirs(rep_dir, exist_ok=True)
    path = os.path.join(rep_dir, "data_report.md")
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    def _kv(d: Dict[str, Any], indent=""):
        return "\n".join([f"{indent}- **{k}**: {v}" for k, v in d.items()])

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Dataset Build Report\n\n")
        f.write(f"- Timestamp: {ts}\n")
        f.write(f"- Input snapshot:\n{_kv(snapshot, indent='  ')}\n\n")
        f.write("## Length Stats\n")
        f.write(f"- 5' UTR: {scan['utr5_len']}\n")
        f.write(f"- 3' UTR: {scan['utr3_len']}\n\n")
        f.write("## Quality\n")
        f.write(f"- 5' invalid ratio: {scan['utr5_invalid_ratio']:.4f}\n")
        f.write(f"- 3' invalid ratio: {scan['utr3_invalid_ratio']:.4f}\n")
        f.write(f"- 5' N ratio (mean/p90): {scan['utr5_N_ratio']}\n")
        f.write(f"- 3' N ratio (mean/p90): {scan['utr3_N_ratio']}\n\n")
        f.write("## Classes\n")
        f.write(f"- n_classes: {scan['n_classes']} (expected {label_n_expected})\n")
        f.write(f"- class_counts: {scan['class_counts']}\n\n")
        f.write("## Encoded Shapes\n")
        f.write(f"- utr5 CxL: {shapes['utr5']}\n")
        f.write(f"- utr3 CxL: {shapes['utr3']}\n\n")
        f.write("## Counts\n")
        f.write(f"- total: {counts['total']}, train: {counts['train']}, val: {counts['val']}, test: {counts['test']}\n")
    return path
