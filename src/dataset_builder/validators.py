# src/dataset_builder/validators.py
from typing import Dict, Any
import pandas as pd
import numpy as np
import re

_BASERE = re.compile(r"^[ACGTUNacgtun]+$")

def basic_scan(df: pd.DataFrame, col_utr5: str, col_utr3: str, col_organ: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    n = len(df)
    out["n_rows"] = int(n)

    def _len_stats(s: pd.Series):
        lens = s.fillna("").astype(str).str.len().to_numpy()
        return dict(
            min=int(lens.min() if len(lens) else 0),
            p50=float(np.percentile(lens, 50)) if len(lens) else 0,
            p90=float(np.percentile(lens, 90)) if len(lens) else 0,
            max=int(lens.max() if len(lens) else 0),
            mean=float(lens.mean() if len(lens) else 0.0),
        )

    def _invalid_ratio(s: pd.Series):
        vals = s.fillna("").astype(str)
        if len(vals) == 0: return 0.0
        ok = vals.map(lambda x: bool(_BASERE.match(x)) if x != "" else True)
        return float(1.0 - ok.mean())

    def _n_ratio(s: pd.Series):
        vals = s.fillna("").astype(str)
        tot = vals.str.len().replace(0, np.nan)
        ncnt = vals.str.upper().str.count("N")
        ratio = (ncnt / tot).fillna(0.0)
        return dict(mean=float(ratio.mean()), p90=float(np.percentile(ratio, 90)) if len(ratio) else 0.0)

    out["utr5_len"] = _len_stats(df[col_utr5])
    out["utr3_len"] = _len_stats(df[col_utr3])
    out["utr5_invalid_ratio"] = _invalid_ratio(df[col_utr5])
    out["utr3_invalid_ratio"] = _invalid_ratio(df[col_utr3])
    out["utr5_N_ratio"] = _n_ratio(df[col_utr5])
    out["utr3_N_ratio"] = _n_ratio(df[col_utr3])

    cls_counts = df[col_organ].value_counts().sort_index()
    out["class_counts"] = {str(k): int(v) for k, v in cls_counts.items()}
    out["n_classes"] = int(cls_counts.shape[0])
    return out
