# src/dataset_builder/label_mapping.py
from typing import Dict, Tuple
import json, os
import pandas as pd

def build_label_mapping(df: pd.DataFrame, col_organ: str) -> Dict[str, int]:
    classes = sorted(list(map(str, pd.unique(df[col_organ]))))
    mapping = {c: i for i, c in enumerate(classes)}
    return mapping

def dump_label_mapping(path: str, mapping: Dict[str, int]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
