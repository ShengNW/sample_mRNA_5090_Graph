# src/dataset_builder/io_reader.py
import os, hashlib, json
from typing import Dict, Any, Tuple
import pandas as pd

def _file_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def read_parquet_with_snapshot(path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"input parquet not found: {path}")
    st = os.stat(path)
    df = pd.read_parquet(path)  # requires pyarrow or fastparquet
    snap = {
        "path": os.path.abspath(path),
        "size_bytes": int(st.st_size),
        "mtime": int(st.st_mtime),
        "sha256": _file_sha256(path),
        "n_rows": int(len(df)),
        "columns": list(df.columns),
    }
    return df, snap

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def dump_json(path: str, obj: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
