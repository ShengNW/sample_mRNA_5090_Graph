# src/dataset_builder/preprocess.py
from typing import Tuple

def clean_seq(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().upper()
    s = s.replace("U", "T")
    # keep only A/C/G/T/N
    allowed = set("ACGTN")
    return "".join(ch if ch in allowed else "N" for ch in s)

def pad_or_trim(seq: str, target_len: int, side: str = "right", pad_char: str = "N") -> str:
    # side = "right": pad at right; "left": pad at left
    if len(seq) == target_len:
        return seq
    if len(seq) > target_len:
        return seq[:target_len]
    pad = pad_char * (target_len - len(seq))
    return seq + pad if side == "right" else pad + seq
