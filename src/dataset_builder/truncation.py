# src/dataset_builder/truncation.py
from typing import Tuple
from .preprocess import pad_or_trim

def truncate_utr5(seq: str, length: int, strategy: str = "tail") -> Tuple[str, str]:
    # returns (seq_after, policy_str)
    if strategy == "head":
        cut = seq[:length]
        pol = f"head@{length}"
    elif strategy == "center":
        if len(seq) <= length:
            cut = seq
        else:
            start = max(0, (len(seq) - length) // 2)
            cut = seq[start:start+length]
        pol = f"center@{length}"
    else:  # tail (default)
        cut = seq[-length:] if len(seq) > length else seq
        pol = f"tail@{length}"
    cut = pad_or_trim(cut, length, side="left")  # pad on the left to keep tail aligned to right
    return cut, pol

def truncate_utr3(seq: str, length: int, strategy: str = "ends_concat") -> Tuple[str, str]:
    if strategy == "ends_concat":
        half = length // 2
        left = seq[:half]
        right = seq[-half:] if len(seq) >= half else seq
        cut = left + right
        pol = f"ends_concat@{length}"
    elif strategy == "head":
        cut = seq[:length]
        pol = f"head@{length}"
    elif strategy == "tail":
        cut = seq[-length:] if len(seq) > length else seq
        pol = f"tail@{length}"
    else:
        # fallback to center
        if len(seq) <= length:
            cut = seq
        else:
            start = max(0, (len(seq) - length) // 2)
            cut = seq[start:start+length]
        pol = f"center@{length}"
    cut = pad_or_trim(cut, length, side="right")
    return cut, pol
