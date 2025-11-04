# src/dataset_builder/encoding.py
import numpy as np
from typing import Dict

def build_alphabet_map(alphabet):
    # Ensure order and include 'N' as last if not present
    alpha = list(alphabet)
    if "N" not in alpha:
        alpha.append("N")
    return {ch: i for i, ch in enumerate(alpha)}

def onehot_encode(seq: str, alphabet_map: Dict[str, int], num_channels: int) -> np.ndarray:
    # Output shape: (C, L) for PyTorch 1D CNN (N, C, L)
    L = len(seq)
    arr = np.zeros((num_channels, L), dtype=np.uint8)
    for j, ch in enumerate(seq):
        idx = alphabet_map.get(ch, alphabet_map.get("N", num_channels - 1))
        if 0 <= idx < num_channels:
            arr[idx, j] = 1
    return arr
