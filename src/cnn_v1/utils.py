# --- file: src/cnn_v1/utils.py ---
from __future__ import annotations
import os, json, math, time, yaml, random
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np
import torch

SEED_DEFAULT = 20251003


def set_seed(seed: int = SEED_DEFAULT):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
