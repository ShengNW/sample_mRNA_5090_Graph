# src/dataset_builder/config.py
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import yaml
import copy

@dataclass
class TruncPolicy:
    strategy: str
    length: int

@dataclass
class Config:
    raw: Dict[str, Any]
    input_parquet: str
    output_dir: str
    col_organ: str
    col_utr5: str
    col_utr3: str
    col_split: Optional[str]
    enc_mode: str
    alphabet: List[str]
    trunc_utr5: TruncPolicy
    trunc_utr3: TruncPolicy
    shard_size: int
    seed: int
#    output_format: str = "npz"  # 新增：npz | pt（默认 npz，向后兼容）
#    split_ratios: List[float]
#    stratify_by: str
     # —— 无默认值的字段必须在前面；下面两个至少要在有默认值的前面 ——
    split_ratios: List[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])
    stratify_by: str = "organ_id"
    # —— 所有带默认值的字段放最后 —— 
    output_format: str = "npz"  # npz | pt（默认 npz，向后兼容）

def _get(d: Dict[str, Any], path: str, default=None):
    cur = d
    for k in path.split("."):
        if k not in cur:
            return default
        cur = cur[k]
    return cur

def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    data = _get(raw, "data", {})
    colmap = _get(raw, "colmap", {})
    encoding = _get(raw, "encoding", {})
    trunc = _get(raw, "truncation", {})
    output = _get(raw, "output", {})
    split = _get(raw, "split", {})

    cfg = Config(
        raw=copy.deepcopy(raw),
        input_parquet=data.get("input_parquet"),
        # output_dir=data.get("output_dir", "data/processed/seq_cnn_v1"),
        output_dir=output.get("dir", "data/processed/seq_cnn_v1"),
        col_organ=colmap.get("organ_id", "organ_id"),
        col_utr5=colmap.get("utr5_seq", "utr5_seq"),
        col_utr3=colmap.get("utr3_seq", "utr3_seq"),
        col_split=colmap.get("split", None),
        enc_mode=encoding.get("mode", "onehot"),
        alphabet=encoding.get("alphabet", ["A", "C", "G", "T", "N"]),
        trunc_utr5=TruncPolicy(
            strategy=_get(trunc, "utr5.strategy", "tail"),
            length=int(_get(trunc, "utr5.length", 1024))
        ),
        trunc_utr3=TruncPolicy(
            strategy=_get(trunc, "utr3.strategy", "ends_concat"),
            length=int(_get(trunc, "utr3.length", 2048))
        ),
        shard_size=int(output.get("shard_size", 8192)),
        seed=int(split.get("seed", 2024)),
        output_format=str(output.get("format", "npz")).lower(),  # 新增：读取 output.format
        split_ratios=list(split.get("ratios", [0.8, 0.1, 0.1])),
        stratify_by=split.get("stratify_by", colmap.get("organ_id", "organ_id")),
    )
    if abs(sum(cfg.split_ratios) - 1.0) > 1e-6:
        s = sum(cfg.split_ratios)
        cfg.split_ratios = [r / s for r in cfg.split_ratios]
    return cfg
