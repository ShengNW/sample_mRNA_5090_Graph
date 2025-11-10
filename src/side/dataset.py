"""Dataset utilities for training the side-feature CNN model."""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler, get_worker_info


logger = logging.getLogger(__name__)


class _ShardCache:
    """LRU cache that keeps a limited number of shards resident in memory."""

    def __init__(self, max_size: int) -> None:
        if max_size < 1:
            raise ValueError("max_size for shard cache must be at least 1")
        self.max_size = max_size
        self._store: "OrderedDict[int, Dict[str, torch.Tensor]]" = OrderedDict()

    def get(self, key: int, loader: Callable[[], Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if key in self._store:
            value = self._store.pop(key)
            self._store[key] = value
            return value
        value = loader()
        self._store[key] = value
        if len(self._store) > self.max_size:
            evicted_key, _ = self._store.popitem(last=False)
            logger.debug("Evicting shard %s from in-memory cache", evicted_key)
        return value


class ShardShuffleSampler(Sampler[int]):
    """Sampler that randomises traversal order while keeping shard locality."""

    def __init__(
        self,
        shard_groups: Iterable[np.ndarray],
        *,
        shuffle_within_shard: bool = True,
        shuffle_shards: bool = True,
        seed: int = 0,
    ) -> None:
        self._groups: List[np.ndarray] = [np.asarray(g, dtype=np.int64) for g in shard_groups if len(g) > 0]
        if not self._groups:
            raise ValueError("ShardShuffleSampler requires at least one non-empty shard group")
        self.shuffle_within_shard = shuffle_within_shard
        self.shuffle_shards = shuffle_shards
        self._base_seed = seed
        self._epoch = 0

    def __len__(self) -> int:
        return int(sum(len(group) for group in self._groups))

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self):  # type: ignore[override]
        rng = np.random.default_rng(self._base_seed + self._epoch)
        shard_order = np.arange(len(self._groups))
        if self.shuffle_shards:
            rng.shuffle(shard_order)
        for shard_idx in shard_order:
            group = self._groups[shard_idx]
            if self.shuffle_within_shard:
                group = group.copy()
                rng.shuffle(group)
            for sample_idx in group:
                yield int(sample_idx)


def load_manifest(dataset_dir: str) -> Dict[str, Any]:
    """Load the JSON manifest produced by the preprocessing script."""

    manifest_path = Path(dataset_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


class UTRFeatureShardDataset(Dataset):
    """Dataset that streams UTR tensors from PyTorch shards with cached loading."""

    def __init__(
        self,
        dataset_dir: str,
        split: str = "train",
        *,
        max_cache_shards: int = 2,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.shards_dir = self.dataset_dir / "shards"
        if not self.shards_dir.exists():
            raise FileNotFoundError(f"Shards directory not found: {self.shards_dir}")
        self.shard_paths: List[Path] = sorted(self.shards_dir.glob("data.part-*.pt"))
        if not self.shard_paths:
            raise FileNotFoundError(f"No .pt shards found under {self.shards_dir}")
        self.index = self._load_index(split)
        if {"part_id", "local_idx"} - set(self.index.columns):
            raise ValueError(f"Index for split '{split}' must contain part_id and local_idx")
        self.index = self.index.reset_index(drop=True)
        self._cache = _ShardCache(max_cache_shards)
        grouped = self.index.groupby("part_id", sort=True).indices
        self._shard_groups: List[np.ndarray] = [
            np.asarray(list(indices), dtype=np.int64) for _, indices in sorted(grouped.items())
        ]
        if not self._shard_groups:
            raise ValueError(f"Split '{split}' did not contain any samples")
        self._max_cache_shards = max_cache_shards
        logger.info(
            "Initialised %s split with %d samples across %d shards (max_cache_shards=%d)",
            split,
            len(self.index),
            len(self.shard_paths),
            max_cache_shards,
        )

    def _load_index(self, split: str) -> pd.DataFrame:
        candidates = [
            self.dataset_dir / "index" / split / "index.parquet",
            self.dataset_dir / f"{split}.index.parquet",
            self.dataset_dir / "index" / split / "index.csv",
            self.dataset_dir / f"{split}.index.csv",
        ]
        for path in candidates:
            if path.exists():
                if path.suffix == ".parquet":
                    return pd.read_parquet(path)
                return pd.read_csv(path)
        raise FileNotFoundError(f"Index file for split '{split}' not found under {self.dataset_dir}")

    def __len__(self) -> int:
        return int(len(self.index))

    def _load_shard(self, part_id: int) -> Dict[str, torch.Tensor]:
        def _load() -> Dict[str, torch.Tensor]:
            shard_path = self.shard_paths[part_id]
            worker = get_worker_info()
            worker_id = worker.id if worker is not None else "main"
            logger.info(
                "Worker %s loading shard %s (%d/%d) with cache size %d",
                worker_id,
                shard_path.name,
                part_id + 1,
                len(self.shard_paths),
                self._max_cache_shards,
            )
            data = torch.load(shard_path, map_location="cpu")
            if not isinstance(data, dict):
                raise ValueError(f"Shard {shard_path} must be a dict with tensors")
            return {
                "utr5": data["utr5"].float(),
                "utr3": data["utr3"].float(),
                "organ_id": data["organ_id"].long(),
                "label": data["label"].float(),
            }

        return self._cache.get(part_id, _load)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.index.iloc[idx]
        part_id = int(row["part_id"])
        local_idx = int(row["local_idx"])
        shard = self._load_shard(part_id)
        utr5 = shard["utr5"][local_idx]
        utr3 = shard["utr3"][local_idx]
        organ = shard["organ_id"][local_idx]
        label = shard["label"][local_idx]
        return {"utr5": utr5, "utr3": utr3, "organ_id": organ, "label": label}

    @property
    def shard_groups(self) -> List[np.ndarray]:
        return self._shard_groups
