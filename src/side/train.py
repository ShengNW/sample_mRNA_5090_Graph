"""Training entry-point for the FiLM-conditioned CNN."""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import yaml

from src.side.dataset import (
    ShardShuffleSampler,
    UTRFeatureShardDataset,
    load_manifest,
)
from src.side.model import DualBranchCNNFiLM


def setup_device() -> Tuple[torch.device, bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    return device, distributed, rank, local_rank, world_size


logger = logging.getLogger(__name__)


def create_dataloaders(
    dataset_dir: str,
    batch_size: int,
    num_workers: int,
    distributed: bool,
    rank: int,
    world_size: int,
    *,
    max_cache_shards: int,
    shard_seed: int,
    persistent_workers: bool,
    prefetch_factor: int,
    split_overrides: Dict[str, Dict] | None = None,
) -> Dict[str, DataLoader]:
    loaders: Dict[str, DataLoader] = {}
    split_overrides = split_overrides or {}
    for split in ("train", "val"):
        split_cfg = split_overrides.get(split, {})
        split_max_cache_shards = split_cfg.get("max_cache_shards", max_cache_shards)
        try:
            dataset = UTRFeatureShardDataset(
                dataset_dir,
                split=split,
                max_cache_shards=split_max_cache_shards,
            )
        except FileNotFoundError:
            if split == "val":
                continue
            raise
        sampler = None
        if distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=(split == "train"),
            )
        elif split == "train":
            sampler = ShardShuffleSampler(
                dataset.shard_groups,
                seed=shard_seed,
            )
        split_batch_size = split_cfg.get("batch_size", batch_size)
        split_num_workers = split_cfg.get("num_workers", num_workers)
        loader_kwargs = dict(
            dataset=dataset,
            batch_size=split_batch_size,
            shuffle=(sampler is None and split == "train"),
            num_workers=split_num_workers,
            pin_memory=True,
            sampler=sampler,
        )
        if split_num_workers > 0:
            loader_kwargs["persistent_workers"] = split_cfg.get(
                "persistent_workers", persistent_workers
            )
            loader_kwargs["prefetch_factor"] = split_cfg.get("prefetch_factor", prefetch_factor)
        loaders[split] = DataLoader(**loader_kwargs)
        if rank == 0:
            num_batches = len(loaders[split])
            logger.info(
                "DataLoader[%s]: %d samples -> %d batches (batch_size=%d, workers=%d)",
                split,
                len(dataset),
                num_batches,
                split_batch_size,
                split_num_workers,
            )
    return loaders


def compute_r2(preds: np.ndarray, targets: np.ndarray) -> float:
    if preds.size == 0:
        return float("nan")
    ss_res = np.sum((preds - targets) ** 2)
    mean_y = np.mean(targets)
    ss_tot = np.sum((targets - mean_y) ** 2)
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def run_training(cfg: Dict) -> None:
    device, distributed, rank, local_rank, world_size = setup_device()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    dataset_dir = cfg["dataset_dir"]
    manifest = load_manifest(dataset_dir)
    organ_vocab = manifest.get("organ_vocab", {})
    num_organs = max(len(organ_vocab), int(cfg.get("num_organs", 0)))
    if num_organs == 0:
        raise ValueError("Number of organs could not be determined from manifest or config")
    input_channels = manifest["shapes"]["utr5"][0]
    if rank == 0:
        logger.info(
            "Starting training run | distributed=%s world_size=%d device=%s", distributed, world_size, device
        )
        logger.info(
            "Detected %d organs (from manifest=%d, config=%d)",
            num_organs,
            len(organ_vocab),
            int(cfg.get("num_organs", 0)),
        )
    loaders = create_dataloaders(
        dataset_dir,
        batch_size=cfg.get("batch_size", 64),
        num_workers=cfg.get("num_workers", 4),
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        max_cache_shards=cfg.get("max_cache_shards", 2),
        shard_seed=cfg.get("dataloader_seed", 0),
        persistent_workers=cfg.get("persistent_workers", True),
        prefetch_factor=cfg.get("prefetch_factor", 2),
        split_overrides=cfg.get("split_loader_overrides"),
    )
    if "train" not in loaders:
        raise RuntimeError("Training split not found in dataset")

    model = DualBranchCNNFiLM(
        in_channels=input_channels,
        num_organs=num_organs,
        conv_channels=cfg.get("conv_channels", [64, 128, 256]),
        stem_channels=cfg.get("stem_channels", 32),
        film_dim=cfg.get("film_dim", 32),
        hidden_dim=cfg.get("hidden_dim", 256),
        dropout=cfg.get("dropout", 0.2),
    ).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("learning_rate", 1e-3), weight_decay=cfg.get("weight_decay", 1e-4))

    best_r2 = -float("inf")
    best_state = None
    epochs = cfg.get("epochs", 20)
    log_interval = max(int(cfg.get("log_interval", 200)), 1)
    train_loader = loaders["train"]
    train_sampler = getattr(train_loader, "sampler", None)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        samples = 0
        epoch_start = time.perf_counter()
        data_time_total = 0.0
        iter_time_total = 0.0
        if distributed and isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)
        elif hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        step_start = time.perf_counter()
        for batch_idx, batch in enumerate(train_loader):
            data_time = time.perf_counter() - step_start
            utr5 = batch["utr5"].to(device, non_blocking=True)
            utr3 = batch["utr3"].to(device, non_blocking=True)
            organ = batch["organ_id"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            preds = model(utr5, utr3, organ)
            loss = criterion(preds, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * target.size(0)
            samples += target.size(0)
            iter_time = time.perf_counter() - step_start
            data_time_total += data_time
            iter_time_total += iter_time
            if rank == 0 and (batch_idx == 0 or (batch_idx + 1) % log_interval == 0):
                batches_per_epoch = len(train_loader)
                avg_iter = iter_time_total / (batch_idx + 1)
                avg_data = data_time_total / (batch_idx + 1)
                throughput = samples / max(iter_time_total, 1e-6)
                logger.info(
                    "Epoch %d/%d | step %d/%d | loss=%.4f | avg_iter=%.3fs | avg_data=%.3fs | samples/s=%.1f",
                    epoch,
                    epochs,
                    batch_idx + 1,
                    batches_per_epoch,
                    loss.item(),
                    avg_iter,
                    avg_data,
                    throughput,
                )
            step_start = time.perf_counter()
        train_loss /= max(samples, 1)
        epoch_time = time.perf_counter() - epoch_start
        if rank == 0:
            logger.info(
                "Epoch %d/%d completed in %.2fs | train_loss=%.4f | samples=%d",
                epoch,
                epochs,
                epoch_time,
                train_loss,
                samples,
            )

        if "val" in loaders:
            model.eval()
            preds_all: List[float] = []
            labels_all: List[float] = []
            val_start = time.perf_counter()
            with torch.no_grad():
                for batch in loaders["val"]:
                    utr5 = batch["utr5"].to(device, non_blocking=True)
                    utr3 = batch["utr3"].to(device, non_blocking=True)
                    organ = batch["organ_id"].to(device, non_blocking=True)
                    target = batch["label"].to(device, non_blocking=True)
                    outputs = model(utr5, utr3, organ)
                    preds_all.append(outputs.detach().cpu().numpy())
                    labels_all.append(target.detach().cpu().numpy())
            if preds_all:
                preds_arr = np.concatenate(preds_all)
                labels_arr = np.concatenate(labels_all)
                r2 = compute_r2(preds_arr, labels_arr)
                if rank == 0:
                    val_time = time.perf_counter() - val_start
                    logger.info("Validation completed in %.2fs | R2=%.4f", val_time, r2)
                if r2 > best_r2 and rank == 0:
                    best_r2 = r2
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if rank == 0 and best_state is not None:
        out_path = Path(cfg.get("output_model_path", "outputs/best_model.pt"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, out_path)
        logger.info("Saved best model to %s (best R2=%.4f)", out_path, best_r2)
    if distributed:
        dist.destroy_process_group()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Train the FiLM CNN with UTR features")
    parser.add_argument("--config", required=True, help="Path to YAML configuration")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    log_level = cfg.get("log_level")
    if log_level:
        logging.getLogger().setLevel(getattr(logging, str(log_level).upper(), logging.INFO))
    logger.info("Loaded training configuration from %s", args.config)
    run_training(cfg)


if __name__ == "__main__":
    main()
