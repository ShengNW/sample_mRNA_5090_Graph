"""Build a processed dataset that augments UTR sequences with RBP and tRNA features.

This script follows the preprocessing steps described in the regression-improve plan:

1. Parse Ensembl GTF annotations to recover per-gene 5'/3' UTR intervals.
2. Intersect UTRs with ENCODE eCLIP peak tracks to derive binary RBP masks.
3. Compute the distance from each gene to the nearest tRNA locus.
4. Combine the above with one-hot encoded UTR sequences to produce tensors with
   seven channels (5 for nucleotides + RBP + tRNA) for both 5' and 3' UTRs.

The resulting dataset is stored as PyTorch shards together with split indices and
manifest metadata so that the training pipeline can load it efficiently.

Usage
-----
python -m src.gen.build_rbp_trna_dataset \
    --dataset-config configs/dataset.yaml \
    --gtf data/external/ref/ensembl/Homo_sapiens.GRCh38.115.gtf.gz \
    --rbp-dir data/external/rbp/encode_eclip \
    --trna-bed data/external/trna/ensembl/tRNA.GRCh38.bed \
    --output-dir data/processed/seq_cnn_v1_rbp_trna

The command line flags allow overriding column names if the raw parquet file uses
non-default naming.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


import numpy as np
import pandas as pd
import torch
import yaml

from src.side.features import load_utr_coords


LOGGER = logging.getLogger(__name__)


def configure_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure root logging handlers for console and optional file output."""

    level = getattr(logging, (log_level or "INFO").upper(), logging.INFO)
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file:
        log_path = Path(log_file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    logging.captureWarnings(True)


def _resolve_path_case_insensitive(path: Path) -> Path:
    """Resolve a potentially mis-cased path by searching case-insensitively."""

    expanded = path.expanduser()
    if expanded.exists():
        return expanded

    # Determine the starting directory (absolute anchor or current working dir).
    if expanded.is_absolute():
        current = Path(expanded.anchor)
        parts = expanded.parts[1:]
    else:
        current = Path.cwd()
        parts = expanded.parts

    for part in parts:
        if part in ("", "."):
            continue
        if part == "..":
            current = current.parent
            continue

        try:
            entries = {p.name.lower(): p for p in current.iterdir()}
        except FileNotFoundError as exc:  # Directory doesn't exist at any case.
            raise FileNotFoundError(
                f"Directory '{current}' not found while resolving '{path}'."
            ) from exc

        match = entries.get(part.lower())
        if match is None:
            raise FileNotFoundError(
                f"Path component '{part}' not found under '{current}' (case-insensitive search)."
            )
        current = match

    if current.exists():
        return current

    raise FileNotFoundError(f"Path '{path}' could not be resolved case-insensitively.")

# ---------------------------------------------------------------------------
# Helper dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TruncationConfig:
    """Lengths used for truncating/padding UTR sequences."""

    utr5_len: int = 1024
    utr3_head: int = 1024
    utr3_tail: int = 1024

    @property
    def utr3_len(self) -> int:
        return self.utr3_head + self.utr3_tail


@dataclass
class UTRTruncationMap:
    """Holds mapping information from genomic positions to truncated indices."""

    chrom: str
    strand: str
    utr5_positions: List[int]
    utr5_index: Dict[int, int]
    utr3_positions: List[int]
    utr3_index: Dict[int, int]
    utr3_head_used: int
    min_start: Optional[int]
    max_end: Optional[int]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _flatten_segments(segments: Sequence[Tuple[int, int]], strand: str) -> List[int]:
    """Return a list of genomic positions following transcript orientation."""

    if not segments:
        return []
    ordered = sorted(segments, key=lambda x: x[0])
    positions: List[int] = []
    if strand == "-":
        ordered = list(reversed(ordered))
        for start, end in ordered:
            for pos in range(end, start - 1, -1):
                positions.append(pos)
    else:
        for start, end in ordered:
            for pos in range(start, end + 1):
                positions.append(pos)
    return positions


def _select_tail(positions: Sequence[int], max_len: int) -> Tuple[List[int], Dict[int, int]]:
    if not positions:
        return [], {}
    if len(positions) <= max_len:
        trimmed = list(positions)
    else:
        trimmed = list(positions[-max_len:])
    mapping = {pos: idx for idx, pos in enumerate(trimmed)}
    return trimmed, mapping


def _select_ends_concat(
    positions: Sequence[int], head_len: int, tail_len: int
) -> Tuple[List[int], Dict[int, int], int]:
    if not positions:
        return [], {}, 0
    total_keep = head_len + tail_len
    if len(positions) <= total_keep:
        trimmed = list(positions)
        mapping = {pos: idx for idx, pos in enumerate(trimmed)}
        head_used = min(len(trimmed), head_len)
        return trimmed, mapping, head_used
    head = list(positions[:head_len])
    tail = list(positions[-tail_len:])
    trimmed = head + tail
    mapping = {pos: idx for idx, pos in enumerate(trimmed)}
    return trimmed, mapping, len(head)


def build_truncation_map(
    utr_coords: Mapping[str, Mapping[str, object]], trunc_cfg: TruncationConfig
) -> Dict[str, UTRTruncationMap]:
    LOGGER.info(
        "Building truncation map for %d genes using config: utr5=%d, utr3_head=%d, utr3_tail=%d",
        len(utr_coords),
        trunc_cfg.utr5_len,
        trunc_cfg.utr3_head,
        trunc_cfg.utr3_tail,
    )
    maps: Dict[str, UTRTruncationMap] = {}
    for gene_id, info in utr_coords.items():
        chrom = info.get("chr")
        strand = info.get("strand", "+")
        if chrom is None:
            continue
        utr5_segments = info.get("5utr") or []
        utr3_segments = info.get("3utr") or []
        pos5 = _flatten_segments(utr5_segments, strand)
        pos3 = _flatten_segments(utr3_segments, strand)
        trimmed5, map5 = _select_tail(pos5, trunc_cfg.utr5_len)
        trimmed3, map3, head_used = _select_ends_concat(
            pos3, trunc_cfg.utr3_head, trunc_cfg.utr3_tail
        )
        maps[gene_id] = UTRTruncationMap(
            chrom=chrom,
            strand=strand,
            utr5_positions=trimmed5,
            utr5_index=map5,
            utr3_positions=trimmed3,
            utr3_index=map3,
            utr3_head_used=head_used,
            min_start=info.get("min_start"),
            max_end=info.get("max_end"),
        )
    utr5_count = sum(1 for info in maps.values() if info.utr5_positions)
    utr3_count = sum(1 for info in maps.values() if info.utr3_positions)
    LOGGER.info(
        "Constructed truncation map for %d genes (%d with 5' UTR, %d with 3' UTR)",
        len(maps),
        utr5_count,
        utr3_count,
    )
    return maps


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------


def _iter_rbp_peak_files(root_dir: Path) -> Iterable[Path]:
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if not name.endswith((".bed", ".bed.gz", ".narrowPeak", ".narrowPeak.gz")):
                continue
            yield Path(dirpath) / name


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path, "r")


def build_rbp_masks(
    peaks_dir: Path, trunc_map: Mapping[str, UTRTruncationMap]
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Return binary masks (per base) for RBP overlaps on 5' and 3' UTRs."""

    LOGGER.info("Building RBP masks from peak directory %s", peaks_dir)
    mask5: Dict[str, np.ndarray] = {
        gene: np.zeros(len(info.utr5_positions), dtype=np.float32)
        for gene, info in trunc_map.items()
    }
    mask3: Dict[str, np.ndarray] = {
        gene: np.zeros(len(info.utr3_positions), dtype=np.float32)
        for gene, info in trunc_map.items()
    }

    genes_by_chr: Dict[str, List[Tuple[int, int, str]]] = defaultdict(list)
    for gene, info in trunc_map.items():
        if info.chrom is None or info.min_start is None or info.max_end is None:
            continue
        genes_by_chr[info.chrom].append((int(info.min_start), int(info.max_end), gene))
    for chrom in genes_by_chr:
        genes_by_chr[chrom].sort(key=lambda x: x[0])

    num_files = 0
    num_peaks = 0
    for path in _iter_rbp_peak_files(peaks_dir):
        num_files += 1
        LOGGER.debug("Processing RBP peak file %s", path)
        with _open_text(path) as handle:
            for line in handle:
                if not line or line.startswith("#"):
                    continue
                cols = line.strip().split()
                if len(cols) < 3:
                    continue
                num_peaks += 1
                chrom = cols[0]
                try:
                    start = int(cols[1])
                    end = int(cols[2])
                except ValueError:
                    continue
                if chrom not in genes_by_chr:
                    continue
                for gene_start, gene_end, gene_id in genes_by_chr[chrom]:
                    if gene_start is None or gene_end is None:
                        continue
                    if gene_start > end:
                        break
                    if gene_end < start:
                        continue
                    info = trunc_map.get(gene_id)
                    if info is None:
                        continue
                    ov_start = max(start, gene_start)
                    ov_end = min(end, gene_end)
                    for pos in range(ov_start, ov_end + 1):
                        idx5 = info.utr5_index.get(pos)
                        if idx5 is not None and idx5 < mask5[gene_id].size:
                            mask5[gene_id][idx5] = 1.0
                        idx3 = info.utr3_index.get(pos)
                        if idx3 is not None and idx3 < mask3[gene_id].size:
                            mask3[gene_id][idx3] = 1.0
    LOGGER.info(
        "Finished building RBP masks from %d files with %d peaks", num_files, num_peaks
    )
    return mask5, mask3


def build_trna_features(
    trna_bed: Path, trunc_map: Mapping[str, UTRTruncationMap]
) -> Dict[str, float]:
    LOGGER.info("Computing tRNA distances from %s", trna_bed)
    trna_bed = _resolve_path_case_insensitive(trna_bed)
    positions: Dict[str, List[int]] = defaultdict(list)
    with open(trna_bed, "r") as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            cols = line.strip().split()
            if len(cols) < 3:
                continue
            chrom = cols[0]
            try:
                start = int(cols[1])
                end = int(cols[2])
            except ValueError:
                continue
            positions[chrom].append((start + end) // 2)
    for chrom in positions:
        positions[chrom].sort()

    def nearest_distance(chrom: str, pos: int) -> Optional[int]:
        arr = positions.get(chrom)
        if not arr:
            return None
        # Binary search
        lo, hi = 0, len(arr)
        while lo < hi:
            mid = (lo + hi) // 2
            if arr[mid] < pos:
                lo = mid + 1
            else:
                hi = mid
        dist_candidates: List[int] = []
        if lo < len(arr):
            dist_candidates.append(abs(arr[lo] - pos))
        if lo > 0:
            dist_candidates.append(abs(arr[lo - 1] - pos))
        return min(dist_candidates) if dist_candidates else None

    trna_dist: Dict[str, float] = {}
    num_assigned = 0
    for gene, info in trunc_map.items():
        if not info.utr5_positions and not info.utr3_positions:
            continue
        chrom = info.chrom
        reference_pos: Optional[int]
        if info.strand == "-" and info.utr5_positions:
            reference_pos = info.utr5_positions[0]
        elif info.utr5_positions:
            reference_pos = info.utr5_positions[0]
        elif info.utr3_positions:
            reference_pos = info.utr3_positions[0]
        else:
            reference_pos = None
        if reference_pos is None:
            continue
        dist = nearest_distance(chrom, reference_pos)
        trna_dist[gene] = float(dist) if dist is not None else float("inf")
        num_assigned += 1
    LOGGER.info("Computed tRNA distances for %d genes", num_assigned)
    return trna_dist


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def _one_hot_encode(
    seq: str,
    length: int,
    strategy: str,
    alphabet: str = "ACGTN",
    head_len: Optional[int] = None,
    tail_len: Optional[int] = None,
) -> np.ndarray:
    seq = (seq or "").upper().replace("U", "T")
    mapping = {ch: idx for idx, ch in enumerate(alphabet)}
    n_ch = len(alphabet)
    arr = np.zeros((n_ch, length), dtype=np.float32)
    idx_n = mapping["N"]

    def mark(char: str, position: int) -> None:
        arr[mapping.get(char, idx_n), position] = 1.0

    if strategy == "tail":
        seq = seq[-length:]
        offset = max(0, length - len(seq))
        if offset > 0:
            arr[idx_n, :offset] = 1.0
        for i, ch in enumerate(seq):
            mark(ch, offset + i)
        if offset + len(seq) < length:
            arr[idx_n, offset + len(seq) :] = 1.0
        return arr

    if strategy == "ends":
        assert head_len is not None and tail_len is not None
        total_keep = head_len + tail_len
        if len(seq) <= total_keep:
            for i, ch in enumerate(seq[:length]):
                mark(ch, i)
            if len(seq) < length:
                arr[idx_n, len(seq) :] = 1.0
            return arr
        head_seq = seq[:head_len]
        tail_seq = seq[-tail_len:]
        for i, ch in enumerate(head_seq):
            if i >= length:
                break
            mark(ch, i)
        gap_start = min(head_len, length)
        gap_end = max(length - tail_len, gap_start)
        if gap_end > gap_start:
            arr[idx_n, gap_start:gap_end] = 1.0
        tail_start = max(length - tail_len, 0)
        for i, ch in enumerate(tail_seq[-(length - tail_start) :]):
            pos = tail_start + i
            if pos < length:
                mark(ch, pos)
        filled = arr.sum(axis=0) > 0
        arr[idx_n, ~filled] = 1.0
        return arr

    # Fallback: simple left alignment with padding on the right
    seq = seq[:length]
    for i, ch in enumerate(seq):
        mark(ch, i)
    if len(seq) < length:
        arr[idx_n, len(seq) :] = 1.0
    return arr


def _prepare_rbp_channel(
    mask_trimmed: Optional[np.ndarray],
    target_len: int,
    align: str,
    head_len: Optional[int] = None,
) -> np.ndarray:
    channel = np.zeros(target_len, dtype=np.float32)
    if mask_trimmed is None or mask_trimmed.size == 0:
        return channel
    if align == "tail":
        start = max(0, target_len - mask_trimmed.size)
        channel[start : start + mask_trimmed.size] = mask_trimmed[-target_len:]
    elif align == "ends":
        used = min(mask_trimmed.size, target_len)
        if head_len is None:
            head_len = used // 2
        head_used = min(head_len, used)
        tail_used = used - head_used
        channel[:head_used] = mask_trimmed[:head_used]
        if tail_used > 0:
            channel[target_len - tail_used :] = mask_trimmed[head_used : head_used + tail_used]
    else:
        channel[: mask_trimmed.size] = mask_trimmed[:target_len]
    return channel


def _prepare_trna_channel(norm_value: float, target_len: int) -> np.ndarray:
    return np.full(target_len, norm_value, dtype=np.float32)


def _normalise_trna_distance(dist: float, cap: float) -> float:
    if math.isinf(dist):
        return 1.0
    return min(dist, cap) / cap


def _write_split_indices(meta: pd.DataFrame, output_dir: Path) -> None:
    index_root = output_dir / "index"
    index_root.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Writing split indices under %s", index_root)
    for split, df_split in meta.groupby("split"):
        split_dir = index_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        table = df_split[["part_id", "local_idx", "global_idx", "organ_id"]]
        parquet_path = split_dir / "index.parquet"
        try:
            table.to_parquet(parquet_path, index=False)
            LOGGER.debug("Wrote parquet index for split '%s' to %s", split, parquet_path)
        except Exception:
            csv_path = split_dir / "index.csv"
            table.to_csv(csv_path, index=False)
            LOGGER.warning(
                "Failed to write parquet for split '%s'; fell back to CSV at %s",
                split,
                csv_path,
            )


def _write_manifest(
    output_dir: Path,
    trunc_cfg: TruncationConfig,
    counts: Mapping[str, int],
    shard_meta: Sequence[Tuple[str, int]],
    organ_vocab: Mapping[int, str],
) -> None:
    manifest = {
        "version": "0.1.0",
        "alphabet": list("ACGTN"),
        "encoding": "onehot",
        "shapes": {
            "utr5": [7, trunc_cfg.utr5_len],
            "utr3": [7, trunc_cfg.utr3_len],
        },
        "counts": dict(counts),
        "shards": [
            {"path": os.path.relpath(path, output_dir), "size": size}
            for path, size in shard_meta
        ],
        "organ_vocab": {int(k): str(v) for k, v in organ_vocab.items()},
        "storage_format": "pt",
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    LOGGER.info("Wrote manifest metadata to %s", manifest_path)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    if not logging.getLogger().handlers:
        configure_logging()
    LOGGER.info("Loading dataset configuration from %s", args.dataset_config)
    with open(args.dataset_config, "r", encoding="utf-8") as fh:
        dataset_cfg = yaml.safe_load(fh)

    trunc_cfg = TruncationConfig(
        utr5_len=int(dataset_cfg["truncation"]["utr5"]["max_len"]),
        utr3_head=int(dataset_cfg["truncation"]["utr3"]["head_len"]),
        utr3_tail=int(dataset_cfg["truncation"]["utr3"]["tail_len"]),
    )
    LOGGER.info(
        "Truncation config resolved to utr5=%d, utr3_head=%d, utr3_tail=%d",
        trunc_cfg.utr5_len,
        trunc_cfg.utr3_head,
        trunc_cfg.utr3_tail,
    )

    input_path = Path(dataset_cfg["data"]["input_parquet"])
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    LOGGER.info("Loading base dataset from %s", input_path)
    df = pd.read_parquet(input_path)
    LOGGER.info("Loaded %d rows with %d columns", len(df), len(df.columns))
    gene_col = args.gene_col or "gene_id"
    if gene_col not in df.columns:
        raise KeyError(f"Column '{gene_col}' not found in input data")
    organ_col = args.organ_col or dataset_cfg["colmap"].get("organ", "organ_id")
    utr5_col = args.utr5_col or dataset_cfg["colmap"].get("utr5", "utr5_seq")
    utr3_col = args.utr3_col or dataset_cfg["colmap"].get("utr3", "utr3_seq")
    # Determine which column should be used as the regression target.  When the
    # dataset config already specifies the mapping we prefer that, but fall back
    # to a list of common column names so that older parquet exports continue to
    # work without additional CLI flags.
    colmap = dataset_cfg.get("colmap", {}) if isinstance(dataset_cfg, dict) else {}
    target_col = args.target_col
    if not target_col:
        target_col = colmap.get("y_reg") or colmap.get("target") or colmap.get("label")
    if not target_col:
        for candidate in ("label_reg_log1p", "expr_value", "y", "y_value"):
            if candidate in df.columns:
                target_col = candidate
                break
    if not target_col:
        raise KeyError(
            "Could not determine target column. Pass --target-col explicitly or"
            " set colmap.y_reg in the dataset config."
        )
    split_col = args.split_col or dataset_cfg["colmap"].get("split", "split")

    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found in dataset. Available columns: {list(df.columns)}"
        )
    if organ_col not in df.columns:
        raise KeyError(f"Organ column '{organ_col}' not found in dataset")
    if utr5_col not in df.columns or utr3_col not in df.columns:
        raise KeyError("UTR sequence columns missing from dataset")

    df = df[[gene_col, organ_col, utr5_col, utr3_col, target_col, split_col]].copy()
    df.rename(
        columns={
            gene_col: "gene_id",
            organ_col: "organ_id",
            utr5_col: "utr5_seq",
            utr3_col: "utr3_seq",
            target_col: "target",
            split_col: "split",
        },
        inplace=True,
    )

    organ_codes = df["organ_id"].astype("category")
    df["organ_index"] = organ_codes.cat.codes.astype(np.int64)
    organ_vocab = {int(code): str(cat) for code, cat in enumerate(organ_codes.cat.categories)}
    LOGGER.info("Detected %d unique organs", len(organ_vocab))

    utr_coords = load_utr_coords(args.gtf)
    LOGGER.info("Loaded UTR coordinates for %d genes from %s", len(utr_coords), args.gtf)
    trunc_map = build_truncation_map(utr_coords, trunc_cfg)

    rbp_mask5, rbp_mask3 = build_rbp_masks(Path(args.rbp_dir), trunc_map)
    trna_dist = build_trna_features(Path(args.trna_bed), trunc_map)

    N = len(df)
    LOGGER.info("Preparing feature tensors for %d samples", N)
    x5 = np.zeros((N, 7, trunc_cfg.utr5_len), dtype=np.float32)
    x3 = np.zeros((N, 7, trunc_cfg.utr3_len), dtype=np.float32)
    organ_arr = df["organ_index"].to_numpy(dtype=np.int64)
    target_arr = df["target"].to_numpy(dtype=np.float32)

    progress_interval = max(1, N // 20)
    for i, row in df.iterrows():
        gene_id = row["gene_id"]
        seq5 = row["utr5_seq"]
        seq3 = row["utr3_seq"]
        trunc_info = trunc_map.get(gene_id)
        onehot5 = _one_hot_encode(seq5, trunc_cfg.utr5_len, "tail")
        onehot3 = _one_hot_encode(
            seq3,
            trunc_cfg.utr3_len,
            "ends",
            head_len=trunc_cfg.utr3_head,
            tail_len=trunc_cfg.utr3_tail,
        )
        mask5 = _prepare_rbp_channel(
            rbp_mask5.get(gene_id),
            trunc_cfg.utr5_len,
            align="tail",
        )
        info3 = trunc_info.utr3_head_used if trunc_info else None
        mask3 = _prepare_rbp_channel(
            rbp_mask3.get(gene_id),
            trunc_cfg.utr3_len,
            align="ends",
            head_len=info3,
        )
        dist = trna_dist.get(gene_id, float("inf"))
        norm_val = _normalise_trna_distance(dist, args.trna_cap)
        trna5 = _prepare_trna_channel(norm_val, trunc_cfg.utr5_len)
        trna3 = _prepare_trna_channel(norm_val, trunc_cfg.utr3_len)
        x5[i] = np.vstack([onehot5, mask5.reshape(1, -1), trna5.reshape(1, -1)])
        x3[i] = np.vstack([onehot3, mask3.reshape(1, -1), trna3.reshape(1, -1)])
        if (i + 1) % progress_interval == 0 or i + 1 == N:
            LOGGER.debug("Encoded %d/%d samples", i + 1, N)

    output_dir = Path(args.output_dir)
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Writing dataset shards to %s", shards_dir)

    shard_size = args.shard_size
    shard_meta: List[Tuple[str, int]] = []
    meta_rows: List[Dict[str, object]] = []
    for part_id in range(int(math.ceil(N / shard_size))):
        start = part_id * shard_size
        end = min((part_id + 1) * shard_size, N)
        payload = {
            "utr5": torch.from_numpy(x5[start:end]),
            "utr3": torch.from_numpy(x3[start:end]),
            "organ_id": torch.from_numpy(organ_arr[start:end]),
            "label": torch.from_numpy(target_arr[start:end]),
        }
        shard_path = shards_dir / f"data.part-{part_id:05d}.pt"
        torch.save(payload, shard_path)
        LOGGER.info("Wrote shard %s with %d samples", shard_path, end - start)
        shard_meta.append((str(shard_path), end - start))
        for local_idx, global_idx in enumerate(range(start, end)):
            meta_rows.append(
                {
                    "global_idx": global_idx,
                    "part_id": part_id,
                    "local_idx": local_idx,
                    "organ_id": int(organ_arr[global_idx]),
                    "split": df.iloc[global_idx]["split"],
                }
            )

    meta_df = pd.DataFrame(meta_rows)
    _write_split_indices(meta_df, output_dir)

    counts = {
        "total": int(N),
        "train": int((df["split"].astype(str) == "train").sum()),
        "val": int((df["split"].astype(str) == "val").sum()),
        "test": int((df["split"].astype(str) == "test").sum()),
    }
    _write_manifest(output_dir, trunc_cfg, counts, shard_meta, organ_vocab)
    LOGGER.info(
        "Dataset build complete with %d total samples (train=%d, val=%d, test=%d)",
        counts["total"],
        counts["train"],
        counts["val"],
        counts["test"],
    )
    LOGGER.info("Manifest and metadata written to %s", output_dir)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", required=True, help="YAML file describing the base dataset")
    parser.add_argument("--gtf", required=True, help="Ensembl GTF with transcript annotations")
    parser.add_argument("--rbp-dir", required=True, help="Directory containing eCLIP peak BED files")
    parser.add_argument("--trna-bed", required=True, help="BED file with tRNA loci")
    parser.add_argument("--output-dir", required=True, help="Directory for the processed dataset")
    parser.add_argument("--gene-col", default=None, help="Column name for gene identifiers")
    parser.add_argument("--organ-col", default=None, help="Column name for organ/tissue IDs")
    parser.add_argument("--utr5-col", default=None, help="Column name for 5' UTR sequences")
    parser.add_argument("--utr3-col", default=None, help="Column name for 3' UTR sequences")
    parser.add_argument("--target-col", default=None, help="Column name for regression target")
    parser.add_argument("--split-col", default=None, help="Column name for dataset split")
    parser.add_argument("--shard-size", type=int, default=50000, help="Number of samples per output shard")
    parser.add_argument("--trna-cap", type=float, default=100_000.0, help="Maximum distance for tRNA normalisation")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ...)")
    parser.add_argument("--log-file", default=None, help="Optional path to write detailed logs")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    configure_logging(args.log_level, args.log_file)
    run(args)


if __name__ == "__main__":
    main()
