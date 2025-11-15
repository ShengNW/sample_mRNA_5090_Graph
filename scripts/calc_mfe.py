"""
计算 UTR 的最小自由能（MFE）。
需要系统已安装 `RNAfold`（ViennaRNA）。将从输入 CSV 中读取 `utr5` 和/或 `utr3` 列，分别计算并合并。
输出列：seq_id, mfe_utr5, mfe_utr3

原始实现会一次性把所有序列写入临时 FASTA，单进程调用 RNAfold 顺序处理，
因此 CPU 利用率极低且几乎没有日志。这里保留原有单条序列计算逻辑，
但通过多进程、分块与进度日志来提升吞吐与可观测性。
"""
import argparse
import logging
import math
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils import read_csv_required, save_merged_csv

DEFAULT_LOG_FILE = "/tmp/calc_mfe.log"
DEFAULT_CHUNK_SIZE = 256
DEFAULT_PROGRESS_INTERVAL = 1000
MAX_WORKERS = 24


def run_rnafold(seq_list):
    """串行调用 RNAfold 处理一个分块，供主进程或进程池复用。"""
    # 使用临时文件批量调用 RNAfold
    with tempfile.TemporaryDirectory() as td:
        in_f = os.path.join(td, "in.fa")
        out_f = os.path.join(td, "out.txt")
        with open(in_f, "w") as f:
            for i, s in enumerate(seq_list):
                f.write(f">s{i}\n{s}\n")
        # -d2：对 dangling ends 的处理更接近常用设置；--noPS：不生成 ps 图片
        cmd = ["RNAfold", "--noPS", "-d2"]
        with open(in_f, "r") as fin, open(out_f, "w") as fout:
            subprocess.run(
                cmd,
                stdin=fin,
                stdout=fout,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
        # 解析输出：每两行一条，第二行括号里包含能量
        mfes = []
        seq_line = None
        with open(out_f) as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith(">"):
                    continue
                if seq_line is None:
                    # RNAfold 输出的第一行是序列，第二行才包含结构 + 能量
                    seq_line = line
                    continue
                # 末尾类似  "....((...))... (-12.30)"
                energy = float(line[line.rfind("(") + 1 : line.rfind(")")])
                mfes.append(energy)
                seq_line = None
        return mfes


def setup_logger(log_file):
    """同时把日志写到 stdout 和文件，方便本地与远端 tail。"""
    logger = logging.getLogger("calc_mfe")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def determine_workers(requested):
    cpu_count = os.cpu_count() or 1
    if requested is not None:
        if requested < 1:
            raise ValueError("--num-workers must be >= 1")
        return requested, cpu_count
    default_workers = max(1, min(cpu_count - 1 if cpu_count > 1 else 1, MAX_WORKERS))
    return default_workers, cpu_count


def chunk_indices(total, chunk_size):
    for start in range(0, total, chunk_size):
        end = min(total, start + chunk_size)
        yield start, end


def log_rnafold_failure(logger, column_name, start, seq_ids, exc):
    seq_preview = ", ".join(seq_ids[:3])
    if len(seq_ids) > 3:
        seq_preview += ", ..."
    row_range = f"{start}-{start + len(seq_ids) - 1}"
    stderr = getattr(exc, "stderr", "") or ""
    stderr = stderr.strip()
    logger.error(
        "%s: RNAfold failed on rows %s (seq_ids: %s). stderr: %s",
        column_name,
        row_range,
        seq_preview,
        stderr,
    )


def compute_mfe_column(
    column_name,
    sequences,
    seq_ids,
    num_workers,
    chunk_size,
    progress_interval,
    logger,
):
    """
    并行计算指定列 MFE。
    通过记录开始 index 并在结果写回时使用切片，保证输出顺序与输入完全一致。
    """
    total = len(sequences)
    if total == 0:
        return []

    logger.info(
        "%s: start computing MFE for %d sequences (workers=%d, chunk_size=%d)",
        column_name,
        total,
        num_workers,
        chunk_size,
    )
    start_time = time.time()
    results = [None] * total
    completed = 0
    next_log_threshold = progress_interval

    def maybe_log(force=False):
        nonlocal next_log_threshold
        if not force and completed < next_log_threshold:
            return
        elapsed = time.time() - start_time
        throughput = completed / elapsed if elapsed > 0 else 0.0
        remaining = (total - completed) / throughput if throughput > 0 else float("inf")
        eta = (
            f"{remaining:.1f}s"
            if math.isfinite(remaining) and remaining >= 0
            else "unknown"
        )
        pct = (completed / total) * 100 if total else 100.0
        logger.info(
            "%s: progress %d/%d (%.2f%%), elapsed %.1fs, %.2f seq/s, ETA %s",
            column_name,
            completed,
            total,
            pct,
            elapsed,
            throughput,
            eta,
        )
        while next_log_threshold <= completed:
            next_log_threshold += progress_interval

    if num_workers == 1:
        for start, end in chunk_indices(total, chunk_size):
            seq_chunk = sequences[start:end]
            try:
                mfes = run_rnafold(seq_chunk)
            except subprocess.CalledProcessError as exc:
                log_rnafold_failure(logger, column_name, start, seq_ids[start:end], exc)
                raise
            results[start:end] = mfes
            completed += len(mfes)
            maybe_log()
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_meta = {}
            for start, end in chunk_indices(total, chunk_size):
                seq_chunk = sequences[start:end]
                future = executor.submit(run_rnafold, seq_chunk)
                future_to_meta[future] = (start, end)
            for future in as_completed(future_to_meta):
                start, end = future_to_meta[future]
                try:
                    mfes = future.result()
                except subprocess.CalledProcessError as exc:
                    log_rnafold_failure(
                        logger, column_name, start, seq_ids[start:end], exc
                    )
                    raise
                results[start:end] = mfes
                completed += len(mfes)
                maybe_log()

    maybe_log(force=True)
    total_time = time.time() - start_time
    avg_rate = total / total_time if total_time > 0 else 0.0
    logger.info(
        "%s: completed %d sequences in %.1fs (avg %.2f seq/s)",
        column_name,
        total,
        total_time,
        avg_rate,
    )
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker processes (default: cpu_count-1, capped at 24)",
    )
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Number of sequences per RNAfold invocation (default: 256)",
    )
    ap.add_argument(
        "--progress-interval",
        type=int,
        default=None,
        help="Log progress every N sequences (default: 1000)",
    )
    ap.add_argument(
        "--log-file", default=DEFAULT_LOG_FILE, help="Log file path (default: /tmp/calc_mfe.log)"
    )
    args = ap.parse_args()

    if args.chunk_size is not None and args.chunk_size < 1:
        ap.error("--chunk-size must be >= 1")
    if args.progress_interval is not None and args.progress_interval < 1:
        ap.error("--progress-interval must be >= 1")

    num_workers, cpu_count = determine_workers(args.num_workers)
    chunk_size = args.chunk_size or DEFAULT_CHUNK_SIZE
    progress_interval = args.progress_interval or DEFAULT_PROGRESS_INTERVAL
    logger = setup_logger(args.log_file)
    logger.info(
        "calc_mfe start | input=%s output=%s num_workers=%d cpu_count=%d chunk_size=%d progress_interval=%d log_file=%s",
        args.input,
        args.output,
        num_workers,
        cpu_count,
        chunk_size,
        progress_interval,
        args.log_file,
    )

    df = read_csv_required(args.input, ["seq_id"])
    out = df[["seq_id"]].copy()
    seq_ids = df["seq_id"].astype(str).tolist()

    if "utr5" in df.columns:
        sequences = df["utr5"].astype(str).tolist()
        out["mfe_utr5"] = compute_mfe_column(
            "utr5",
            sequences,
            seq_ids,
            num_workers,
            chunk_size,
            progress_interval,
            logger,
        )
    if "utr3" in df.columns:
        sequences = df["utr3"].astype(str).tolist()
        out["mfe_utr3"] = compute_mfe_column(
            "utr3",
            sequences,
            seq_ids,
            num_workers,
            chunk_size,
            progress_interval,
            logger,
        )

    save_merged_csv(out, args.output)
    logger.info("Saved %s with columns: %s", args.output, out.columns.tolist())


if __name__ == "__main__":
    main()
