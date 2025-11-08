# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, time, argparse, random
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, BatchSampler, SequentialSampler
import torch.nn as nn
import torch.optim as optim

from src.cnn_v1.dataset import ShardedUTRDataset
from src.cnn_v1.dataset_side import AugmentedUTRDataset, collate_with_side
from src.cnn_v1.model import DualBranchCNN
from src.cnn_v1.model_side import DualBranchCNNWithSide
from src.features.store import FeatureStore
# ---- 内置 R² 计算（避免外部依赖名不匹配）----
import numpy as _np
def r2_score_np(y_true, y_pred):
    y_true = _np.asarray(y_true); y_pred = _np.asarray(y_pred)
    num = _np.sum((y_true - y_pred)**2)
    den = _np.sum((y_true - y_true.mean())**2)
    return 1.0 - (num/den if den > 0 else _np.nan)

# 栈快照（卡住时： kill -USR1 <PID>）
import faulthandler, signal
faulthandler.enable(all_threads=True)
faulthandler.register(signal.SIGUSR1, file=open("stack.txt","w"), all_threads=True)

def load_manifest(dataset_dir: str):
    path = os.path.join(dataset_dir, "manifest.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# --------（旧）按分片“成组”的 BatchSampler（保留，不再默认使用）--------
class PartGroupedBatchSampler(Sampler):
    """
    给定每个样本所属分片的 part_id（针对“当前 split”），把同一分片内的样本尽量凑成 batch。
    采样器产出的索引是“当前 split 的局部索引”，与 Dataset.__getitem__ 对齐。
    """
    def __init__(self, part_ids, batch_size, drop_last=True, shuffle_groups=True, seed=42):
        self.batch_size = int(batch_size)
        self.drop_last = drop_last
        self.shuffle_groups = shuffle_groups
        self.rng = random.Random(seed)

        # part_id -> [局部索引...]
        buckets = defaultdict(list)
        for i, p in enumerate(part_ids):
            buckets[int(p)].append(i)

        parts = list(buckets.keys())
        if self.shuffle_groups:
            self.rng.shuffle(parts)

        self.batches = []
        for p in parts:
            lst = buckets[p]
            if self.shuffle_groups:
                self.rng.shuffle(lst)
            for i in range(0, len(lst), self.batch_size):
                chunk = lst[i:i+self.batch_size]
                if len(chunk) < self.batch_size and self.drop_last:
                    continue
                self.batches.append(chunk)

        if self.shuffle_groups:
            self.rng.shuffle(self.batches)

    def __iter__(self):
        for b in self.batches:
            yield b

    def __len__(self):
        return len(self.batches)

# --------（新）“块采样器”：顺序遍历保障分片局部性；按块随机、块内可打散 -------
class _PartBlockSampler(Sampler[int]):
    """
    核心思路：
      - 用 SequentialSampler(dataset) 保持“顺序=分片局部性”（假设索引写入已按分片连续）
      - 把顺序索引切成 block（约 2~3 个 batch）
      - 随机打乱“块顺序”，可选地对“块内”再打散
      - 仍不跨分片；I/O 局部性与吞吐保持
    """
    def __init__(self, dataset, block_size: int, intra_shuffle: bool, seed: int = 0):
        self.base = SequentialSampler(dataset)
        self.block = int(block_size)
        self.intra = bool(intra_shuffle)
        self.rng = random.Random(int(seed))
        self._indices = None

    def __iter__(self):
        if self._indices is None:
            self._indices = list(iter(self.base))
        idxs = self._indices
        if self.block <= 0:
            # 退化为完全顺序
            for j in idxs:
                yield j
            return
        blocks = [idxs[i:i+self.block] for i in range(0, len(idxs), self.block)]
        self.rng.shuffle(blocks)                  # 打乱块顺序（保持分片局部）
        for b in blocks:
            if self.intra:
                self.rng.shuffle(b)               # ★块内打散（仍在分片内）
            for j in b:
                yield j

    def __len__(self):
        if self._indices is not None:
            return len(self._indices)
        # 未 materialize 时，退回 base 的长度
        return len(list(iter(self.base)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default="data/processed/seq_cnn_v1_reg")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--batch_size", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--huber_beta", type=float, default=1.0)
    ap.add_argument("--logdir", default="outputs/cnn_v1_reg_side")
    # DataLoader 控制
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--prefetch", type=int, default=2)
    ap.add_argument("--pin-memory", type=int, default=1)
    ap.add_argument("--persistent-workers", type=int, default=1)
    ap.add_argument("--print-every", type=int, default=200)
    # 采样策略
    ap.add_argument("--sampler", choices=["part","default"], default="part")  # ← 默认按分片局部
    ap.add_argument("--drop-last", type=int, default=1)
    # 侧表
    ap.add_argument("--tissue", default=None)
    ap.add_argument("--rbp", default=None)
    ap.add_argument("--struct", default=None)
    ap.add_argument("--organ-id-type", default="auto", choices=["auto","int","str"])
    # --- 评估与保存相关开关 ---
    ap.add_argument("--save-eval-md", type=int, default=1,
                    help="将每轮val的macro R2与Top/Bottom-6写入metrics_round1.md")
    ap.add_argument("--save-best-by", choices=["val_loss","macro_r2"], default="macro_r2",
                    help="best.pt的度量：val_loss越小越好或macro_r2越大越好（默认macro_r2）")
    # --- 改动 A：逐目标标准化（与 017 对齐） ---
    ap.add_argument("--standardize-y", type=int, default=1,
                    help="对每个目标做标准化(y-mean)/std；评估前再反标准化")
    # --- 改动 B：块采样器参数 ---
    ap.add_argument("--block-size", type=int, default=1024,
                    help="sampler=part 时的分片内块大小（约 2~3 个 batch）")
    ap.add_argument("--intra-block-shuffle", type=int, default=1,
                    help="是否对每个块内的样本顺序打乱")

    args = ap.parse_args()

    print(f"[debug] PID={os.getpid()}  卡住时执行： kill -USR1 {os.getpid()}")
    os.makedirs(args.logdir, exist_ok=True)
    # checkpoints 目录
    ckpt_dir = os.path.join(args.logdir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    torch.backends.cudnn.benchmark = True

    fs = FeatureStore(tissue_path=args.tissue, rbp_path=args.rbp, struct_path=args.struct,
                      organ_id_type=args.organ_id_type)

    man = load_manifest(args.dataset_dir)
    out_dim = int(man.get("y_dim", man.get("num_outputs", 54)))
    part_sizes = [int(p["size"]) for p in man.get("parts", [])]  # 保留统计/日志用途

    # 读取 label 名称
    def _load_label_names(dataset_dir, k):
        lm = os.path.join(dataset_dir, "label_mapping.json")
        if os.path.exists(lm):
            m = json.load(open(lm, "r", encoding="utf-8"))
            if isinstance(m, list) and len(m) == k:
                return m
            if isinstance(m, dict) and len(m) == k:
                return [m.get(str(i), str(i)) for i in range(k)]
        return [f"task_{i}" for i in range(k)]
    label_names = _load_label_names(args.dataset_dir, out_dim)

    def build_loader(split):
        base = ShardedUTRDataset(args.dataset_dir, split=split)
        # 优先 JSON 索引；存在就用（避免 parquet 的慢路径）
        index_json = os.path.join(args.dataset_dir, "index", f"{split}.json")
        index_json = index_json if os.path.exists(index_json) else None
        aux = AugmentedUTRDataset(base, fs, index_json=index_json)

        if args.sampler == "part":
            # ★ 新：块采样器，保障分片局部性 + 增加随机度
            part_block = _PartBlockSampler(aux, args.block_size, bool(args.intra_block_shuffle), seed=0)
            return DataLoader(
                aux,
                batch_sampler=BatchSampler(part_block, batch_size=args.batch_size, drop_last=bool(args.drop_last)),
                num_workers=args.num_workers,
                pin_memory=bool(args.pin_memory),
                prefetch_factor=args.prefetch if args.num_workers > 0 else None,
                persistent_workers=bool(args.persistent_workers) if args.num_workers > 0 else False,
                collate_fn=collate_with_side,
            )
        else:
            return DataLoader(
                aux,
                batch_size=args.batch_size,
                shuffle=(split=="train"),
                num_workers=args.num_workers,
                pin_memory=bool(args.pin_memory),
                prefetch_factor=args.prefetch if args.num_workers > 0 else None,
                persistent_workers=bool(args.persistent_workers) if args.num_workers > 0 else False,
                collate_fn=collate_with_side,
            )

    loaders = {s: build_loader(s) for s in args.splits.split(",")}

    base = DualBranchCNN(in_ch=5, emb_dim=8, channels=[64,128,256], num_classes=out_dim)
    dims = fs.dims
    print(f"[Side] dims: {dims} (organ_id_type={args.organ_id_type})")
    model = DualBranchCNNWithSide(base, side_dims=dims, out_dim=out_dim, hidden=256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    crit = nn.HuberLoss(delta=args.huber_beta)
    opt  = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # --- 改动 A：统计 / 读取 y 的 mean/std（只做一次并缓存到 logdir/y_stats.json） ---
    y_stats_path = os.path.join(args.logdir, "y_stats.json")
    y_mean = None
    y_std = None
    if bool(args.standardize_y):
        if os.path.exists(y_stats_path):
            js = json.load(open(y_stats_path, "r"))
            y_mean = np.array(js["mean"], dtype=np.float32)
            y_std  = np.array(js["std"],  dtype=np.float32)
            print(f"[y_stats] loaded from {y_stats_path}")
        else:
            print("[y_stats] computing from train loader once ...")
            tot = 0
            s1 = np.zeros(out_dim, dtype=np.float64)
            s2 = np.zeros(out_dim, dtype=np.float64)
            with torch.no_grad():
                for batch in loaders["train"]:
                    y  = batch["label"].float().numpy()   # (B, D)
                    s1 += y.sum(axis=0)
                    s2 += (y * y).sum(axis=0)
                    tot += y.shape[0]
            y_mean = (s1 / max(tot, 1)).astype(np.float32)
            var = (s2 / max(tot, 1)) - (y_mean.astype(np.float64) ** 2)
            var = np.maximum(var, 0.0)
            y_std = np.sqrt(var).astype(np.float32)
            json.dump({"mean": y_mean.tolist(), "std": y_std.tolist()},
                      open(y_stats_path, "w"), indent=2)
            print(f"[y_stats] saved to {y_stats_path}")

        # 打印几项（与 017 对齐的可见性）
        def _preview(arr, k=6):
            return ", ".join([f"{arr[i]:.4g}" for i in range(min(k, len(arr)))])
        print(f"[y_stats] y_mean[:6]=[{_preview(y_mean)}]")
        print(f"[y_stats] y_std [:6]=[{_preview(y_std)}]")

        # to torch tensors（常驻 device）
        y_mean_t = torch.from_numpy(y_mean).to(device)
        # 防止除 0：用 1e-6 下限
        y_std_t  = torch.from_numpy(np.clip(y_std, 1e-6, None)).to(device)
    else:
        y_mean_t = None
        y_std_t = None

    # --- 评估函数（支持反标准化后算 R² 与记录） ---
    def eval_split(model, loader, device, label_names, write_to=None,
                   y_mean_t: torch.Tensor | None = None, y_std_t: torch.Tensor | None = None,
                   standardize: bool = False):
        model.eval()
        Ys, Ps = [], []
        with torch.no_grad():
            for batch in loader:
                x5 = batch["utr5"].to(device, non_blocking=True).float()
                x3 = batch["utr3"].to(device, non_blocking=True).float()
                y  = batch["label"].to(device, non_blocking=True).float()
                side = {k: v.to(device, non_blocking=True).float() for k,v in batch.get("side", {}).items()}
                yhat = model(x5, x3, side)  # 原始尺度的输出
                if standardize and (y_mean_t is not None) and (y_std_t is not None):
                    # 按说明：先转到“标准化空间”的 yhat_std，再反标准化回来再评估
                    yhat_std = (yhat - y_mean_t) / y_std_t
                    yhat = yhat_std * y_std_t + y_mean_t
                Ys.append(y.cpu().numpy()); Ps.append(yhat.cpu().numpy())
        if len(Ys) == 0:
            print("[val] 空验证集，跳过 R² 评估")
            return float("nan")
        Y = np.concatenate(Ys, 0); P = np.concatenate(Ps, 0)
        per = []
        for j in range(P.shape[1]):
            yj, pj = Y[:, j], P[:, j]
            if np.allclose(yj.var(), 0):
                r2 = float("nan")
            else:
                r2 = float(r2_score_np(yj, pj))
            per.append({"idx": j, "name": label_names[j], "r2": r2})
        macro = float(np.nanmean([d["r2"] for d in per]))
        keep = [d for d in per if not np.isnan(d["r2"])]
        top6 = sorted(keep, key=lambda d: d["r2"], reverse=True)[:6]
        bot6 = sorted(keep, key=lambda d: d["r2"])[:6]
        print(f"[val] macro_R2={macro:.4f}  top6={[ (d['name'], round(d['r2'],4)) for d in top6 ]}  "
              f"bottom6={[ (d['name'], round(d['r2'],4)) for d in bot6 ]}")
        if write_to is not None:
            json.dump({"macro_R2": macro, "per_task": per, "top6": top6, "bottom6": bot6},
                      open(os.path.join(write_to, "eval_val.json"), "w"), ensure_ascii=False, indent=2)
            if args.save_eval_md:
                with open(os.path.join(write_to, "metrics_round1.md"), "a", encoding="utf-8") as f:
                    f.write(f"## val\n\nmacro R²: **{macro:.4f}**\n\n")
                    f.write("|rank|task|R²|\n|---:|:---|---:|\n")
                    for r, d in enumerate(top6, 1): f.write(f"|{r}|{d['name']}|{d['r2']:.4f}|\n")
                    f.write("\nBottom-6:\n\n|rank|task|R²|\n|---:|:---|---:|\n")
                    for r, d in enumerate(bot6, 1): f.write(f"|{r}|{d['name']}|{d['r2']:.4f}|\n\n")
        return macro

    # warmup：拿到首个 batch（触发 DataLoader 构建）
    wb = next(iter(loaders["train"]))
    print(f"[warmup] got first batch: x5={tuple(wb['utr5'].shape)} x3={tuple(wb['utr3'].shape)}")

    for epoch in range(1, args.epochs+1):
        model.train()
        tot = 0.0; n = 0; step = 0
        t0 = time.time()
        for batch in loaders["train"]:
            step += 1
            x5 = batch["utr5"].to(device, non_blocking=True).float()
            x3 = batch["utr3"].to(device, non_blocking=True).float()
            y  = batch["label"].to(device, non_blocking=True).float()
            side = {k: v.to(device, non_blocking=True).float() for k,v in batch.get("side", {}).items()}

            opt.zero_grad(set_to_none=True)
            yhat = model(x5, x3, side)

            if bool(args.standardize_y) and (y_mean_t is not None) and (y_std_t is not None):
                # 在“标准化空间”计算损失
                y_std   = (y    - y_mean_t) / y_std_t
                yhat_std= (yhat - y_mean_t) / y_std_t
                loss = crit(yhat_std, y_std)
            else:
                loss = crit(yhat, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tot += float(loss.item()) * y.size(0); n += y.size(0)

            if step % max(1, args.print_every) == 0:
                dt = time.time() - t0
                sps = (args.print_every*args.batch_size)/max(dt,1e-6)
                print(f"[train] step={step} dt={dt:.2f}s ~{sps:.1f} samples/s loss={loss.item():.4f}")
                t0 = time.time()
        train_loss = tot / max(n,1)

        # val（loss 仍在标准化空间上度量，保持可比性）
        model.eval()
        with torch.no_grad():
            tot = 0.0; n=0
            for batch in loaders.get("val", []):
                x5 = batch["utr5"].to(device, non_blocking=True).float()
                x3 = batch["utr3"].to(device, non_blocking=True).float()
                y  = batch["label"].to(device, non_blocking=True).float()
                side = {k: v.to(device, non_blocking=True).float() for k,v in batch.get("side", {}).items()}
                yhat = model(x5, x3, side)
                if bool(args.standardize_y) and (y_mean_t is not None) and (y_std_t is not None):
                    y_std    = (y    - y_mean_t) / y_std_t
                    yhat_std = (yhat - y_mean_t) / y_std_t
                    loss = crit(yhat_std, y_std)
                else:
                    loss = crit(yhat, y)
                tot += float(loss.item()) * y.size(0); n += y.size(0)
            val_loss = tot / max(n,1) if n>0 else float("nan")
        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        # 计算 val 的 macro R² + Top/Bottom-6（按说明：反标准化后评估）
        val_macro_r2 = eval_split(
            model, loaders.get("val", []), device, label_names,
            write_to=args.logdir,
            y_mean_t=y_mean_t, y_std_t=y_std_t,
            standardize=bool(args.standardize_y)
        )

        # 始终保存 last.pt
        torch.save({"epoch": epoch, "state_dict": model.state_dict()},
                   os.path.join(ckpt_dir, "last.pt"))

        # 依据选择的度量保存 best.pt
        if epoch == 1 and not hasattr(main, "_best_inited"):
            main._best_inited = True
            if args.save_best_by == "val_loss":
                main._best_val = val_loss
            else:
                main._best_val = val_macro_r2

        better = False
        if args.save_best_by == "val_loss":
            if val_loss <= main._best_val:
                better = True
                main._best_val = val_loss
        else:  # macro_r2 越大越好
            if val_macro_r2 >= main._best_val:
                better = True
                main._best_val = val_macro_r2

        if better:
            torch.save({"epoch": epoch, "state_dict": model.state_dict()},
                       os.path.join(ckpt_dir, "best.pt"))
            tag = (f"val_loss={val_loss:.4f}" if args.save_best_by=="val_loss"
                   else f"macro_R2={val_macro_r2:.4f}")
            print(f"[ckpt] saved best.pt @ epoch {epoch} ({tag})")

    # （可选）测试略
    # ...

if __name__ == "__main__":
    main()
