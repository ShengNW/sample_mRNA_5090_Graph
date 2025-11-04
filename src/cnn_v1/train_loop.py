from __future__ import annotations
import os, math, time
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler
from .dataset import ShardedUTRDataset
from .model import DualBranchCNN
from .utils import ensure_dir, set_seed
import torch.backends.cudnn as cudnn


class PartGroupedSampler(Sampler):
    """
    分片感知采样：分片之间随机、分片内顺序；可选把每个分片再按 block 切块并打乱块顺序。
    兼顾 I/O 连续性与梯度噪声。
    """
    def __init__(self, dataset, block_size=4096, seed=0):
        self.ds = dataset
        self.block = int(block_size) if block_size and block_size > 0 else None
        self.rng = np.random.RandomState(seed)
        # 按分片把全局索引分组（需要 dataset 里已有 self._idx_part / self._idx_local）
        parts = getattr(dataset, "_idx_part", None)
        if parts is None:
            raise ValueError("dataset 缺少 _idx_part/_idx_local（请确认使用 ShardedUTRDataset 的 split 索引路径）")
        self.groups = []
        for p in np.unique(parts):
            idxs = np.where(parts == p)[0]
            self.groups.append(idxs)
        self.epoch = 0

    def set_epoch(self, ep: int):
        self.epoch = ep

    def __iter__(self):
        # 分片顺序打乱
        order = list(range(len(self.groups)))
        self.rng.shuffle(order)
        for gi in order:
            idxs = self.groups[gi]
            if self.block:
                # 分片内按 block 切块，随机重排块序，块内顺序不变
                n = (len(idxs) + self.block - 1) // self.block
                blocks = [idxs[i*self.block:(i+1)*self.block] for i in range(n)]
                self.rng.shuffle(blocks)
                for b in blocks:
                    for j in b:
                        yield int(j)
            else:
                for j in idxs:
                    yield int(j)

    def __len__(self):
        return int(sum(g.size for g in self.groups))


# ===== 新增：固定一小撮样本重复训练的采样器（用于“小样本过拟合”自检） =====
class OverfitSubsetSampler(Sampler):
    def __init__(self, dataset, n_samples=2048, seed=0):
        self.n = min(int(n_samples), len(dataset))
        self.idxs = np.random.RandomState(int(seed)).choice(len(dataset), self.n, replace=False)
    def __iter__(self):
        # 每个 epoch 内反复遍历同一小撮（顺序打乱）
        order = np.random.permutation(self.idxs)
        for i in order:
            yield int(i)
    def __len__(self):
        return int(self.n)


class AverageMeter:
    def __init__(self):
        self.v=0; self.s=0
    def update(self, val, n=1):
        self.v += val*n; self.s += n
    @property
    def avg(self):
        return self.v/max(1,self.s)


def _worker_init_fn(worker_id: int):
    # 轻量 worker 初始化：独立随机种子 + 限制每 worker 线程数，避免过度并行
    import random, numpy as _np
    seed = torch.initial_seed() % 2**32
    random.seed(seed); _np.random.seed(seed)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')



def _build_criterion(cfg: Dict[str, Any], device: str | torch.device = "cpu") -> nn.Module:
    """
    Build loss by task type.
    - classification  -> CrossEntropyLoss (optionally with class weights / label smoothing)
    - regression      -> MSELoss
    """
    task = str(cfg.get('task', 'classification')).lower()
    if task in ('regression', 'mtl_regression', 'multi_target_regression'):
        # Multi-target regression: predict a vector and minimize MSE
        return nn.MSELoss()
    # === default: classification ===
    loss_cfg = cfg.get('loss', {})
    label_smoothing = float(loss_cfg.get('label_smoothing', 0.0))
    class_weights = loss_cfg.get('class_weights', None)
    weight_tensor: Optional[torch.Tensor] = None
    if isinstance(class_weights, str) and os.path.exists(class_weights):
        # 读取 JSON 列表或 {"weights": [...]} 两种格式
        import json
        with open(class_weights, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        if isinstance(obj, dict) and 'weights' in obj:
            obj = obj['weights']
        weight_tensor = torch.tensor(obj, dtype=torch.float32, device=device)
    elif isinstance(class_weights, str) and class_weights.lower() == "auto":
        # 自动根据训练 split 的 class_idx 频次计算反频率权重
        try:
            import pandas as _pd
        except Exception:
            raise RuntimeError("class_weights=auto 需要 pandas。请 `pip install pandas pyarrow`。")
        ddir = cfg['data']['dataset_dir']
        ip = os.path.join(ddir, "index", "train", "index.parquet")
        if not os.path.exists(ip):
            ip = os.path.join(ddir, "index", "train", "index.csv")
        df = _pd.read_parquet(ip) if ip.endswith(".parquet") else _pd.read_csv(ip)
        counts = df["class_idx"].value_counts().sort_index()
        # 防止除零：clip 到 1
        weights = 1.0 / counts.clip(lower=1).to_numpy(dtype=float)
        # 归一到均值=1（数值更稳）
        weights = weights / weights.mean()
        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)
    return criterion




def build_model(n_channels: int, cfg: Dict[str, Any]) -> torch.nn.Module:
    mcfg = cfg['model']
    emb = int(mcfg.get('embedding', {}).get('dim', 8))
    share = bool(mcfg.get('embedding', {}).get('share_between_utr5_utr3', False))
    trunk = mcfg.get('trunk', {})
    channels = tuple(trunk.get('channels', [64,128,256]))
    task = str(cfg.get('task', 'classification')).lower()
    if task in ('regression','mtl_regression','multi_target_regression'):
        out_dim = int(cfg['data'].get('num_targets', cfg['data'].get('num_classes', 54)))
    else:
        out_dim = int(cfg['data']['num_classes'])
    model = DualBranchCNN(in_ch=n_channels, num_classes=out_dim,
                          emb_dim=emb, channels=channels, share_branches=share)
    return model



def infer_n_channels(sample_batch) -> int:
    c5 = int(sample_batch['utr5'].shape[0])
    c3 = int(sample_batch['utr3'].shape[0])
    assert c5==c3, f"两路通道数不一致: {c5} vs {c3}"
    return c5


# ===== 新增：带 BN/bias 排除的参数分组优化器构造 =====
def build_optimizer_with_bn_exclusion(model, optim_cfg):
    import torch
    import torch.nn as nn
    lr = float(optim_cfg.get('lr', 3e-4))
    wd = float(optim_cfg.get('weight_decay', 1e-2))
    decay, no_decay = [], []

    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm1d):
            # BN 的 weight/bias 一律不做权重衰减
            for p in [module.weight, module.bias]:
                if p is not None and p.requires_grad:
                    no_decay.append(p)

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # 已经收录到 no_decay 的跳过
        if any(p is q for q in no_decay):
            continue
        # 其余 bias 一律不做衰减
        if name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)

    param_groups = [
        {"params": decay, "weight_decay": wd},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    opt_name = str(optim_cfg.get('optimizer', 'adamw')).lower()
    if opt_name == "adam":
        return torch.optim.Adam(param_groups, lr=lr)
    else:
        return torch.optim.AdamW(param_groups, lr=lr)


def train_one_epoch(model, loader, optimizer, device, criterion, task, num_classes,
                    scaler=None, amp_enabled=False, head_only: bool=False):
    model.train()
    loss_meter = AverageMeter()
    correct=0; total=0; i_SNW = 0
    t0 = time.time()
    for batch in loader:
        data_ready = time.time()

        x5 = batch['utr5'].to(device, non_blocking=True)
        x3 = batch['utr3'].to(device, non_blocking=True)
        y  = batch['label'].to(device, non_blocking=True)

        # --- 检查 1）：第一批输入是否几乎常量（只打印一次） ---
        if i_SNW == 0:
            with torch.no_grad():
                m5 = x5.mean().item(); s5 = x5.std().item()
                m3 = x3.mean().item(); s3 = x3.std().item()
            print(f"[debug] x5 mean={m5:.4f} std={s5:.4f} | x3 mean={m3:.4f} std={s3:.4f}")
            # --- 检查 2）：第一批标签直方图（只打印一次） ---
            #y_np = y.detach().cpu().numpy()
            #counts = np.bincount(y_np, minlength=int(num_classes))

            # 对于回归任务（y 为浮点或多维），不要用 bincount
            is_regression = (num_classes is None) or torch.is_floating_point(y)
            if not is_regression:
                y_np = y.detach().view(-1).to(torch.int64).cpu().numpy()
                counts = np.bincount(y_np, minlength=int(num_classes))
            else:
                counts = None
                # 如需日志统计，可用直方图（可选）
                # y_hist, y_edges = np.histogram(y.detach().view(-1).cpu().numpy(), bins=10)

            #print("[debug] label hist (first batch):", counts.tolist(), "sum=", int(counts.sum()))
            # --- 新增：主干特征统计（只打印一次） ---
            try:
                with torch.no_grad():
                    f5 = model.branch5(x5); f3 = model.branch3(x3)
                    h  = torch.cat([f5, f3], dim=-1)
                    s_f5 = f5.std().item(); s_f3 = f3.std().item(); s_h = h.std().item()
                print(f"[debug] f5 std={s_f5:.4f} f3 std={s_f3:.4f} h std={s_h:.4f}")
            except Exception as _e:
                print("[debug] trunk feature stats skipped:", repr(_e))
            # --- 新增：只训 head 时确认可训练参数数量（只打印一次） ---
            if head_only:
                n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"[debug] head-only mode enabled, trainable params={n_trainable}")
            # --- 新增：BN γ 监控（只在每个 epoch 的第一批后打印一次） ---
            with torch.no_grad():
                gammas = [m.weight.detach().abs().mean().item()
                          for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
                if len(gammas) > 0:
                    print(f"[debug] bn.gamma mean={np.mean(gammas):.4f} min={np.min(gammas):.4f}")
                else:
                    print("[debug] bn.gamma monitor: no BatchNorm1d modules found")

        optimizer.zero_grad(set_to_none=True)
        if amp_enabled and scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x5, x3)
                loss = criterion(logits, y.float() if task.startswith('reg') else y)
            # --- 实证排错A：第一步，在 backward 之后打印一次 head 权重/梯度范数 ---
            scaler.scale(loss).backward()
            if i_SNW == 0:
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        if n.endswith('head.3.weight'):
                            print('[debug] head.W norm', p.norm().item(),
                                  'grad', (p.grad.norm().item() if p.grad is not None else -1))
                            break
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x5, x3)
            loss = criterion(logits, y.float() if task.startswith('reg') else y)
            # --- 实证排错A：第一步，在 backward 之后打印一次 head 权重/梯度范数 ---
            loss.backward()
            if i_SNW == 0:
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        if n.endswith('head.3.weight'):
                            print('[debug] head.W norm', p.norm().item(),
                                  'grad', (p.grad.norm().item() if p.grad is not None else -1))
                            break
            optimizer.step()


        loss_meter.update(loss.item(), y.size(0))
        if task.startswith('reg'):
            # accumulate for RMSE if desired (we keep it simple: compute on-the-fly average MSE via loss_meter)
            pass
        else:
            pred = torch.argmax(logits, dim=-1)
            correct += (pred==y).sum().item()
            total += y.size(0)

        # --- 先记录 step_done，再打印日志（保持 3.2 调整） ---
        step_done = time.time()

        i_SNW += 1
        if i_SNW % 200 == 0:
            print(f"[train] step={i_SNW} data_time={data_ready-t0:.3f}s "
                  f"step_time={step_done-data_ready:.3f}s loss={loss.item():.4f}")

        t0 = time.time()

    if task.startswith('reg'):
        # approximate RMSE from averaged loss (MSE); for exact compute, need accumulated preds
        rmse = float(math.sqrt(max(loss_meter.avg, 0.0)))
        return {'loss': loss_meter.avg, 'rmse': rmse}
    else:
        acc = correct/max(1,total)
        return {'loss': loss_meter.avg, 'acc': acc}




def evaluate(model, loader, device, criterion, task):
    model.eval()
    loss_meter = AverageMeter()
    if task.startswith('reg'):
        all_pred=[]; all_y=[]
    else:
        correct=0; total=0
        all_logits=[]; all_y=[]
    with torch.no_grad():
        for batch in loader:
            x5 = batch['utr5'].to(device, non_blocking=True)
            x3 = batch['utr3'].to(device, non_blocking=True)
            y  = batch['label'].to(device, non_blocking=True)
            logits = model(x5, x3)
            loss = criterion(logits, y.float() if task.startswith('reg') else y)
            loss_meter.update(loss.item(), y.size(0))
            if task.startswith('reg'):
                all_pred.append(logits.cpu())
                all_y.append(y.float().cpu())
            else:
                pred = torch.argmax(logits, dim=-1)
                correct += (pred==y).sum().item()
                total += y.size(0)
                all_logits.append(logits.cpu())
                all_y.append(y.cpu())
    if task.startswith('reg'):
        y_true = torch.cat(all_y, dim=0).numpy() if all_y else np.zeros((0,54), dtype=np.float32)
        y_pred = torch.cat(all_pred, dim=0).numpy() if all_pred else np.zeros_like(y_true)
        # macro RMSE across targets
        mse = np.mean((y_pred - y_true)**2, axis=0)
        rmse = float(np.sqrt(np.mean(mse)))
        # R^2 macro
        var = np.var(y_true, axis=0)
        r2  = float(1.0 - np.mean(mse / (var + 1e-12)))
        return {'loss': loss_meter.avg, 'rmse': rmse, 'r2': r2, 'y_true': y_true, 'y_pred': y_pred}
    else:
        acc = correct/max(1,total)
        logits = torch.cat(all_logits, dim=0).numpy() if all_logits else np.zeros((0,))
        y_true = torch.cat(all_y, dim=0).numpy() if all_y else np.zeros((0,), dtype=int)
        return {'loss': loss_meter.avg, 'acc': acc, 'logits': logits, 'y_true': y_true}



def run_training(cfg: Dict[str, Any]):
    # --- 实证排错B：唯一启动标记（用于确认跑到的是这份代码） ---
    print("[stamp] train_loop v2025-10-28c, sampler=PartGroupedSampler, shuffle=False")

    # 读取 debug 配置（保留：用于 train_one_epoch 的行为分支打印）
    dbg = cfg.get('debug', {})
    head_only_dbg = bool(dbg.get('head_only', False))
    _unused_head_lr = float(dbg.get('head_lr', 1e-2))  # 兼容旧配置，不再在此处生效

    # --- 新增：读取训练日程 ---
    sched = cfg.get('schedule', {})
    warmup_epochs     = int(sched.get('warmup_epochs', int(cfg['train'].get('epochs', 3))))
    probe_head_epochs = int(sched.get('probe_head_epochs', 0))
    probe_head_lr     = float(sched.get('probe_head_lr', 1e-2))

    # --- 新增：小样本过拟合自检开关 ---
    limit_n = int(dbg.get('limit_train_samples', 0))  # 0 表示不用

    set_seed(int(cfg.get('seed', 20251003)))
    task = str(cfg.get('task','classification')).lower()
    dcfg = cfg['data']
    dataset_dir = dcfg['dataset_dir']
    bs = int(cfg['train'].get('batch_size', 64))
    num_workers = int(cfg['train'].get('num_workers', 4))
    pin_memory = bool(cfg['train'].get('pin_memory', True))
    prefetch_factor = int(cfg['train'].get('prefetch_factor', 2)) if num_workers>0 else None
    persistent_workers = bool(cfg['train'].get('persistent_workers', True)) and num_workers>0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True

    print(f"[Data] Building datasets from {dataset_dir} ...")
    train_ds = ShardedUTRDataset(dataset_dir, 'train')
    val_ds   = ShardedUTRDataset(dataset_dir, 'val')
    print(f"[Data] train_len={len(train_ds)} val_len={len(val_ds)}")

    # （检查）打印当前 split 的 (part_id, local_idx) 前 8 个，核对是否对齐
    try:
        _p = getattr(train_ds, "_idx_part", None)
        _l = getattr(train_ds, "_idx_local", None)
        if _p is not None and _l is not None:
            pairs = list(zip(_p[:8].tolist(), _l[:8].tolist()))
            print(f"[debug] first8 (part_id, local_idx) in train split: {pairs}")
        else:
            print("[debug] train_ds 缺少 _idx_part/_idx_local，跳过对齐预检打印。")
    except Exception as _e:
        print("[debug] 读取 (part, local) 预检失败：", repr(_e))

    # 用一个样本推断通道数（4 或 5）
    n_channels = infer_n_channels(train_ds[0])
    print(f"[Data] inferred input channels = {n_channels}")

    # 训练集使用采样器：默认分片感知；当 limit_n>0 时改用 OverfitSubsetSampler
    if limit_n > 0:
        print(f"[debug] OverfitSubsetSampler enabled: n={limit_n}")
        sampler = OverfitSubsetSampler(train_ds, n_samples=limit_n, seed=int(cfg.get('seed', 0)))
    else:
        sampler = PartGroupedSampler(
            train_ds,
            block_size=int(cfg['train'].get('block_shuffle', 4096)),
            seed=int(cfg.get('seed', 0))
        )

    train_loader_kwargs = dict(
        batch_size=bs,
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn
    )
    if persistent_workers:
        train_loader_kwargs['persistent_workers'] = True
    if prefetch_factor is not None:
        train_loader_kwargs['prefetch_factor'] = prefetch_factor

    # 验证集：保持顺序，不使用训练 sampler
    val_loader_kwargs = dict(
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn
    )
    if persistent_workers:
        val_loader_kwargs['persistent_workers'] = True
    if prefetch_factor is not None:
        val_loader_kwargs['prefetch_factor'] = prefetch_factor

    train_loader = DataLoader(train_ds, **train_loader_kwargs)
    val_loader   = DataLoader(val_ds, **val_loader_kwargs)
    print(f"[Data] DataLoader ready (num_workers={num_workers}, prefetch_factor={prefetch_factor}, pin_memory={pin_memory})")

    model = build_model(n_channels, cfg)
    model.to(device)

    # --- 构造优化器：改为“BN/bias 不做权重衰减”的参数分组 ---
    optim_cfg = cfg.get('optim', {'optimizer':'adamw','lr':1e-3,'weight_decay':1e-2})
    optimizer = build_optimizer_with_bn_exclusion(model, optim_cfg)

    criterion = _build_criterion(cfg, device=device)

    amp_enabled = bool(cfg['train'].get('amp', False)) and (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    out_dir = cfg['logging']['out_dir']
    ckpt_dir = cfg['logging']['checkpoint_dir']
    ensure_dir(out_dir); ensure_dir(ckpt_dir)

    best_path = os.path.join(ckpt_dir, 'best.pt')
    # choose metric by task
    if task.startswith('reg'):
        best_metric = float('inf')  # lower is better (RMSE)
    else:
        best_metric = -1.0
    num_classes = int(cfg['data'].get('num_targets', cfg['data'].get('num_classes', 54)))

    # === 阶段A：端到端训练 warmup_epochs 轮 ===
    for ep in range(1, warmup_epochs+1):
        if hasattr(train_loader.sampler, "set_epoch"):  # 你已有
            train_loader.sampler.set_epoch(ep)
        tr = train_one_epoch(model, train_loader, optimizer, device, criterion,
                             task, num_classes, scaler, amp_enabled, head_only=False)
        va = evaluate(model, val_loader, device, criterion, task)
        if task.startswith('reg'):
            metric = va['rmse']
        else:
            metric = va['acc']
        print(f"[Epoch {ep:03d}] train_loss={tr['loss']:.4f} " + (f"train_acc={tr.get('acc', float('nan')):.4f} " if not task.startswith('reg') else f"train_rmse={tr.get('rmse', float('nan')):.4f} ") + f"val_loss={va['loss']:.4f} " + (f"val_acc={va.get('acc', float('nan')):.4f}" if not task.startswith('reg') else f"val_rmse={va.get('rmse', float('nan')):.4f} r2={va.get('r2', float('nan')):.3f}"))
        if (task.startswith('reg') and metric < best_metric) or (not task.startswith('reg') and metric > best_metric):
            best_metric = metric
            torch.save({'model': model.state_dict(), 'cfg': cfg, 'n_channels': n_channels}, best_path)

    # === 阶段B（可选）：做线性探针（仅分类） ===
    if (not task.startswith('reg')) and probe_head_epochs > 0:
        print("[probe] freeze trunk + reset head，head-only 线性可分性探针开始")
        # 冻结 trunk、重置 head
        if hasattr(model, "freeze_trunk"): model.freeze_trunk(True)
        if hasattr(model, "reset_head"):   model.reset_head()
        # 关闭 head 内 dropout 干扰
        for m in model.head.modules():
            if isinstance(m, nn.Dropout): m.p = 0.0
        # 只训 head 的优化器（保持原逻辑，不改动）
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=probe_head_lr, weight_decay=0.0)
        for ep in range(1, probe_head_epochs+1):
            # 注意：这里把 head_only=True 传给 train_one_epoch
            tr = train_one_epoch(model, train_loader, optimizer, device, criterion,
                                 task, num_classes, scaler, amp_enabled, head_only=True)
            va = evaluate(model, val_loader, device, criterion, task)
            print(f"[Probe {ep:02d}] train_loss={tr['loss']:.4f} train_acc={tr['acc']:.4f} "
                  f"val_loss={va['loss']:.4f} val_acc={va['acc']:.4f}")

    # 训练结束后照常保存 last
    torch.save({'model': model.state_dict(), 'cfg': cfg, 'n_channels': n_channels},
               os.path.join(ckpt_dir, 'last.pt'))
    return best_path
