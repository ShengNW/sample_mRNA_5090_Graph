# --- file: src/eval_suite/explain.py ---
from __future__ import annotations
import torch
import torch.nn.functional as F


def grad_saliency(model, x5, x3, target_idx=None):
    """最简单的梯度显著性：对 max logit 求导，返回两路热力图（与输入同长度）。"""
    model.eval()
    x5 = x5.clone().detach().requires_grad_(True)
    x3 = x3.clone().detach().requires_grad_(True)
    logits = model(x5, x3)
    if target_idx is None:
        target_idx = logits.argmax(dim=1)
    target = logits.gather(1, target_idx.view(-1,1)).sum()
    target.backward()
    s5 = x5.grad.detach().abs().sum(dim=1)  # (B,L5)
    s3 = x3.grad.detach().abs().sum(dim=1)  # (B,L3)
    # 归一化到 [0,1]
    s5 = (s5 - s5.min(dim=1, keepdim=True).values) / (s5.max(dim=1, keepdim=True).values - s5.min(dim=1, keepdim=True).values + 1e-8)
    s3 = (s3 - s3.min(dim=1, keepdim=True).values) / (s3.max(dim=1, keepdim=True).values - s3.min(dim=1, keepdim=True).values + 1e-8)
    return s5, s3
