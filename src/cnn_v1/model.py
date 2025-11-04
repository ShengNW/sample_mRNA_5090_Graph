# --- file: src/cnn_v1/model.py ---
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, k=7, drop=0.0):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=1, padding=p)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = self.act(y)
        y = self.drop(y)
        return y + self.proj(x)

class BranchCNN(nn.Module):
    def __init__(self, in_ch: int, emb_dim: int, channels=(64,128,256)):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, emb_dim, kernel_size=1)
        blocks = []
        c = emb_dim
        for c_next in channels:
            blocks.append(ResBlock1D(c, c_next, k=7, drop=0.0))   # 先不加dropout，先学起来
            c = c_next
        self.blocks = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        self.out_channels = c * 2  # mean+max 拼接

    def forward(self, x):
        x = self.proj(x)
        x = self.blocks(x)
        m = self.gap(x).squeeze(-1)
        M = self.gmp(x).squeeze(-1)
        return torch.cat([m, M], dim=-1)

class DualBranchCNN(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, emb_dim: int = 8,
                 channels=(64,128,256), share_branches: bool = False):
        super().__init__()
        if share_branches:
            self.branch = BranchCNN(in_ch, emb_dim, channels)
            self.branch5 = self.branch
            self.branch3 = self.branch
        else:
            self.branch5 = BranchCNN(in_ch, emb_dim, channels)
            self.branch3 = BranchCNN(in_ch, emb_dim, channels)
        fused_in = self.branch5.out_channels + self.branch3.out_channels
        self.head = nn.Sequential(
            nn.Linear(fused_in, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, utr5, utr3):
        f5 = self.branch5(utr5)
        f3 = self.branch3(utr3)
        h = torch.cat([f5, f3], dim=-1)
        logits = self.head(h)
        return logits

    # === 新增：抽特征接口 ===
    @torch.no_grad()
    def forward_features(self, utr5, utr3):
        """只做特征抽取（不经过 head），返回 f5, f3, concat(h)"""
        self.eval()
        f5 = self.branch5(utr5)
        f3 = self.branch3(utr3)
        h = torch.cat([f5, f3], dim=-1)
        return f5, f3, h

    # === 兼容方案 A：冻结/解冻 trunk ===
    def freeze_trunk(self, flag: bool = True):
        """
        冻结/解冻双分支主干（trunk）。
        flag=True: 冻结 trunk（不训练，且 BN/Dropout 切到 eval）
        flag=False: 解冻 trunk（恢复训练模式）
        """
        trunks = [self.branch5, self.branch3]
        # 去重（share_branches=True 时两者相同）
        seen = set()
        uniq_trunks = []
        for m in trunks:
            if id(m) not in seen:
                uniq_trunks.append(m)
                seen.add(id(m))

        for m in uniq_trunks:
            # 参数开关
            for p in m.parameters():
                p.requires_grad = (not flag)
            # 训练/评估模式切换
            if flag:
                m.eval()
                # 如果 trunk 里有归一化层，切 eval，避免动统计量
                for sub in m.modules():
                    if hasattr(sub, 'track_running_stats'):
                        sub.eval()
            else:
                m.train()

    def reset_head(self):
        """重置 head 的参数到初始状态（调用模块自带的 reset_parameters）。"""
        for m in self.head.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

    # === 兼容旧调用：保留无参冻结/解冻接口 ===
    def unfreeze_trunk(self):
        """解冻两支主干（兼容旧接口）。"""
        self.freeze_trunk(False)
