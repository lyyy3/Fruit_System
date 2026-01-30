"""
增强注意力模块
包含：ECA、CBAM空间注意力、组合注意力
"""
import torch
import torch.nn as nn
import math

from ultralytics.nn.modules import Conv
from ultralytics.nn.modules.block import C2PSA, C3k2


class ECA(nn.Module):
    """ECA-Net: Efficient Channel Attention"""
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """CBAM的空间注意力部分"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 通道维度的max和avg
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        y = torch.cat([max_out, avg_out], dim=1)
        y = self.sigmoid(self.conv(y))
        return x * y


class ECA_SA(nn.Module):
    """ECA + 空间注意力的组合"""
    def __init__(self, channels, c2=None):
        super().__init__()
        self.eca = ECA(channels)
        self.sa = SpatialAttention()
    
    def forward(self, x):
        x = self.eca(x)  # 先通道注意力
        x = self.sa(x)   # 再空间注意力
        return x


class C2PSA_ECA(C2PSA):
    """C2PSA + ECA + 残差"""
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        self.eca = ECA(c1)
    
    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        out = self.cv2(torch.cat((a, b), 1))
        return x + self.eca(out)  # 残差连接


class C3k2_ECA(C3k2):
    """C3k2 + ECA + 残差 (用于Neck多位置增强)"""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.eca = ECA(c2)
        self.use_residual = (c1 == c2)  # 只有输入输出通道相同时才能残差
    
    def forward(self, x):
        # 原始C3k2处理
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        # ECA + 条件残差
        out = self.eca(out)
        return x + out if self.use_residual else out


__all__ = ['ECA', 'SpatialAttention', 'ECA_SA', 'C2PSA_ECA', 'C3k2_ECA']