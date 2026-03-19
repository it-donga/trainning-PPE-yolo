import torch
import torch.nn as nn
import torch.nn.functional as F

class BiFPN_Add(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-4
        self.w = nn.Parameter(torch.ones(2))

    def forward(self, x):
        x1, x2 = x

        # Align spatial size
        if x1.shape[2:] != x2.shape[2:]:
            x2 = F.interpolate(x2, size=x1.shape[2:], mode='nearest')

        # 🔥 Stable weight
        w = F.softplus(self.w)
        w = w / (w.sum() + self.eps)

        # 🔥 Weighted sum (no Conv)
        return w[0] * x1 + w[1] * x2