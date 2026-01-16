import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, ratio=16):
        super().__init__()
        self.ratio = ratio
        self.mlp = None
        self.sigmoid = nn.Sigmoid()

    def _build(self, channels):
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // self.ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // self.ratio, channels, 1, bias=False),
        )

    def forward(self, x):
        if self.mlp is None:
            self._build(x.shape[1])

        avg = torch.mean(x, dim=(2, 3), keepdim=True)
        mx, _ = torch.max(x, dim=2, keepdim=True)
        mx, _ = torch.max(mx, dim=3, keepdim=True)

        return x * self.sigmoid(self.mlp(avg) + self.mlp(mx))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CustomCBAM(nn.Module):
    def __init__(self, channels, ratio=16):
        super().__init__()
        self.ca = ChannelAttention(ratio)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x
