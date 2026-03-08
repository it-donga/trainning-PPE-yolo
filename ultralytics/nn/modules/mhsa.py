import torch
import torch.nn as nn

class MHSA(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()

        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):

        B, C, H, W = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(B, self.heads, C // self.heads, H * W)
        k = k.reshape(B, self.heads, C // self.heads, H * W)
        v = v.reshape(B, self.heads, C // self.heads, H * W)

        attn = torch.softmax((q.transpose(-2, -1) @ k) * self.scale, dim=-1)

        out = (attn @ v.transpose(-2, -1)).transpose(-2, -1)

        out = out.reshape(B, C, H, W)

        return self.proj(out)