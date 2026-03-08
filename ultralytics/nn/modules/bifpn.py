import torch
import torch.nn as nn
import torch.nn.functional as F


class BiFPN(nn.Module):
    def __init__(self, c1, c2=256):
        super().__init__()

        c3, c4, c5 = c1

        self.p3_in = nn.Conv2d(c3, c2, 1)
        self.p4_in = nn.Conv2d(c4, c2, 1)
        self.p5_in = nn.Conv2d(c5, c2, 1)

        self.p3_out = nn.Conv2d(c2, c2, 3, padding=1)
        self.p4_out = nn.Conv2d(c2, c2, 3, padding=1)
        self.p5_out = nn.Conv2d(c2, c2, 3, padding=1)

    def forward(self, x):

        # x = [p3,p4,p5]
        if isinstance(x, (list, tuple)):
            p3, p4, p5 = x
        else:
            raise ValueError("BiFPN expects 3 feature maps")

        p3 = self.p3_in(p3)
        p4 = self.p4_in(p4)
        p5 = self.p5_in(p5)

        # top-down
        p4_td = p4 + F.interpolate(p5, scale_factor=2, mode="nearest")
        p3_td = p3 + F.interpolate(p4_td, scale_factor=2, mode="nearest")

        # bottom-up
        p4_out = p4 + p4_td + F.max_pool2d(p3_td, 2)
        p5_out = p5 + F.max_pool2d(p4_out, 2)

        return [self.p3_out(p3_td), self.p4_out(p4_out), self.p5_out(p5_out)]