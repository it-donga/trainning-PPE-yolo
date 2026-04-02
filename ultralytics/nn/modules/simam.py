import torch
import torch.nn as nn


class SimAM(nn.Module):
    """
    SimAM: A Simple Parameter-Free Attention Module
    https://arxiv.org/abs/2103.06252
    """

    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        # mean
        x_mean = x.mean(dim=[2, 3], keepdim=True)

        # variance
        d = (x - x_mean).pow(2)

        # attention energy
        v = d.sum(dim=[2, 3], keepdim=True) / n

        # importance factor
        e_inv = d / (4 * (v + self.e_lambda)) + 0.5

        return x * torch.sigmoid(e_inv)