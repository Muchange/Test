import torch
import torch.nn.functional as F
import math
from torch import nn

class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GCT(nn.Module):

    def __init__(self, num_channels, epsilon=1e-5):
        super(GCT, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon

    def forward(self, x):
        embedding = (x.pow(2).sum((2,3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
        norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
        gate = 1. + torch.tanh(embedding * norm + self.beta)

        return x * gate

class dw_GCT(nn.Module):

    def __init__(self, in_channels,out_channels):
        super(dw_GCT, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,kernel_size=7, padding=3, groups=in_channels),
            LayerNorm(in_channels, eps=1e-6, data_format="channels_first"),
            GCT(in_channels),
            nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x):
        return self.net(x)