from audioop import bias
from operator import mod
import xxlimited
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from pkg_resources import ResourceManager
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .maxvit import *
from .cgb import *

class MySequential(nn.Sequential):
    def forward(self,*inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class DoubleConv3x3(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
    def forward(self, x):
        return self.double_conv(x)


class CGB(nn.Module):
    def __init__(self, first_num_channels,num_channels, drop_path, grid_size):
        super().__init__()
        self.cgb = CGB_(first_num_channels,num_channels, drop_path, grid_size)

    def forward(self, x,y):
        return self.cgb(x,y)


class en_MBconv(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, dim_head, window_size, use_attention = True,dropout=0.):
        super().__init__()
        if use_attention:
            self.enmb =nn.Sequential(
            nn.MaxPool2d(2),
            MBConv(
                in_channels,
                out_channels,
                expansion_rate=4,
                shrinkage_rate=0.25,
                dropout=0.
            ),
            attention_block(in_channels, dim_head, window_size, dropout=0.)
            )
        else:
            self.enmb = nn.Sequential(
                nn.MaxPool2d(2),
                MBConv(
                    in_channels,
                    out_channels,
                    expansion_rate=4,
                    shrinkage_rate=0.25,
                    dropout=0.
                )
            )
    def forward(self, x):
        return self.enmb(x)


class de_MBconv(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, up_in_channels,dim_head, window_size, use_attention = True, dropout=0.):
        super().__init__()
        self.use_attention = use_attention
        self.mbconv = MBConv(
            up_in_channels,
            out_channels,
            expansion_rate=4,
            shrinkage_rate=0.25,
            dropout=0.)
        self.at_block = attention_block(in_channels, dim_head, window_size, dropout=0.)
        self.conv_ch = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x1, x2):
        # print('---befor',x2.shape,x1.shape)
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        x1 = self.conv_ch(x1)
        # print('------after',x2.shape,x1.shape)
        out = torch.cat([x2, x1], dim=1)
        # print('------after',out.shape)
        if self.use_attention:
            out = self.mbconv(out)
            out = self.at_block(out)
        else:
            out = self.mbconv(out)
        return out

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, up_in_channels,bilinear=True):
        super().__init__()
        self.conv = DoubleConv3x3(up_in_channels, out_channels)
        self.conv_ch = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv_out = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x1, x2):
        # print('---befor',x2.shape,x1.shape)
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        x1 = self.conv_ch(x1)
        # print('------after',x2.shape,x1.shape)
        out = torch.cat([x2, x1], dim=1)
        # print('------after',out.shape)
        out = self.conv(out)
        return out

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

# I'll finish my work soon

