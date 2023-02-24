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
from .cgb_down import *
from .davit import *

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, in_chans=3, embed_dim=768):
        super().__init__()

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, padding = 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        #B,N,C
        #print(x.shape)
        return x

# MBConv
class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias=False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1')
        )

    def forward(self, x):
        return x * self.gate(x)

class MBConvResidual(nn.Module):
    def __init__(self, fn, dim_in,dim_out, dropout=0.):
        super().__init__()
        # print(dim_in,dim_out)
        self.fn = fn
        self.dropsample = Dropsample(dropout)
        self.shortcut = nn.Sequential(
        nn.BatchNorm2d(dim_in),
        nn.GELU(),
        nn.Conv2d(dim_in,dim_out, 1)
        )

    def forward(self, x):
        residual = x
        out = self.fn(x)
        # print(out.shape)
        out = self.dropsample(out)
        # print(out.shape)
        residual = self.shortcut(residual)
        # print(out.shape,x.shape)
        out = out + residual
        return out


class Dropsample(nn.Module):
    def __init__(self, prob=0):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        device = x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1), device=device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)


def MBConv(
        dim_in,
        dim_out,
        expansion_rate=4,
        shrinkage_rate=0.25,
        dropout=0.
):
    hidden_dim = int(expansion_rate * dim_out)
    net = nn.Sequential(
        nn.Conv2d(dim_in, hidden_dim, 3, padding=1),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        nn.Conv2d( hidden_dim, hidden_dim, 3, padding=1),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate=shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
        nn.BatchNorm2d(dim_out),
        nn.GELU(),
    )
    net = MBConvResidual(net, dim_in,dim_out,dropout=dropout)
    return net


class MySequential(nn.Sequential):
    def forward(self,*inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class Shortcut(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.shortcut = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.GELU(),
        nn.Conv2d(in_channels,out_channels, 1),
        )
    def forward(self, x):
        residual = self.shortcut(residual)
        return residual

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
        self.shortcut = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.GELU(),
        nn.Conv2d(in_channels,out_channels, 1),
        )
    def forward(self, x):
        residual = x
        out =  self.double_conv(x)
        residual = self.shortcut(residual)
        out = out + residual
        return out


class CGB(nn.Module):
    def __init__(self, first_num_channels,num_channels, drop_path, grid_size):
        super().__init__()
        self.cgb = CGB_(first_num_channels,num_channels, drop_path, grid_size)

    def forward(self, x,y):
        return self.cgb(x,y)


class en_MBconv(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # print('--------------')
        self.enmb =nn.Sequential(
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

    def __init__(self, in_channels, out_channels, up_in_channels):
        super().__init__()
        self.upconv = DoubleConv3x3(up_in_channels,out_channels,)
        self.conv_ch = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # print(in_channels, out_channels, up_in_channels)
    def forward(self, x1, x2):
        # print('---befor',x2.shape,x1.shape)
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        x1 = self.conv_ch(x1)
        # print('------after',x2.shape,x1.shape)
        out = torch.cat([x2, x1], dim=1)
        # print('------after',out.shape)
        out = self.upconv(out)
        return out


class low_davit(nn.Module):
    def __init__(self, dim, num_heads,window_size,drop_path):
        super().__init__()
        # print(num_heads)
        window_size = window_size[0]
        depth = 1
        self.linear = nn.Linear(dim, dim,bias = True)
        self.patch = PatchEmbed(dim, dim)
        self.block_spa = nn.ModuleList([
            SpatialBlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path=drop_path,
            norm_layer=nn.LayerNorm,
            ffn=True,
            cpe_act=False,
            window_size=window_size,
        )for i in range(depth)
        ])
        self.block_chan = nn.ModuleList([
            ChannelBlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path=drop_path,
            norm_layer=nn.LayerNorm,
            ffn=True,
            cpe_act=False
        )for i in range(depth)
        ])
    def forward(self,x):
        #input:B,C,H,W
        B,C,H,W = x.shape
        x = self.patch(x)
        spa = x
        chan = x
        for blk_spa in self.block_spa:
            spa,_ = blk_spa(spa,(H,W))
        for blk_chan in self.block_chan:
            chan,_ = blk_chan(chan,(H,W))
        sc = spa + chan
        #B,N,C
        # print(sc.shape)
        out = self.linear(sc)
        out = out.transpose(-1,-2).reshape(B,C,H,W)
        return out

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

# I'll finish my work soon

