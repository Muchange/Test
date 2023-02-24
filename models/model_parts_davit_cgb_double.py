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
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )
    def forward(self, x):
        out =  self.double_conv(x)
        return out


class DoubleConv3x3_residual(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            )
        self.shortcut = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(in_channels,out_channels, 1),
        )
    def forward(self, x):
        residual = x
        out =  self.double_conv(x)
        residual = self.shortcut(residual)
        out = out + residual
        return out


class CGB(nn.Module):
    def __init__(self, first_num_channels,num_channels, drop_path, grid_size,same):
        super().__init__()
        self.cgb = CGB_(first_num_channels,num_channels, drop_path, grid_size,same)

    def forward(self, x,y):
        return self.cgb(x,y)


class en_conv(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # print('--------------')
        self.en =nn.Sequential(
        nn.MaxPool2d(2),
        DoubleConv3x3_residual(in_channels,out_channels)
        ) 
    def forward(self, x):
        return self.en(x)


class de_conv(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, up_in_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # self.conv_ch = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.upconv = DoubleConv3x3_residual(up_in_channels,out_channels)
        # print(in_channels, out_channels, up_in_channels)
    def forward(self, x1, x2):
        # print('---befor',x2.shape,x1.shape)
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # print('------after',x2.shape,x1.shape)
        out = torch.cat([x2, x1], dim=1)
        # print('------after',out.shape)
        out = self.upconv(out)
        return out

class de_cat(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, out_channels,up_in_channels):
        super().__init__()
        self.conv = DoubleConv3x3_residual(up_in_channels,out_channels)
    def forward(self, x1, x2):
        out = torch.cat([x2, x1], dim=1)
        out = self.conv(out)
        return out

class low_davit(nn.Module):
    def __init__(self, dim, num_heads,window_size,drop_path):
        super().__init__()
        window_size = window_size[0]
        self.patch = PatchEmbed(dim, dim)
        depth = 1
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
        self.low_norm = nn.LayerNorm(dim)
        self.low_conv = nn.Conv2d(dim, dim, kernel_size=1)
    def forward(self,x):
        #input:B,C,H,W
        B,C,H,W = x.shape
        x = self.patch(x)
        for blk_spa in self.block_spa:
            spa,_ = blk_spa(spa,(H,W))
        for blk_chan in self.block_chan:
            chan,_ = blk_chan(chan,(H,W))
        sc = spa + chan
        out = self.da_norm(sc)
        out = self.low_conv(out)
        out = out.transpose(-1,-2).reshape(B,C,H,W)
        return out

class low_catdavit(nn.Module):
    def __init__(self,in_channels,num_heads,window_size,drop):
        super().__init__()
        # print(num_heads)
        self.davit = low_davit(dim=in_channels, num_heads = num_heads,window_size= window_size,drop_path = drop)
        self.conv = nn.Conv2d(in_channels,in_channels, kernel_size=1,bias = True)
    def forward(self, x1, x2):
        x = self.davit(x1)
        out = x + x2
        out = self.conv(out)
        return out

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

# I'll finish my work soon

