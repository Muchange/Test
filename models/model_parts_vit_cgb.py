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
from .vit import *

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
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def forward(self, x):
        out =  self.double_conv(x)
        return out


class DoubleConv3x3_residual(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            )
        self.shortcut = nn.Sequential(
        nn.Conv2d(in_channels,out_channels, 1),
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        residual = x
        out =  self.double_conv(x)
        residual = self.shortcut(residual)
        out = out + residual
        out = self.relu(out)
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

class Low_vit(nn.Module):
    def __init__(self, in_feats,dim, num_heads,drop_path,norm_layer=nn.LayerNorm):
        super().__init__()
        self.vit = Vit(dim = dim, num_heads = num_heads,drop_path = drop_path)
        num_patches = in_feats * in_feats
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
        self.pos_drop = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm = norm_layer(dim)
        self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        #input:B,C,H,W
        B, C, H, W= x.shape
        x = x.reshape(B,C,H*W).permute(0,2,1)
        #shape:B,N,C
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.vit(x)
        x = self.norm(x)
        x = self.proj(x)
        x = self.pos_drop(x)
        #shape:B,N,C
        out = x.permute(0,2,1).reshape(B,C,H,W)
        #B,C,H,W
        return out

class low_catvit(nn.Module):
    def __init__(self, feat_size,in_channels,num_heads,drop):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,in_channels, kernel_size=1,bias = True)
        self.maxvit = Low_vit(in_feats = feat_size,dim = in_channels,num_heads = num_heads,drop_path = drop)
    def forward(self, x1, x2):
        x = self.maxvit(x1)
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

