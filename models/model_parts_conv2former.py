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
import os
from .conv2former import *
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

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, in_chans, embed_dim):
        super().__init__()
        # print(first_channels)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=2,stride = 2, padding = 0)
        self.norm = LayerNorm(embed_dim, eps=1e-6, data_format="channels_first")
    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

class Conv_residual(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,stride=1, padding=1),
            )
        self.shortcut = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels,out_channels,kernel_size=3, stride=1, padding=1),
        )
    def forward(self, x):
        residual = x
        out =  self.double_conv(x)
        residual = self.shortcut(residual)
        out = out + residual
        return out

class inConv_down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.inconv_down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            # nn.Conv2d(out_channels, out_channels, kernel_size=2,stride=2, padding=0),
            # nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            Conv_residual(out_channels, out_channels)
        )

    def forward(self, x):
        out = self.inconv_down(x)
        return out

class en_down(nn.Module):
    def __init__(self, in_channels, out_channels,depths,drop_rate):
        super().__init__()
        # self.embed_dim = embed_dim
        self.patch = PatchEmbed(in_channels, out_channels)
        self.c2f_block = c2f_Block(dim = out_channels,depths = depths, drop_rate = drop_rate)

    def forward(self, x):
        x = self.patch(x)
        out = self.c2f_block(x)
        return out

class de_up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.upconv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.upconv_residual = Conv_residual(out_channels,out_channels)
        # print(in_channels, out_channels, up_in_channels)
    def forward(self, x1, x2):
        # print('---befor',x1.shape,x2.shape)
        # x1 = self.up(x1)
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        out = torch.cat([x2, x1], dim=1)
        # print('---after', out.shape)
        out = self.upconv(out)
        out = self.upconv_residual(out)
        return out


class merge_c2f_d(nn.Module):
    def __init__(self, in_channels,out_channels,type):
        super().__init__()
        self.type = type
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv2former = ConvMod(out_channels)
    def forward(self, x1, x2):
        if self.type == 'down':
            x1 = nn.MaxPool2d(2)(x1)
        elif self.type == 'up':
            x2 = F.interpolate(x2, size=x1.shape[-2:], mode='bilinear', align_corners=True)
        # assert x1.shape[1] == x2.shape[1]
        merge = torch.cat([x1,x2], dim=1)
        # print(merge.shape)
        merge = self.conv(merge)
        out = self.conv2former(merge)
        return out


class merge_c2f_t(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        # self.down = Down_sample_conv(first_num_channels, num_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv2former = ConvMod(out_channels)
    def forward(self, x1, x2,x3):
        # print(x1.shape,x2.shape)
        x1 = nn.MaxPool2d(2)(x1)
        x3 = F.interpolate(x3, size=x2.shape[-2:], mode='bilinear', align_corners=True)
        # assert x1.shape[1] == x2.shape[1] ,'shape no same'
        merge = torch.cat([x1, x2, x3], dim=1)
        # print(merge.shape)
        merge = self.conv(merge)
        out = self.conv2former(merge)
        return out


class merge_cat(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        # self.down = Down_sample_conv(first_num_channels, num_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    def forward(self, x1, x2):
        # print(x1.shape,x2.shape)
        x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True)
        merge = torch.cat([x1, x2], dim=1)
        # print(merge.shape)
        out = self.conv(merge)
        return out

class up_in(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # self.conv_1 = Conv_residual(in_channels,out_channels)
        # self.conv_2 = Conv_residual(out_channels,out_channels)
        # self.upinconv = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.upinconv_residual = Conv_residual(in_channels,in_channels)
    def forward(self, x):
        # print(x.shape)
        # x = self.upinconv(x)
        out = self.upinconv_residual(x)
        return out

class up_out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # self.conv_1 = Conv_residual(in_channels,out_channels)
        # self.conv_2 = Conv_residual(out_channels,out_channels)
        self.upconv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.upconv_residual = Conv_residual(out_channels,out_channels)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.upconv(x)
        out = self.upconv_residual(x)
        return out

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)


