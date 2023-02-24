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
from timm.models.efficientnet_blocks import SqueezeExcite, DepthwiseSeparableConv
import os
# from davit import  * *
# class GRN(nn.Module):
#     """ GRN (Global Response Normalization) layer
#     """
#     def __init__(self, dim):
#         super().__init__()
#         self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
#         self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

#     def forward(self, x):
#         Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
#         Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
#         return self.gamma * (x * Nx) + self.beta + x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim ))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim ))

    def forward(self, x):
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        out = self.gamma * (x * Nx) + self.beta + x
        out = out.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        return out

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

class dw_se(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            LayerNorm(in_channels, eps=1e-6, data_format="channels_first"),
            SqueezeExcite(in_chs=in_channels, rd_ratio=0.25),
            nn.Conv2d(in_channels, in_channels,kernel_size=7, padding=3, groups=in_channels),
            # nn.Conv2d(in_channels, out_channels, 1)
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        # self.net = nn.Sequential(
        #     LayerNorm(in_channels, eps=1e-6, data_format="channels_first"),
        #     SqueezeExcite(in_chs=in_channels, rd_ratio=0.25),
        #     nn.Conv2d(in_channels, in_channels,kernel_size=5, padding=2, groups=in_channels),
        #     nn.Conv2d(in_channels, out_channels,kernel_size=7, padding=3, groups=out_channels),
        #     nn.Conv2d(out_channels, out_channels*4, 1),
        #     nn.GELU(),
        #     nn.Conv2d(out_channels*4, out_channels, 1)
        # )
    #     self.net  = nn.Sequential(
    #         LayerNorm(in_channels, eps=1e-6, data_format="channels_first"),
    #         GRN(in_channels),
    #         nn.Conv2d(in_channels, in_channels,kernel_size=7, padding=3, groups=in_channels),
    #         LayerNorm(in_channels, eps=1e-6, data_format="channels_first"),
    #         nn.Conv2d(in_channels, out_channels*4, 1),
    #         nn.GELU(),
    #         GRN(out_channels*4),
    #         nn.Conv2d(out_channels*4, out_channels, 1),
    #     )
    #     self.shorcut = nn.Sequential(
    #                                 nn.Conv2d(in_channels, out_channels, 1),
    #                                 GRN(out_channels),
    #     )
    def forward(self, x):
       return self.net(x)

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
        # print(x.shape)
        out =  self.double_conv(x)
        residual = self.shortcut(residual)
        out = out + residual
        return out

# class Inconv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.inconv_down = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
#             nn.Conv2d(out_channels, out_channels, kernel_size=2,stride=2, padding=0),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#         )
#
#     def forward(self, x):
#         out = self.inconv_down(x)
#         return out

class Inconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.inconv =  nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            Conv_residual(out_channels, out_channels),

        )
    def forward(self, x):
        # print(x.shape)
        out = self.inconv(x)
        # print(out.shape)
        return out

class encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # self.embed_dim = embed_dim
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            # nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2,padding=0),
            Conv_residual(in_channels, out_channels),
        )

    def forward(self, x):
        out = self.down(x)
        return out

class merge_c2f_d_c(nn.Module):
    def __init__(self, in_channels,out_channels,type):
        super().__init__()
        self.type = type
        if self.type == 'down':
            self.conv_change = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2,padding=0)
            # self.dwconv_se = dw_se(out_channels * 2,out_channels)
        elif self.type == 'up':
            self.conv_change = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            # self.dwconv_se = nn.Conv2d(out_channels * 2,out_channels,kernel_size=3, stride=1,padding=1)
        # self.van = Van_block(in_chans = in_channels, embed_dims = embed_dims,mlp_ratios = mlp_ratios,drop_rate=drop_path)
        # self.dwconv_se = dw_se(out_channels * 2,out_channels)
        self.dwconv_se = nn.Conv2d(out_channels * 2,out_channels,kernel_size=3, stride=1,padding=1)
    def forward(self, x1, x2):
        x2 = self.conv_change(x2)
        merge = torch.cat([x1,x2], dim=1)
        # print(merge.shape)
        # merge = self.conv(merge)
        out = self.dwconv_se(merge)
        return out

class merge_c2f_d_lsc1x1(nn.Module):
    def __init__(self, in_channels,out_channels,type):
        super().__init__()
        self.type = type
        if self.type == 'down':
            self.conv_change = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2,padding=0)
            # self.dwconv_se = dw_se(out_channels * 2,out_channels)
        elif self.type == 'up':
            self.conv_change = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.dwconv_se = nn.Sequential(
            LayerNorm(out_channels * 2, eps=1e-6, data_format="channels_first"),
            SqueezeExcite(in_chs=out_channels * 2, rd_ratio=0.25),
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
        )
    def forward(self, x1, x2):
        x2 = self.conv_change(x2)
        merge = torch.cat([x1,x2], dim=1)
        out = self.dwconv_se(merge)
        return out

class merge_c2f_d_lsc3x3(nn.Module):
    def __init__(self, in_channels,out_channels,type):
        super().__init__()
        self.type = type
        if self.type == 'down':
            self.conv_change = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2,padding=0)
        elif self.type == 'up':
            self.conv_change = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.dwconv_se = nn.Sequential(
            LayerNorm(out_channels * 2, eps=1e-6, data_format="channels_first"),
            SqueezeExcite(in_chs=out_channels * 2, rd_ratio=0.25),
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        )
    def forward(self, x1, x2):
        x2 = self.conv_change(x2)
        merge = torch.cat([x1,x2], dim=1)
        out = self.dwconv_se(merge)
        return out

class merge_c2f_d(nn.Module):
    def __init__(self, in_channels,out_channels,type):
        super().__init__()
        self.type = type
        if self.type == 'down':
            self.conv_change = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2,padding=0)
        elif self.type == 'up':
            self.conv_change = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.dwconv_se = dw_se(out_channels * 2,out_channels)
    def forward(self, x1, x2):
        x2 = self.conv_change(x2)
        merge = torch.cat([x1,x2], dim=1)
        out = self.dwconv_se(merge)
        return out


class merge_c2f_t(nn.Module):
    def __init__(self, down_channels,up_channels,out_channels):
        super().__init__()
        self.conv_down = nn.Conv2d(down_channels, out_channels, kernel_size=2, stride=2,padding=0)
        self.conv_up = nn.ConvTranspose2d(up_channels, out_channels, kernel_size=2, stride=2)
        self.dwconv_se = dw_se(out_channels * 3,out_channels)
    def forward(self, x1, x2,x3):
        # print(x1.shape,x2.shape)
        x1 = self.conv_down(x1)
        x3 = self.conv_up(x3)
        merge = torch.cat([x1, x2, x3], dim=1)
        out = self.dwconv_se(merge)
        return out

class merge_c2f_t_c(nn.Module):
    def __init__(self, down_channels,up_channels,out_channels):
        super().__init__()
        self.conv_down = nn.Conv2d(down_channels, out_channels, kernel_size=2, stride=2,padding=0)
        self.conv_up = nn.ConvTranspose2d(up_channels, out_channels, kernel_size=2, stride=2)
        self.dwconv_se = nn.Conv2d(out_channels * 3,out_channels,kernel_size=3, stride=1,padding=1)
    def forward(self, x1, x2,x3):
        x1 = self.conv_down(x1)
        x3 = self.conv_up(x3)
        merge = torch.cat([x1, x2, x3], dim=1)
        out = self.dwconv_se(merge)
        return out

class merge_c2f_t_lsc3x3(nn.Module):
    def __init__(self, down_channels,up_channels,out_channels):
        super().__init__()
        self.conv_down = nn.Conv2d(down_channels, out_channels, kernel_size=2, stride=2,padding=0)
        self.conv_up = nn.ConvTranspose2d(up_channels, out_channels, kernel_size=2, stride=2)
        self.dwconv_se = nn.Sequential(
            LayerNorm(out_channels * 3, eps=1e-6, data_format="channels_first"),
            SqueezeExcite(in_chs=out_channels * 3, rd_ratio=0.25),
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=3, padding=1)
        )
    def forward(self, x1, x2,x3):
        x1 = self.conv_down(x1)
        x3 = self.conv_up(x3)
        merge = torch.cat([x1, x2, x3], dim=1)
        out = self.dwconv_se(merge)
        return out

class merge_c2f_t_lsc1x1(nn.Module):
    def __init__(self, down_channels,up_channels,out_channels):
        super().__init__()
        self.conv_down = nn.Conv2d(down_channels, out_channels, kernel_size=2, stride=2,padding=0)
        self.conv_up = nn.ConvTranspose2d(up_channels, out_channels, kernel_size=2, stride=2)
        self.dwconv_se = nn.Sequential(
            LayerNorm(out_channels * 3, eps=1e-6, data_format="channels_first"),
            SqueezeExcite(in_chs=out_channels * 3, rd_ratio=0.25),
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)
        )
    def forward(self, x1, x2,x3):
        x1 = self.conv_down(x1)
        x3 = self.conv_up(x3)
        merge = torch.cat([x1, x2, x3], dim=1)
        out = self.dwconv_se(merge)
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

class de_up_low(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels):
        super().__init__()
        self.upconv = Conv_residual(in_channels,in_channels)
    def forward(self, x):
        out = self.upconv(x)
        return out

class de_up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels,up_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # self.up = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.upconv = Conv_residual(up_channels,out_channels)
        # print(in_channels, out_channels, up_in_channels)
    def forward(self, x1, x2):
        # print('---befor',x1.shape,x2.shape)
        x1 = self.up(x1)
        # x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        out = torch.cat([x2, x1], dim=1)
        # print('---after', out.shape)
        out = self.upconv(out)
        return out

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        # self.conv_res = Conv_residual(in_channels, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        x = self.conv(x)
        return x


