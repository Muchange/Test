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
from cgb_down import CGB_
from cross_conv import CGB_conv
from davit import *

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, in_chans=3, embed_dim=768):
        super().__init__()

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=1,stride = 1, padding = 0)

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

class Down_sample_conv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_sample_conv = nn.Sequential(
        #    nn.MaxPool2d(2),
           nn.Conv2d(in_channels, out_channels, kernel_size=2,stride=2, padding=0, bias=True)
        )

    def forward(self, x):
        return self.down_sample_conv(x)

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
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            )
        self.shortcut = nn.Sequential(
        nn.Conv2d(in_channels,out_channels, 1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True),
        )
    def forward(self, x):
        residual = x
        out =  self.double_conv(x)
        residual = self.shortcut(residual)
        out = out + residual
        return out

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

class inConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.double_conv = Conv_residual(out_channels,out_channels)
    def forward(self, x):
        out = self.conv(x)
        out =  self.double_conv(out)
        return out

class CGB(nn.Module):
    def __init__(self, first_num_channels,num_channels, drop_path, grid_size):
        super().__init__()
        self.cgb = CGB_(first_num_channels,num_channels, drop_path, grid_size)

    def forward(self, x,y):
        return self.cgb(x,y)


class en_conv(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # print('--------------')
        self.en =nn.Sequential(
        nn.MaxPool2d(2),
        Conv_residual(in_channels,out_channels)
        ) 
    def forward(self, x):
        return self.en(x)


class de_conv(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, up_in_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # self.conv_ch = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.upconv = Conv_residual(up_in_channels,out_channels)
        # print(in_channels, out_channels, up_in_channels)
    def forward(self, x1, x2):
        # print('---befor',x2.shape,x1.shape)
        x1 = self.up(x1)
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
        self.embed_dim = 768
        self.patch = PatchEmbed(dim, self.embed_dim)
        attention_types = ('spatial','channel')
        depth = 1
        self.Block = nn.ModuleList([
            MySequential(*[
                SpatialBlock(
                    dim=self.embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=4,
                    qkv_bias=True,
                    drop_path=drop_path,
                    norm_layer=nn.LayerNorm,
                    ffn=True,
                    cpe_act=False,
                    window_size=window_size,
                )if attention_type == 'spatial' else
                ChannelBlock(
                    dim=self.embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=4,
                    qkv_bias=True,
                    drop_path=drop_path,
                    norm_layer=nn.LayerNorm,
                    ffn=True,
                    cpe_act=False
                ) if attention_type =='channel' else None
                for index,attention_type in enumerate(attention_types)])
                for i in range(depth)
            ])
        self.da_norm = nn.LayerNorm(self.embed_dim)
        self.da_conv = nn.Conv2d(self.embed_dim,dim, kernel_size=3,padding=1,bias = True)
    def forward(self,x):
        #input:B,C,H,W
        B,C,H,W = x.shape
        x = self.patch(x)
        for blk in self.Block:
            x,_ = blk(x,(H,W))
        out = self.da_norm(x)
        out = out.transpose(-1,-2).reshape(B,self.embed_dim,H,W)
        out = self.da_conv(out)
        print('davit',out.shape)
        return out


class en_davit(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels,num_heads,window_size,drop_path):
        super().__init__()
        # print('--------------')
        self.en =nn.Sequential(
        nn.MaxPool2d(2),
        low_davit(dim = in_channels, num_heads = num_heads,window_size = window_size,drop_path =drop_path),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,bias=True)
        )
    def forward(self, x):
        return self.en(x)



class low_catdavit(nn.Module):
    def __init__(self,in_channels,num_heads,window_size,drop):
        super().__init__()
        # print(num_heads)
        self.davit = low_davit(dim=in_channels, num_heads = num_heads,window_size= window_size,drop_path = drop)
        self.conv = nn.Conv2d(in_channels,in_channels, kernel_size=3,padding=1,bias = True)
    def forward(self, x1, x2):
        x = self.davit(x1)
        out = x + x2
        out = self.conv(out)
        return out

class low_catconv(nn.Module):
    def __init__(self, first_num_channels, num_channels):
        super().__init__()
        self.cgb = CGB_conv(first_num_channels, num_channels)

    def forward(self, x, y):
        return self.cgb(x, y)

class low_catskip(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = Down_sample_conv(in_channels,out_channels)
        self.conv = nn.Conv2d(out_channels*2,out_channels, kernel_size=3,padding=1,bias = True)

    def forward(self, x1, x2):
        x2 = self.down(x2)
        out = torch.cat([x1, x2], dim=1)
        out = self.conv(out)
        return out


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

# I'll finish my work soon

