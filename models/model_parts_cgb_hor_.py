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
from functools import partial
# from .cgb_down import *
# from .hornet import *
from hornet import Block,gnconv,HorNet,GlobalLocalFilter,LayerNorm
from cgb_down import CGB_

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
        out = self.double_conv(x)
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


class Conv_res(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
        self.bn1 =  nn.BatchNorm2d(out_channels),
        self.relu1 = nn.ReLU(inplace=True),

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        self.bn2 = nn.BatchNorm2d(out_channels),
        self.relu2 = nn.ReLU(inplace=True),

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1),
        self.bn3 = nn.BatchNorm2d(out_channels),

        self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        self.shortcut_bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        out = self.bn3(x)

        residual = self.shortcut_conv(residual)
        residual = self.shortcut_bn(residual)
        out += residual
        out = self.relu(out)
        return out

class Hornet_Block(nn.Module):
    def __init__(self, in_channels, Hornet_depths,order,gnconv_h,gnconv_w,
                 drop_path_rate= 0.,layer_scale_init_value = 1e-6,gflayer = False):
        super().__init__()
        s = 1.0 / 3.0
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum([Hornet_depths]))]
        cur = 0
        # print(in_channels)
        self.stage = nn.Sequential(
            *[Block(dim=in_channels, drop_path=dp_rates[cur+j],order = order,s = s,h = gnconv_h,w =gnconv_w,
                    layer_scale_init_value=layer_scale_init_value,gflayer = False) for j in range(Hornet_depths)]
        )
        cur += Hornet_depths
    def forward(self, x):
        out = self.stage(x)
        return out

class inConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_res = Conv_res(in_channels,out_channels)
    def forward(self, x):
        out = self.conv_res(x)
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
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # print('------after',x2.shape,x1.shape)
        out = torch.cat([x2, x1], dim=1)
        # print('------after',out.shape)
        out = self.upconv(out)
        return out



class Hornet_down(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels = 128, out_channels = 256,Hornet_depths=[18],
                 order=4, gnconv_h=14, gnconv_w=8,gflayer = False
                 ):
        super().__init__()
        # print('--------------')
        self.downsample_layers = nn.Sequential(
                LayerNorm(in_channels, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
            )


        self.down_hornet = Hornet_Block(in_channels = out_channels, Hornet_depths=Hornet_depths,
                     order=order, gnconv_h=gnconv_h, gnconv_w=gnconv_w,
                     drop_path_rate=0., layer_scale_init_value=1e-6,gflayer = False)

        self.down_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.downsample_layers(x)
        out = self.down_hornet(x)
        residue = out
        out = self.down_conv(out)
        out += residue
        return out


class Hornet_up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, up_in_channels,Hornet_depths,
                 order, gnconv_h, gnconv_w,gflayer = False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.upsample_layers = nn.Sequential(
            LayerNorm(up_in_channels, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(up_in_channels, out_channels, kernel_size=1),
        )

        self.up_hornet = Hornet_Block(in_channels=out_channels, Hornet_depths=Hornet_depths,
                                   order=order, gnconv_h=gnconv_h, gnconv_w=gnconv_w,
                                   drop_path_rate=0., layer_scale_init_value=1e-6,gflayer = False)
        self.up_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
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
        out = self.upsample_layers(out)
        out = self.up_hornet(out)
        residue = out
        out = self.up_conv(out)
        out += residue
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





class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

# I'll finish my work soon

