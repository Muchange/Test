from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import torch.fft
from hornet import Block,gnconv,HorNet,GlobalLocalFilter
from thop import profile


if __name__ == '__main__':
    s = 1.0 / 3.0
    block = Block
    in_chan = 128
    gnconv = [
        partial(gnconv, order=4, s=s, h=14, w=8, gflayer=GlobalLocalFilter),
        partial(gnconv, order=5, s=s, h=7, w=4, gflayer=GlobalLocalFilter),
    ]
    model = HorNet(in_chans=in_chan, num_classes=1000, depths=[18, 2], base_dim=256, block=block,gnconv = gnconv)
    x = torch.randn((8, 128, 32, 32))
    flops, params = profile(model, (x,))
    print('Total_params: {} M'.format(params / 1000000.0))
    print('Total_flops: {} M'.format(params / 1000000.0))