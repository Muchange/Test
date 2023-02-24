from email.mime import image
from turtle import xcor
import torch
import torch.nn as nn
from .model_parts_van_davit import *
from matplotlib import pyplot as plt
from fvcore.nn import FlopCountAnalysis, parameter_count_table, jit_handles
from typing import Any, Callable, List, Optional, Union
from numbers import Number
from numpy import prod
import numpy as np
from thop import profile
import math

class MCM_Hit_mor(nn.Module):
    def __init__(self, n_channels=1, n_classes=9, window_size=(8,8)):
        super(MCM_Hit_mor, self).__init__()
        num_heads = 12
        embed_dim = [64, 96, 192, 384, 768]
        drop = 0.3
        drop_path_rate = 0.1
        drop_rate = [x.item() for x in torch.linspace(0, drop_path_rate, 4)]
        # feat_size = [256,128,64,32,16]
        num_channels = [32,64,128,256,512]
        # num_channels = [32, 72, 144, 288, 576]
        up_num_channels = [num_channels[0]+num_channels[1],
                           num_channels[1]+num_channels[2],
                           num_channels[2]+num_channels[3],
                           num_channels[3]+num_channels[4]]  #384:256*2,256:128*2,128:64*2,64:32*
        merge_channels = [num_channels[0]+num_channels[1],
                          num_channels[0]+num_channels[1]+num_channels[2],
                          num_channels[1]+num_channels[2]+num_channels[3],
                          num_channels[2]+num_channels[3]]
        cat_merge_channels = [num_channels[0] + num_channels[1],
                              num_channels[1] + num_channels[2],
                              num_channels[2] + num_channels[3],
                              num_channels[3] + num_channels[4]]

        self.inc = Inconv(in_channels = n_channels, out_channels = num_channels[0])            #B,32,256,256
        self.endown1 = encoder(in_channels = num_channels[0], out_channels = num_channels[0])    #B,64,128,128
        self.endown2 = encoder(in_channels = num_channels[0], out_channels = num_channels[1])    #B,128,64,64
        self.endown3 = encoder(in_channels = num_channels[1], out_channels = num_channels[2])    #B,192,32,32
        self.endown4 = encoder(in_channels = num_channels[2], out_channels = num_channels[3])    #B,256,16,16

        self.merge1 = merge_c2f_d(in_channels = merge_channels[0], embed_dims = embed_dim[0],out_channels = num_channels[0],drop_path = drop_rate[0],type= 'up') #256 + 128，256：down
        self.merge2 = merge_c2f_t(in_channels = merge_channels[1], embed_dims = embed_dim[1],out_channels = num_channels[1],drop_path = drop_rate[1])  #128 + 64，64：upsample
        self.merge3 = merge_c2f_t(in_channels = merge_channels[2], embed_dims = embed_dim[2],out_channels = num_channels[2],drop_path = drop_rate[2])
        self.merge4 = merge_c2f_d(in_channels = merge_channels[3], embed_dims = embed_dim[3],out_channels = num_channels[3],drop_path = drop_rate[3],type= 'down')

        self.cat_merg1 = merge_cat(in_channels = cat_merge_channels[0], out_channels = num_channels[0])
        self.cat_merg2 = merge_cat(in_channels = cat_merge_channels[1], out_channels = num_channels[1])
        self.cat_merg3 = merge_cat(in_channels = cat_merge_channels[2], out_channels = num_channels[2])
        self.cat_merg4 = merge_cat(in_channels = cat_merge_channels[3], out_channels=num_channels[3])

        self.low = low_davit(in_channels = num_channels[3], out_channels = num_channels[4],num_heads = num_heads,
                             embed_dim = embed_dim[4],window_size = window_size,drop_path = drop)    #B,256,16,16

        self.up1 = de_up(in_channels = up_num_channels[0], out_channels = num_channels[0])            #B,256,16,16
        self.up2 = de_up(in_channels = up_num_channels[1], out_channels = num_channels[1])             #B,192,32,32
        self.up3 = de_up(in_channels = up_num_channels[2], out_channels = num_channels[2])             #B,128,64,64
        self.up4 = de_up(in_channels = up_num_channels[3], out_channels = num_channels[3])   #B,64,128,128

        self.outc = OutConv(num_channels[0], n_classes)

        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        # elif isinstance(m, nn.Conv2d):
        #     fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #     fan_out //= m.groups
        #     m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        #     if m.bias is not None:
        #         m.bias.data.zero_()

    def forward(self, x):
        x1 = self.inc(x)
        # print(x1.shape)
        x2 = self.endown1(x1)
        x3 = self.endown2(x2)
        x4 = self.endown3(x3)
        x5 = self.endown4(x4)
        # print(x1.shape,x2.shape,x3.shape,x4.shape,x5.shape)

        merge1 = self.merge1(x2, x3)
        merge2 = self.merge2(x2, x3,x4)
        merge3 = self.merge3(x3, x4,x5)
        merge4 = self.merge4(x4, x5)
        # print(merge1.shape,merge2.shape,merge3.shape,merge4.shape)

        low = self.low(x5)
        # print(low.shape)

        cat_mer4 = self.cat_merg4(merge4,low)
        cat_mer3 = self.cat_merg3(merge3, cat_mer4)
        cat_mer2 = self.cat_merg2(merge2, cat_mer3)
        cat_mer1 = self.cat_merg1(merge1, cat_mer2)
        # print(cat_mer1.shape, cat_mer2.shape, cat_mer3.shape, cat_mer4.shape)

        up4 = self.up4(low, cat_mer4)
        up3 = self.up3(up4, cat_mer3)
        up2 = self.up2(up3, cat_mer2)
        up1 = self.up1(up2, cat_mer1)
        out = self.outc(up1)
        return out
if __name__ == '__main__':
    x = torch.randn((1, 1, 256, 256))
    model = MCM_Hit_mor(n_channels=1, n_classes=9, window_size=(8,8))
    flops, param = profile(model, inputs=(x,))

    params =sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('flops:{} G'.format(flops/ 1e9))
    print('param: {} M'.format(param / 1e6))
    # print(model)

