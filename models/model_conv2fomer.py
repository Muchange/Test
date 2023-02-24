from email.mime import image
from turtle import xcor
import torch
import torch.nn as nn
from .model_parts_conv2former import *
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
        # feat_size = [256,128,64,32,16]
        num_channels = [32,64,128,256,512]
        # num_channels = [32, 72, 144, 288, 576]
        up_num_channels = [num_channels[0]+num_channels[1],
                           num_channels[1]+num_channels[2],
                           num_channels[2]+num_channels[3],
                           num_channels[3]+num_channels[4]]  #384:256*2,256:128*2,128:64*2,64:32*
        merge_channels = [num_channels[1]+num_channels[2],
                          num_channels[1]+num_channels[2]+num_channels[3],
                          num_channels[2]+num_channels[3]+num_channels[4],
                          num_channels[3]+num_channels[4]]
        depths = [1, 1, 2, 8]
        drop_path_rate = 0.1
        drop_rate = [x.item() for x in torch.linspace(0, drop_path_rate, 4)]
        self.inc = inConv_down(in_channels = n_channels, out_channels = num_channels[0])            #B,32,256,256
        self.down1 = en_down(in_channels = num_channels[0], out_channels = num_channels[1],depths = depths[0],drop_rate = drop_rate[0])    #B,64,128,128
        self.down2 = en_down(in_channels = num_channels[1], out_channels = num_channels[2],depths = depths[1],drop_rate = drop_rate[1])    #B,128,64,64
        self.down3 = en_down(in_channels = num_channels[2], out_channels = num_channels[3],depths = depths[2],drop_rate = drop_rate[2])    #B,192,32,32
        self.down4 = en_down(in_channels = num_channels[3], out_channels = num_channels[4],depths = depths[3],drop_rate = drop_rate[3])    #B,256,16,16


        self.merge1 = merge_c2f_d(in_channels = merge_channels[0], out_channels = num_channels[1],type= 'up') #256 + 128，256：down
        self.merge2 = merge_c2f_t(in_channels = merge_channels[1], out_channels = num_channels[2])  #128 + 64，64：upsample
        self.merge3 = merge_c2f_t(in_channels = merge_channels[2], out_channels = num_channels[3])
        self.merge4 = merge_c2f_d(in_channels = merge_channels[3], out_channels = num_channels[4],type= 'down')

        self.cat_merg1 = merge_cat(in_channels = up_num_channels[1], out_channels = num_channels[1])
        self.cat_merg2 = merge_cat(in_channels = up_num_channels[2], out_channels = num_channels[2])
        self.cat_merg3 = merge_cat(in_channels = up_num_channels[3], out_channels = num_channels[3])
        # self.low = en_conv(in_channels = num_channels[3], out_channels = num_channels[4],drop_path= drop)    #B,256,16,16

        self.up_in = up_in(in_channels = num_channels[4])
        self.up1 = de_up(in_channels = up_num_channels[3], out_channels = num_channels[3])            #B,256,16,16
        self.up2 = de_up(in_channels = up_num_channels[2], out_channels = num_channels[2])             #B,192,32,32
        self.up3 = de_up(in_channels = up_num_channels[1], out_channels = num_channels[1])             #B,128,64,64
        self.up_out = up_out(in_channels = num_channels[1], out_channels = num_channels[0])   #B,64,128,128

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
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        raw = x
        x1 = self.inc(x)
        # print(x1.shape)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # print(x1.shape,x2.shape,x3.shape,x4.shape,x5.shape)

        merge1 = self.merge1(x2, x3)
        merge2 = self.merge2(x2, x3,x4)
        merge3 = self.merge3(x3, x4,x5)
        merge4 = self.merge4(x4, x5)

        cat_mer1 = self.cat_merg1(merge1,merge2)
        cat_mer2 = self.cat_merg2(merge2,merge3)
        cat_mer3 = self.cat_merg3(merge3,merge4)

        # print(merge1.shape,merge2.shape,merge3.shape,merge4.shape)

        x = self.up_in(merge4)
        x = self.up1(x, cat_mer3)
        x = self.up2(x, cat_mer2)
        x = self.up3(x, cat_mer1)
        x = self.up_out(x)
        logits = self.outc(x)
        # out = nn.Sigmoid()(logits)
        out = logits
        return out
if __name__ == '__main__':
    x = torch.randn((1, 1, 256, 256))
    model = MCM_Hit_mor(n_channels=1, n_classes=9, window_size=(8,8))
    flops, param = profile(model, inputs=(x,))

    params =sum(p.numel() for p in model.parameters() if p.requires_grad)

    fca1 = FlopCountAnalysis(model, x)
    # # handlers = {
    # #     'aten::fft_rfft2': rfft_flop_jit,
    # #     'aten::fft_irfft2': rfft_flop_jit,
    # #     'aten::max_pool2d':flop_maxpool,
    # # }
    # # fca1.set_op_handle(**handlers)
    # # print(parameter_count_table(model))

    # print("FLOPs: {} G".format(fca1.total()/ 1e9))
    # print('Params: {} M'.format(params / 1e6))
    print('flops:{} G'.format(flops/ 1e9))
    print('param: {} M'.format(param / 1e6))
    # print(model)

