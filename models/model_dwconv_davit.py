from email.mime import image
from turtle import xcor
import torch
import torch.nn as nn
from model_parts_dwconv_davit import *
from matplotlib import pyplot as plt
from fvcore.nn import FlopCountAnalysis, parameter_count_table, jit_handles
from typing import Any, Callable, List, Optional, Union
from numbers import Number
from numpy import prod
import numpy as np
from thop import profile


class MCM_Hit_mor(nn.Module):
    def __init__(self, n_channels=1, n_classes=9, window_size=(8,8)):
        super(MCM_Hit_mor, self).__init__()
        # feat_size = [256,128,64,32,16]
        num_channels = [32,64,128,256,512]
        up_num_channels = [64,128,256,512]  #384:256*2,256:128*2,128:64*2,64:32*
        embed_dim = [384,768]
        num_heads = [8,12]
        depth = 1
        drop_path_rate = 0.3
        drop = [x.item() for x in torch.linspace(0, drop_path_rate, 2)]
        self.inc = inConv(in_channels = n_channels, out_channels = num_channels[0])            #B,32,256,256
        self.down1 = en_conv(in_channels = num_channels[0], out_channels = num_channels[1])    #B,64,128,128
        self.down2 = en_conv(in_channels = num_channels[1], out_channels = num_channels[2])    #B,128,64,64
        self.down3 = en_conv(in_channels = num_channels[2], out_channels = num_channels[3])    #B,192,32,32
        self.down4 = en_conv(in_channels = num_channels[3], out_channels = num_channels[4])    #B,256,16,16


        self.skip1 = merge_dwconv(first_num_channels = num_channels[0],num_channels = num_channels[1]) #256 + 128，256：down
        self.skip2 = merge_dwconv(first_num_channels = num_channels[1],num_channels = num_channels[2])  #128 + 64，64：upsample
        self.skip3 = merge_davit(first_num_channels = num_channels[2],num_channels = num_channels[3],
                                 embed_dim = embed_dim[0],num_heads = num_heads[0],window_size = window_size,depth = depth,drop = drop[0])
        self.skip4 = merge_davit(first_num_channels = num_channels[3],num_channels = num_channels[4],
                                 embed_dim = embed_dim[1],num_heads = num_heads[1],window_size = window_size,depth= depth,drop = drop[1])

        self.up1 = de_conv(in_channels = num_channels[4], out_channels = num_channels[3],
                            up_in_channels = up_num_channels[3])            #B,256,16,16
        self.up2 = de_conv(in_channels = num_channels[3], out_channels = num_channels[2],
                            up_in_channels = up_num_channels[2])             #B,192,32,32
        self.up3 = de_conv(in_channels = num_channels[2], out_channels = num_channels[1],
                            up_in_channels = up_num_channels[1])             #B,128,64,64
        self.up4 = de_conv(in_channels = num_channels[1], out_channels = num_channels[0],
                            up_in_channels = up_num_channels[0])             #B,64,128,128

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

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # print(x1.shape,x2.shape,x3.shape,x4.shape,x5.shape)

        sk1 = self.skip1(x2, x1)
        sk2 = self.skip2(x3, x2)
        sk3 = self.skip3(x4, x3)
        sk4 = self.skip4(x5, x4)
        
        # print(sk1.shape,sk2.shape,sk3.shape,sk4.shape)

        x = self.up1(sk4, sk3)
        x = self.up2(x, sk2)
        x = self.up3(x, sk1)
        x = self.up4(x, x1)
        logits = self.outc(x)
        # out = nn.Sigmoid()(logits)
        out = logits
        return out
if __name__ == '__main__':
    x = torch.randn((1, 1, 256, 256))
    model = MCM_Hit_mor(n_channels=1, n_classes=9, window_size=(8,8))
    flops, params = profile(model, inputs=(x,))

    params =sum(p.numel() for p in model.parameters() if p.requires_grad)

    fca1 = FlopCountAnalysis(model, x)
    # # handlers = {
    # #     'aten::fft_rfft2': rfft_flop_jit,
    # #     'aten::fft_irfft2': rfft_flop_jit,
    # #     'aten::max_pool2d':flop_maxpool,
    # # }
    # # fca1.set_op_handle(**handlers)
    # # print(parameter_count_table(model))

    print("FLOPs: {} G".format(fca1.total()/ 1e9))
    print('flops:{} G'.format(flops/ 1e9))
    print('Params: {} M'.format(params / 1e6))
