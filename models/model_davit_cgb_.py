from email.mime import image
from turtle import xcor
import torch
import torch.nn as nn
from .model_parts_davit_cgb import *
from matplotlib import pyplot as plt
from fvcore.nn import FlopCountAnalysis, parameter_count_table, jit_handles
from typing import Any, Callable, List, Optional, Union
from numbers import Number
from numpy import prod
import numpy as np


class MCM_Hit_mor(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, window_size=(8,8)):
        super(MCM_Hit_mor, self).__init__()
        feat_size = [256,128,64,32,16]
        num_channels = [32,64,128,256,512]
        up_num_channels = [64,128,256,512]  #384:256*2,256:128*2,128:64*2,64:32*2
        num_heads = 12
        embed_dim = 768
        grid_size = window_size
        dt_win_size = (int(window_size[0]/2),int(window_size[0]/2))
        # print(grid_size,dt_win_size)
        #segment_dim = 32
        drop_path_rate = 0.3
        drop = [x.item() for x in torch.linspace(0, drop_path_rate, 5)]
        self.inc = inConv(in_channels = n_channels, out_channels = num_channels[0])            #B,32,256,256
        self.down1 = en_conv(in_channels = num_channels[0], out_channels = num_channels[1])    #B,64,128,128
        self.down2 = en_conv(in_channels = num_channels[1], out_channels = num_channels[2])    #B,128,64,64
        self.down3 = en_conv(in_channels = num_channels[2], out_channels = num_channels[3])    #B,192,32,32
        self.down4 = en_conv(in_channels = num_channels[3], out_channels = num_channels[4])    #B,256,16,16
        # self.down5 = en_conv(in_channels = num_channels[4], out_channels = num_channels[5])    #B,256,16,16

        
        # self.cat1 = cat_skip(num_channels[0],num_channels[1]) #256 + 128，256：down
        # self.cat2 = cat_skip(num_channels[1],num_channels[2])  #128 + 64，64：upsample
        # self.cat3 = cat_skip(num_channels[2],num_channels[3])  #64 + 32，32：upsample
        # self.cat4 = cat_skip(num_channels[3],num_channels[4])  #32 + 16，16：upsample
        
        self.CGB1 = CGB(first_num_channels = num_channels[0],num_channels = num_channels[1], 
                        grid_size = grid_size,drop_path = drop[0]) #256 + 128，256：down
        self.CGB2 = CGB(first_num_channels = num_channels[1],num_channels = num_channels[2], 
                        grid_size = grid_size,drop_path = drop[1])  #128 + 64，64：upsample
        self.CGB3 = CGB(first_num_channels = num_channels[2],num_channels = num_channels[3], 
                        grid_size = grid_size,drop_path = drop[2])  #64 + 32，32：upsample
        self.CGB4 = CGB(first_num_channels = num_channels[3],num_channels = num_channels[4], 
                        grid_size = grid_size,drop_path = drop[3])  #32 + 16，16：upsample
        
        # self.Low = low_catdavit(in_channels = num_channels[4], num_heads = num_heads,
        #                         embed_dim = embed_dim,window_size = grid_size,drop = drop[4])
        # self.Low = low_davit(dim = num_channels[4], num_heads = num_heads,
        #                         embed_dim = embed_dim,window_size = grid_size,drop_path = drop[4])
        # self.Low1 = low_merge_davit(first_num_channels = num_channels[4],num_channels = num_channels[5],
        #                             num_heads = num_heads[1], embed_dim = embed_dim[1],
        #                             window_size = dt_win_size, drop = drop[4])
        # self.Low2 = low_merge_davit(first_num_channels = num_channels[3],num_channels = num_channels[4],
        #                             num_heads = num_heads[0], embed_dim = embed_dim[0],
        #                             window_size = grid_size, drop = drop[3])

        # self.up1 = de_conv(in_channels = num_channels[5], out_channels = num_channels[4],
        #                     up_in_channels = up_num_channels[4])            #B,256,16,16

        # self.de = decoder(in_channels = num_channels[4])            #B,256,16,16

        self.up1 = de_conv(in_channels = num_channels[4], out_channels = num_channels[3],
                            up_in_channels = up_num_channels[3])            #B,256,16,16
        self.up2 = de_conv(in_channels = num_channels[3], out_channels = num_channels[2],
                            up_in_channels = up_num_channels[2])             #B,192,32,32
        self.up3 = de_conv(in_channels = num_channels[2], out_channels = num_channels[1],
                            up_in_channels = up_num_channels[1])             #B,128,64,64
        self.up4 = de_conv(in_channels = num_channels[1], out_channels = num_channels[0],
                            up_in_channels = up_num_channels[0])             #B,64,128,128
        # self.cat = de_cat(out_channels = num_channels[1],up_in_channels = up_num_channels[1])    #B,64,128,128
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
        # x6 = self.down5(x5)

        #print(x1.shape,x2.shape,x3.shape,x4.shape,x5.shape)

        # cat1 = self.cat1(x2, x1)
        # cat2 = self.cat2(x3, x2)
        # cat3 = self.cat3(x4, x3)
        # cat4 = self.cat4(x5, x4)

        cgb1 = self.CGB1(x2, x1)
        cgb2 = self.CGB2(x3, x2)
        cgb3 = self.CGB3(x4, x3)
        cgb4 = self.CGB4(x5, x4)
        
        #print(cgb1.shape,cgb2.shape,cgb3.shape,cgb4.shape)

        low = self.Low(x5)
        # low = self.Low(x5,x4)
        # low1 = self.Low1(x6,x5)
        # low2 = self.Low2(x5,x4)
        #print('low',low.shape)
        #print(cgb3.shape,cgb4.shape)
        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        # x = self.up3(x, x2)

        # x = self.up1(cat4, cat3)
        # x = self.up2(x, cat2)
        # x = self.up3(x, cat1)
        
        # x = self.up1(cgb4, cgb3)
        x = self.de(low, cgb4)
        x = self.up1(low, cgb3)
        x = self.up2(x, cgb2)
        x = self.up3(x, cgb1)
        x = self.up4(x, x1)
        logits = self.outc(x)
        # out = nn.Sigmoid()(logits)
        out = logits
        return out
if __name__ == '__main__':
    x = torch.randn((1, 1, 256, 256))
    model = MCM_Hit_mor(n_channels=1, n_classes=9, window_size=(8,8))
    params =sum(p.numel() for p in model.parameters() if p.requires_grad)

    fca1 = FlopCountAnalysis(model, x)
    # handlers = {
    #     'aten::fft_rfft2': rfft_flop_jit,
    #     'aten::fft_irfft2': rfft_flop_jit,
    #     'aten::max_pool2d':flop_maxpool,
    # }
    # fca1.set_op_handle(**handlers)
    # print(parameter_count_table(model))
    print("FLOPs: {} G".format(fca1.total()/ 1e9))
    print('Params: {} M'.format(params / 1e6))
