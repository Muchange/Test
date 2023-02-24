from email.mime import image
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
# from model_parts_cgb_hor import inConv,en_conv,Hornet_down,CGB,Hornet_up,de_conv,OutConv
from .model_parts_cgb_hor import *
from matplotlib import pyplot as plt
from thop import profile
from fvcore.nn import FlopCountAnalysis, parameter_count_table, jit_handles
from typing import Any, Callable, List, Optional, Union
from numbers import Number
from numpy import prod
import numpy as np

class MCM_Hit_mor(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, window_size=(8,8)):
        super(MCM_Hit_mor, self).__init__()
        # num_channels = [32,64,128,192,256,512]
        # up_num_channels = [96,192,320,448,768]
        num_channels = [32,64,128,256,512]
        up_num_channels = [64,128,256,512]
        # up_num_channels = [64,128,256,512]  #384:256*2,256:128*2,128:64*2,64:32*2
        grid_size =window_size
        drop_path_rate = 0.3
        drop = [x.item() for x in torch.linspace(0, drop_path_rate, 5)]
        hornet_depths = [1,1,3,2]
        gnconv_hw = [14,8,7,4]
        order = [2,3,4,5]
        gflayer = True

        self.inc = inConv(in_channels = n_channels, out_channels = num_channels[0])            #B,32,256,256
        # self.down1 = en_conv(in_channels = num_channels[0], out_channels = num_channels[1])    #B,64,128,128
        # self.down2 = en_conv(in_channels = num_channels[1], out_channels = num_channels[2])    #B,128,64,64
        # self.down3 = en_conv(in_channels = num_channels[2], out_channels = num_channels[3])    #B,128,64,64
        # self.down4 = en_conv(in_channels=num_channels[3], out_channels=num_channels[4])  # B,128,64,64
        # self.down5 = en_conv(in_channels=num_channels[4], out_channels=num_channels[5])  # B,128,64,64

        self.down1 = Hornet_down(in_channels=num_channels[0], out_channels=num_channels[1],
                                 Hornet_depths=hornet_depths[0],
                                 order=order[0], gnconv_h=gnconv_hw[0], gnconv_w=gnconv_hw[1],
                                 grid_size =grid_size, gflayer=False
                                 )
        self.down2 = Hornet_down(in_channels=num_channels[1], out_channels=num_channels[2],
                                 Hornet_depths=hornet_depths[1],
                                 order=order[1], gnconv_h=gnconv_hw[0], gnconv_w=gnconv_hw[1],
                                 grid_size =grid_size,gflayer=False
                                 )
        self.down3 = Hornet_down(in_channels = num_channels[2], out_channels = num_channels[3],
                                 Hornet_depths=hornet_depths[2],
                                 order=order[2], gnconv_h=gnconv_hw[0], gnconv_w=gnconv_hw[1],
                                 grid_size =grid_size,gflayer=True
                                 )    #B,192,32,32

        self.down4 = Hornet_down(in_channels = num_channels[3], out_channels = num_channels[4],
                                 Hornet_depths=hornet_depths[3],
                                 order=order[3], gnconv_h=gnconv_hw[2], gnconv_w=gnconv_hw[3],
                                 grid_size =grid_size,gflayer=True
                                 )    #B,256,16,16

        self.CGB1 = CGB(num_channels[0],num_channels[1], drop[0], grid_size) #256 + 128，256：down
        self.CGB2 = CGB(num_channels[1],num_channels[2], drop[1], grid_size)  #128 + 64，64：upsample
        self.CGB3 = CGB(num_channels[2],num_channels[3], drop[2], grid_size)  #64 + 32，32：upsample
        self.CGB4 = CGB(num_channels[3],num_channels[4], drop[3], grid_size)  #64 + 32，32：upsample

        #
        # self.up1 = Hornet_up(in_channels = num_channels[4], out_channels = num_channels[3],
        #                         up_in_channels = up_num_channels[3],
        #                         Hornet_depths=hornet_depths[3],
        #                         order=order[3], gnconv_h=gnconv_hw[2], gnconv_w=gnconv_hw[3],gflayer=True
        #                      )             #B,128,64,64
        # self.up2 = Hornet_up(in_channels = num_channels[3], out_channels = num_channels[2],
        #                         up_in_channels = up_num_channels[2],
        #                         Hornet_depths=hornet_depths[2],
        #                         order=order[2], gnconv_h=gnconv_hw[0], gnconv_w=gnconv_hw[1],gflayer=True
        #                      )             #B,64,128,128

        # self.up3 = Hornet_up(in_channels=num_channels[2], out_channels=num_channels[1],
        #                       up_in_channels=up_num_channels[1],
        #                      Hornet_depths=hornet_depths[1],
        #                      order=order[1], gnconv_h=gnconv_hw[2], gnconv_w=gnconv_hw[3],gflayer=False
        #                      )  # B,128,64,64
        # self.up4 = Hornet_up(in_channels=num_channels[1], out_channels=num_channels[0],
        #                      up_in_channels=up_num_channels[0],
        #                      Hornet_depths=hornet_depths[0],
        #                      order=order[0], gnconv_h=gnconv_hw[0], gnconv_w=gnconv_hw[1],gflayer=False
        #                      )  # B,64,128,128
        self.up1 = de_conv(in_channels=num_channels[4], out_channels=num_channels[3],
                            up_in_channels=up_num_channels[3])  # B,256,16,16
        self.up2 = de_conv(in_channels=num_channels[3], out_channels=num_channels[2],
                            up_in_channels=up_num_channels[2])  # B,256,16,16
        self.up3 = de_conv(in_channels=num_channels[2], out_channels=num_channels[1],
                            up_in_channels=up_num_channels[1])  # B,256,16,16
        self.up4 = de_conv(in_channels=num_channels[1], out_channels=num_channels[0],
                            up_in_channels=up_num_channels[0])  # B,192,32,32
        self.outc = OutConv(num_channels[0], n_classes)

        self.uniform_init = False
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if not self.uniform_init:
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        else:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)


    def forward(self, x):
        # data_ = []
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # x6 = self.down5(x5)

        # print(x1.shape,x2.shape,x3.shape,x4.shape,x5.shape,x6.shape)

        cgb1 = self.CGB1(x2, x1)
        cgb2 = self.CGB2(x3, x2)
        cgb3 = self.CGB3(x4, x3)
        cgb4 = self.CGB4(x5, x4)
        # cgb5 = self.CGB5(x6, x5)

        #print(cgb1.shape,cgb2.shape,cgb3.shape,cgb4.shape)

        # low = self.Low(x5,cgb4)
        #print('low',low.shape)
        #print(cgb3.shape,cgb4.shape)
        x = self.up1(cgb4, cgb3)
        x = self.up2(x, cgb2)
        x = self.up3(x, cgb1)
        x = self.up4(x, x1)
        # x = self.up5(x, x1)
        logits = self.outc(x)
        # out = nn.Sigmoid()(logits)
        out = logits

        return out
if __name__ == '__main__':
    crop_size = 256
    # main_path = 'D:/Medical Imaging/2Dunet/LungSegData/02Normalized_subsetUR/1GMQX2WE.npy'
    image = np.load('./test_img.npy')
    image = crop(image, crop_size)
    image = image.transpose(2,1,0)
    image = np.expand_dims(image,0)
    image = torch.from_numpy(image).float()
    print(image.shape)
    model = MCM_Hit_mor(n_channels=1, n_classes=9, window_size=(8,8))
    out,data_list = model(image)

    plt.figure()
    for i in range(len(data_list)):
        x = data_list[i][:,0,:].squeeze(0)
        x = x.detach().cpu().numpy()
        plt.subplot(4, 5, i + 1)
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x, cmap='gray')
    plt.show()



