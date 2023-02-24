""" Full assembly of the parts to form the complete network """


import torch
import torch.nn as nn
from .model_parts_ import *


class MCM_Hit_(nn.Module):
    def __init__(self, n_channels, n_classes, window_size):
        super(MCM_Hit_, self).__init__()
        drop=0.1
        feat_size = [256,128,64,32,16]
        num_channels = [32,64,128,256,512]
        #num_channels = [32,64,96,128,160]
        up_num_channels = [32,96,160,224,384]
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(num_channels[0], num_channels[1])
        self.down2 = Down(num_channels[1], num_channels[2])
        self.down3 = Down(num_channels[2], num_channels[3])
        self.down4 = Down(num_channels[3], num_channels[4])

        self.CGB1 = CGB(num_channels[0],num_channels[1], drop, window_size,feat_size[1])
        self.CGB2 = CGB(num_channels[1],num_channels[2], drop, window_size,feat_size[2])
        self.CGB3 = CGB(num_channels[2],num_channels[3], drop, window_size,feat_size[3])
        self.CGB4 = CGB(num_channels[3],num_channels[4], drop, window_size,feat_size[4])

        self.up1 = Up(num_channels[4], num_channels[3])
        self.up2 = Up(num_channels[3], num_channels[2])
        self.up3 = Up(num_channels[2], num_channels[1])
        self.up4 = Up(num_channels[1], num_channels[0])
        self.outc = OutConv(num_channels[0], n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        #print(x1.shape,x2.shape,x3.shape,x4.shape,x5.shape)

        cgb1 = self.CGB1(x2, x1)
        cgb2 = self.CGB2(x3, x2)
        cgb3 = self.CGB3(x4, x3)
        cgb4 = self.CGB4(x5, x4)
        #print(cgb1.shape,cgb2.shape,cgb3.shape,cgb4.shape)

        #print(cgb3.shape,cgb4.shape)
        x = self.up1(cgb4, cgb3)
        x = self.up2(x, cgb2)
        x = self.up3(x, cgb1)
        x = self.up4(x, x1)
        logits = self.outc(x)
        out = nn.Sigmoid()(logits)
        return out
if __name__ == '__main__':
    net = MCM_Hit(n_channels = 1, n_classes = 1,grid_size=(16,16))
    print(net)
