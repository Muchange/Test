from email.mime import image
import torch
import torch.nn as nn
from .model_parts import *
from matplotlib import pyplot as plt

class MCM_Hit_mor(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, window_size=(8,8)):
        super(MCM_Hit_mor, self).__init__()
        feat_size = [256,128,64,32,16]
        num_channels = [32,64,128,192,256]
        #num_channels = [32,64,96,128,160]
        up_num_channels = [64,128,256,384]  #384:256*2,256:128*2,128:64*2,64:32*2
        num_heads =8
        # segment_dim = 32
        drop_path_rate = 0.1
        self.inc = DoubleConv(n_channels, 32)
        drop = [x.item() for x in torch.linspace(0, drop_path_rate, 5)]
        self.down1 = Down(num_channels[0], num_channels[1])
        self.down2 = Down(num_channels[1], num_channels[2])
        self.down3 = Down(num_channels[2], num_channels[3])
        self.down4 = Down(num_channels[3], num_channels[4])

        self.CGB1 = CGB(num_channels[0],num_channels[1], drop[0], window_size)
        self.CGB2 = CGB(num_channels[1],num_channels[2], drop[1], window_size)
        self.CGB3 = CGB(num_channels[2],num_channels[3], drop[2], window_size)
        self.CGB4 = CGB(num_channels[3],num_channels[4], drop[3], window_size)

        # self.Low = low_catvit(feat_size[4],num_channels[4],num_heads,drop_path_rate)
        # self.Low = low_catmaxvit(num_channels[4],num_heads, window_size,drop)
        # print(num_heads)
        self.Low = low_catdavit(num_channels[4],num_heads,window_size, drop[4])

        self.up1 = Up(num_channels[4], num_channels[3],up_num_channels[3])
        self.up2 = Up(num_channels[3], num_channels[2],up_num_channels[2])
        self.up3 = Up(num_channels[2], num_channels[1],up_num_channels[1])
        self.up4 = Up(num_channels[1], num_channels[0],up_num_channels[0])
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

        #print(x1.shape,x2.shape,x3.shape,x4.shape,x5.shape)

        cgb1 = self.CGB1(x2, x1)
        cgb2 = self.CGB2(x3, x2)
        cgb3 = self.CGB3(x4, x3)
        cgb4 = self.CGB4(x5, x4)
        #print(cgb1.shape,cgb2.shape,cgb3.shape,cgb4.shape)

        low = self.Low(x5,cgb4)
        #print('low',low.shape)
        #print(cgb3.shape,cgb4.shape)
        x = self.up1(cgb4, cgb3)
        x = self.up2(x, cgb2)
        x = self.up3(x, cgb1)
        x = self.up4(x, x1)
        logits = self.outc(x)
        # out = nn.Sigmoid()(logits)
        out = logits
        return out
if __name__ == '__main__':
    net = MCM_Hit_mor(n_channels = 1, n_classes = 1,grid_size=(16,16))
    print(net)
