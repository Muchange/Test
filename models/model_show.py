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

        # self.Low = Low_catvit(feat_size[4],num_channels[4],num_heads,drop)
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
        Re = []
        x1 = self.inc(x)
        Re.append(x1.squeeze(0))
        x2 = self.down1(x1)
        Re.append(x2.squeeze(0))
        x3 = self.down2(x2)
        Re.append(x3.squeeze(0))
        x4 = self.down3(x3)
        Re.append(x4.squeeze(0))
        x5 = self.down4(x4)
        Re.append(x5.squeeze(0))
        #print(x1.shape,x2.shape,x3.shape,x4.shape,x5.shape)

        cgb1 = self.CGB1(x2, x1)
        Re.append(cgb1.squeeze(0))
        cgb2 = self.CGB2(x3, x2)
        Re.append(cgb2.squeeze(0))
        cgb3 = self.CGB3(x4, x3)
        Re.append(cgb3.squeeze(0))
        cgb4 = self.CGB4(x5, x4)
        Re.append(cgb4.squeeze(0))
        #print(cgb1.shape,cgb2.shape,cgb3.shape,cgb4.shape)

        low = self.Low(x5,cgb4)
        Re.append(low.squeeze(0))
        #print('low',low.shape)
        # Re.append(x)
        #print(cgb3.shape,cgb4.shape)
        x = self.up1(low, cgb3)
        Re.append(x.squeeze(0))
        x = self.up2(x, cgb2)
        Re.append(x.squeeze(0))
        x = self.up3(x, cgb1)
        Re.append(x.squeeze(0))
        x = self.up4(x, x1)
        Re.append(x.squeeze(0))
        logits = self.outc(x)
        Re.append(logits.squeeze(0))
        out = nn.Sigmoid()(logits)
        Re.append(out.squeeze(0))

        print(len(Re))
        plt.figure()
        for i in range(14):
            image = Re[i].cpu().detach().numpy()
            print(image.shape)
            x = image[1,:,:]
            plt.subplot(4,4,i+1)
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(x,cmap='gray')
        plt.figure()
        for i in range(2):
            image = Re[i+14].cpu().detach().numpy()
            x = image[0,:,:]
            plt.subplot(1,2,i+1)
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(x,cmap='gray')
        plt.show()
        return out
if __name__ == '__main__':
    net = MCM_Hit_mor(n_channels = 1, n_classes = 1,grid_size=(16,16))
    print(net)
