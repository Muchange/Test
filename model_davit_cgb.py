from email.mime import image
import torch
import torch.nn as nn
from model_parts_davit_cgb import inConv,en_conv,en_davit,CGB,de_conv,OutConv,low_catskip,low_catconv
from matplotlib import pyplot as plt

class MCM_Hit_mor(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, window_size=(7,7),grid_size = (2,2)):
        super(MCM_Hit_mor, self).__init__()
        feat_size = [256,128,64,32,16]
        num_channels = [32,64,128,256,512]
        up_num_channels = [64,128,256,512]  #384:256*2,256:128*2,128:64*2,64:32*2
        num_heads = 12
        grid_size =window_size
        #segment_dim = 32
        drop_path_rate = 0.3
        drop = [x.item() for x in torch.linspace(0, drop_path_rate, 5)]
        self.inc = inConv(in_channels = n_channels, out_channels = num_channels[0])            #B,32,256,256
        self.down1 = en_conv(in_channels = num_channels[0], out_channels = num_channels[1])    #B,64,128,128
        self.down2 = en_conv(in_channels = num_channels[1], out_channels = num_channels[2])    #B,128,64,64
        # self.down3 = en_conv(in_channels = num_channels[2], out_channels = num_channels[3])    #B,192,32,32
        # self.down4 = en_conv(in_channels = num_channels[3], out_channels = num_channels[4])    #B,256,16,16
        # self.down1 = en_davit(in_channels=num_channels[0], out_channels=num_channels[1], num_heads=num_heads,
        #                       window_size=window_size, drop_path=drop[0])  # B,192,32,32
        # self.down2 = en_davit(in_channels=num_channels[1], out_channels=num_channels[2], num_heads=num_heads,
        #                       window_size=window_size, drop_path=drop[0])  # B,256,16,16
        self.down3 = en_davit(in_channels = num_channels[2], out_channels = num_channels[3],num_heads = num_heads,
                               window_size = window_size,drop_path = drop[0])  # B,192,32,32
        self.down4 = en_davit(in_channels = num_channels[3], out_channels = num_channels[4],num_heads = num_heads,
                               window_size = window_size,drop_path = drop[0])  # B,256,16,16

        # self.CGB1 = CGB(num_channels[0],num_channels[1], drop[0], grid_size) #256 + 128，256：down
        # self.CGB2 = CGB(num_channels[1],num_channels[2], drop[1], grid_size)  #128 + 64，64：upsample
        self.CGB1 = low_catconv(num_channels[0], num_channels[1])  # 256 + 128，256：down
        self.CGB2 = low_catconv(num_channels[1], num_channels[2])  # 128 + 64，64：upsample
        self.CGB3 = CGB(num_channels[2],num_channels[3], drop[0], grid_size)  #64 + 32，32：upsample
        self.CGB4 = CGB(num_channels[3],num_channels[4], drop[1], grid_size)  #32 + 16，16：upsample
        
        # self.Low = low_catdavit(num_channels[4],num_heads,window_size, drop[4])

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

    #     self.apply(self._init_weights)
    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        print(x1.shape,x2.shape,x3.shape,x4.shape,x5.shape)

        cgb1 = self.CGB1(x2, x1)
        cgb2 = self.CGB2(x3, x2)
        cgb3 = self.CGB3(x4, x3)
        cgb4 = self.CGB4(x5, x4)
        
        #print(cgb1.shape,cgb2.shape,cgb3.shape,cgb4.shape)

        # low = self.Low(x5,cgb4)
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

    in_chan = 1
    image_size = 256
    model = MCM_Hit_mor( n_channels=1, n_classes=9, window_size=(8,8))
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    x = torch.randn((1, in_chan, image_size, image_size))
    from fvcore.nn import FlopCountAnalysis, parameter_count_table, jit_handles
    fca1 = FlopCountAnalysis(model, x)
    print("FLOPs: {} G".format(fca1.total() / 1e9))
    print('Params: {} M'.format(params / 1e6))
