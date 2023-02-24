""" Parts of the U-Net model """

from audioop import bias
import xxlimited
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .davit import *
class Down_sample_conv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_sample_conv = nn.Sequential(
        #    nn.MaxPool2d(2),
           nn.Conv2d(in_channels, out_channels, kernel_size=2,stride=2, padding=0, bias=True)
        )

    def forward(self, x):
        return self.down_sample_conv(x)

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, in_chans=3, embed_dim=768):
        super().__init__()

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=1,stride = 1, padding = 0)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        # newshape = (x.shape[2],x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        #B,N,C
        #print(x.shape)
        return x


class C_davit(nn.Module):
    def __init__(self, first_num_channels,num_channels, num_heads,embed_dim,window_size,depth,drop_path):
        super(C_davit, self).__init__()

        window_size = window_size[0]
        self.embed_dim = embed_dim
        attention_types = ('spatial','channel')
        depth = depth

        self.down = Down_sample_conv(first_num_channels,num_channels)

        self.patch = PatchEmbed(num_channels, self.embed_dim)
        self.LN = nn.LayerNorm(self.embed_dim)
        self.linear = nn.Linear(self.embed_dim,self.embed_dim, bias=True)
        self.act_gelu = nn.GELU()
        self.drop = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.Block = nn.ModuleList([
            MySequential(*[
                SpatialBlock(
                    dim=self.embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=4,
                    qkv_bias=True,
                    drop_path=drop_path,
                    norm_layer=nn.LayerNorm,
                    ffn=True,
                    cpe_act=False,
                    window_size=window_size,
                )if attention_type == 'spatial' else
                ChannelBlock(
                    dim=self.embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=4,
                    qkv_bias=True,
                    drop_path=drop_path,
                    norm_layer=nn.LayerNorm,
                    ffn=True,
                    cpe_act=False
                ) if attention_type =='channel' else None
                for index,attention_type in enumerate(attention_types)])
                for i in range(depth)
            ])
        self.conv_out = nn.Conv2d(self.embed_dim,num_channels, kernel_size=3,padding=1,bias = True)

    def forward(self, x,y):
        # print('cgb')
        # print(x.shape,y.shape)
        y = self.down(y)
        # print(x.shape,y.shape)
        # n,c,h,w
        assert y.shape == x.shape ,'cross shape no same'
        

        B,C,H,W = x.shape
        x_pa = self.patch(x)
        shortcut_x = x_pa
        #b,n,c
        # Get gating weights from X
        x = self.LN(x_pa)
        x = self.linear(x)
        x = self.act_gelu(x)
        gx = x
        for blk in self.Block:
            gx,_ = blk(gx,(H,W))
        gx = self.LN(gx)
        #b,n,c
        # gx = gx.transpose(-1,-2).reshape(B,self.embed_dim,H,W)
        # gx = self.conv_out(gx)
        #b,c,h,w

        # Get gating weights from Y
        B,C,H,W = y.shape
        y_pa = self.patch(y)
        shortcut_y = y_pa
        y = self.LN(y_pa)
        y = self.linear(y)
        y = self.act_gelu(y)
        gy = y
        for blk in self.Block:
            gy,_ = blk(gy,(H,W))
        gy = self.LN(gy)
        # gy = gy.transpose(-1,-2).reshape(B,self.embed_dim,H,W)
        # gy = self.conv_out(gy)
        #print('-------gxy:',gy.shape)

        # Apply cross gating: X = X * GY, Y = Y * GX
        # print(y.shape,gx.shape)
        y = y * gx
        y = self.LN(y)
        y = self.linear(y)
        y = self.drop(y)
        y = y + shortcut_y
       

        x = x * gy  # gating x using
        x = self.LN(x)
        x = self.linear(x)
        x = self.drop(x)
        out = x + y + shortcut_x  # get all aggregated signals
        out = out.transpose(-1,-2).reshape(B,self.embed_dim,H,W)
        out = self.conv_out(out)
        #n,c,h,w
        
        return out

