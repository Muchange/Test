from audioop import bias
from operator import mod
import xxlimited

from pkg_resources import ResourceManager
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .maxvit import *
from .davit import *
from .vit import *
from .cgb import *

class MySequential(nn.Sequential):
    def forward(self,*inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, in_chans=3, embed_dim=768):
        super().__init__()
        # img_size = to_2tuple(img_size)
        # patch_size = to_2tuple(patch_size)
        # num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        # self.img_size = img_size
        # self.patch_size = patch_size
        # self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, padding = 1)

    def forward(self, x):
        B, C, H, W = x.shape

        # FIXME look at relaxing size constraints
        #assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # print(x.shape)
        # print(self.proj(x).shape)
        # print(self.proj(x).flatten(2).shape)
        x = self.proj(x).flatten(2).transpose(1, 2)
        #B,N,C
        #print(x.shape)
        return x

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class low_cat(nn.Module):
    def __init__(self, in_channels,num_heads, window_size,drop):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,in_channels, kernel_size=1,bias = True)
        self.maxvit = Low_maxvit(in_channels,num_heads, window_size,drop)
    def forward(self, x1, x2):
        x = self.maxvit(x1)
        out = x + x2
        out = self.conv(out)
        return out

class CGB(nn.Module):
    def __init__(self, first_num_channels,num_channels, drop_path, grid_size):
        super().__init__()
        self.cgb = CGB_(first_num_channels,num_channels, drop_path, grid_size)

    def forward(self, x,y):
        return self.cgb(x,y)

class Low_vit(nn.Module):
    def __init__(self, in_feats,dim, num_heads,drop_path,norm_layer=nn.LayerNorm):
        super().__init__()
        self.vit = Vit(dim = dim, num_heads = num_heads,drop_path = drop_path)
        num_patches = in_feats * in_feats
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
        self.pos_drop = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm = norm_layer(dim)
        self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        #input:B,C,H,W
        B, C, H, W= x.shape
        x = x.reshape(B,C,H*W).permute(0,2,1)
        #shape:B,N,C
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.vit(x)
        x = self.norm(x)
        x = self.proj(x)
        x = self.pos_drop(x)
        #shape:B,N,C
        out = x.permute(0,2,1).reshape(B,C,H,W)
        #B,C,H,W
        return out

class low_catvit(nn.Module):
    def __init__(self, feat_size,in_channels,num_heads,drop):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,in_channels, kernel_size=1,bias = True)
        self.maxvit = Low_vit(in_feats = feat_size,dim = in_channels,num_heads = num_heads,drop_path = drop)
    def forward(self, x1, x2):
        x = self.maxvit(x1)
        out = x + x2
        return out


class Low_maxvit(nn.Module):
    def __init__(self, dim, num_heads,grid_size,drop,norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.vit = MaxViTBlock(
            in_channels = dim,
            out_channels = dim,
            num_heads = num_heads,
            grid_window_size = grid_size,
            attn_drop = drop,
            drop = drop,
            drop_path = drop,)
    def forward(self, x):
        out = self.vit(x)
        return out

class low_catmaxvit(nn.Module):
    def __init__(self, in_channels,num_heads, window_size,drop):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,in_channels, kernel_size=1,bias = True)
        self.maxvit = Low_maxvit(dim=in_channels,num_heads=num_heads, grid_size=window_size,drop=drop)
    def forward(self, x1, x2):
        x = self.maxvit(x1)
        out = x + x2
        out = self.conv(out)
        return out

class low_davit(nn.Module):
    def __init__(self, dim, num_heads,window_size,drop_path):
        super().__init__()
        # print(num_heads)
        window_size = window_size[0]
        depth = 1
        # self.Cha_model = ChannelBlock(
        #     dim=dim,
        #     num_heads=num_heads,
        #     mlp_ratio=4,
        #     qkv_bias=True,
        #     drop_path=drop_path,
        #     norm_layer=nn.LayerNorm,
        #     ffn=True,
        #     cpe_act=False
        # )
        # self.Spa_model = SpatialBlock(
        #     dim=dim,
        #     num_heads=num_heads,
        #     mlp_ratio=4,
        #     qkv_bias=True,
        #     drop_path=drop_path,
        #     norm_layer=nn.LayerNorm,
        #     ffn=True,
        #     cpe_act=False,
        #     window_size=window_size,
        # )
        attention_types = ('spatial','channel')
        self.Block = nn.ModuleList([
            MySequential(*[
                SpatialBlock(
                    dim=dim,
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
                    dim=dim,
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
        # self.vit = Vit(dim, num_heads,drop_path)
        self.linear = nn.Linear(dim, dim,bias = True)
        self.drop = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.patch = PatchEmbed(dim, dim)
        self.out = nn.Conv2d(dim,dim, kernel_size=1,bias = True)
        self.block_spa = nn.ModuleList([
            SpatialBlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path=drop_path,
            norm_layer=nn.LayerNorm,
            ffn=True,
            cpe_act=False,
            window_size=window_size,
        )for i in range(depth)
        ])
        self.block_chan = nn.ModuleList([
            ChannelBlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path=drop_path,
            norm_layer=nn.LayerNorm,
            ffn=True,
            cpe_act=False
        )for i in range(depth)
        ])
    def forward(self,x):
        #input:B,C,H,W
        B,C,H,W = x.shape
        x = self.patch(x)
        spa = x
        chan = x
        #B,N,C
        # spa,_  = self.Spa_model(x,(H,W))
        # chan,_ = self.Cha_model(x,(H,W))
        # for blk_spa in self.block_spa:
        #     spa,_ = blk_spa(spa,(H,W))
        # for blk_chan in self.block_chan:
        #     chan,_ = blk_chan(chan,(H,W))
        # sc = spa + chan
        for blk in self.Block:
            x,_ = blk(x,(H,W))
        #B,N,C
        # print(sc.shape)
        out = self.linear(x)
        out = out.transpose(-1,-2).reshape(B,C,H,W)
        return out

class low_catdavit(nn.Module):
    def __init__(self,in_channels,num_heads,window_size,drop):
        super().__init__()
        # print(num_heads)
        self.conv = nn.Conv2d(in_channels,in_channels, kernel_size=1,bias = True)
        self.davit = low_davit( dim= in_channels, num_heads = num_heads,window_size= window_size,drop_path  = drop)
    def forward(self, x1, x2):
        x = self.davit(x1)
        out = x + x2
        out = self.conv(out)
        return out




# class Up(nn.Module):
#     """Upscaling then double conv"""

#     def __init__(self, in_channels, out_channels, up_in_channels,bilinear=True):
#         super().__init__()
#         self.conv = DoubleConv(up_in_channels, out_channels)
#         self.conv_ch = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         self.conv_out = nn.Conv2d(out_channels, out_channels, kernel_size=1)

#     def forward(self, x1, x2):
#         # print('---befor',x2.shape,x1.shape)
#         x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
#         x1 = self.conv_ch(x1)
#         # print('------after',x2.shape,x1.shape)
#         out = torch.cat([x2, x1], dim=1)
#         # print('------after',out.shape)
#         out = self.conv(out)
#         return out

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, up_in_channels,bilinear=True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(up_in_channels, out_channels)

    def forward(self, x1, x2):
        # print('---befor',x2.shape,x1.shape)
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # print('------after',x2.shape,x1.shape)
        x = torch.cat([x2, x1], dim=1)
        out  =  self.conv(x)
        # print('------after',out.shape)
        return out

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# I'll finish my work soon

