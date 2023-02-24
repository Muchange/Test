import itertools
from traceback import print_tb
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class MySequential(nn.Sequential): 
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, in_chans, embed_dim):
        super().__init__()
        # print(first_channels)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=1,stride =1, padding = 0)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x = self.proj(x)
        newshape = (x.shape[2],x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x,newshape

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp,self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvPosEnc(nn.Module):
    def __init__(self, dim, k=3, act=False, normtype=False):
        super(ConvPosEnc, self).__init__()
        self.proj = nn.Conv2d(dim,
                              dim,
                              to_2tuple(k),
                              to_2tuple(1),
                              to_2tuple(k // 2),
                              groups=dim)
        self.normtype = normtype
        if self.normtype == 'batch':
            self.norm = nn.BatchNorm2d(dim)
        elif self.normtype == 'layer':
            self.norm = nn.LayerNorm(dim)
        self.activation = nn.GELU() if act else nn.Identity()

    def forward(self, x, size: Tuple[int, int]):
        B, N, C = x.shape
        H, W = size
        assert N == H * W

        feat = x.transpose(1, 2).view(B, C, H, W)
        feat = self.proj(feat)
        if self.normtype == 'batch':
            feat = self.norm(feat).flatten(2).transpose(1, 2)
        elif self.normtype == 'layer':
            feat = self.norm(feat.flatten(2).transpose(1, 2))
        else:
            feat = feat.flatten(2).transpose(1, 2)
        x = x + self.activation(feat)
        return x

class ChannelAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super(ChannelAttention,self).__init__()
        self.num_heads = num_heads
        # print(dim,num_heads)
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        # print(B,N,C)
        # print(self.qkv(x).shape)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        k = k * self.scale
        attention = k.transpose(-1, -2) @ v
        attention = attention.softmax(dim=-1)
        x = (attention @ q.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class ChannelBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ffn=True, cpe_act=False):
        super(ChannelBlock,self).__init__()

        self.cpe = nn.ModuleList([ConvPosEnc(dim=dim, k=3, act=cpe_act),
                                  ConvPosEnc(dim=dim, k=3, act=cpe_act)])
        self.ffn = ffn
        self.norm1 = norm_layer(dim)
        # print(num_heads)
        self.attn = ChannelAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.ffn:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer)

    def forward(self, x, size):
        x = self.cpe[0](x, size)
        cur = self.norm1(x)
        cur = self.attn(cur)
        x = x + self.drop_path(cur)

        x = self.cpe[1](x, size)
        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, size


def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True):

        super(WindowAttention,self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SpatialBlock(nn.Module):
    r""" Windows Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True, drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ffn=True, cpe_act=False):
        super(SpatialBlock,self).__init__()
        self.dim = dim
        self.ffn = ffn
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.cpe = nn.ModuleList([ConvPosEnc(dim=dim, k=3, act=cpe_act),
                                  ConvPosEnc(dim=dim, k=3, act=cpe_act)])

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.ffn:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer)

    def forward(self, x, size):
        H, W = size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = self.cpe[0](x, size)
        x = self.norm1(shortcut)
        x = x.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)

        # merge windows
        attn_windows = attn_windows.view(-1,
                                         self.window_size,
                                         self.window_size,
                                         C)
        x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        x = self.cpe[1](x, size)
        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, size

class davit(nn.Module):
    def __init__(self, in_channels,num_heads,embed_dim,window_size,drop_path):
        super().__init__()
        self.embed_dim = embed_dim
        window_size = window_size[0]
        self.patch_embeds = PatchEmbed(in_chans=in_channels,embed_dim=self.embed_dim)
        # overlapped=overlapped_patch)
        attention_types = ('spatial','channel')
        depth = 1
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
        self.da_norm = nn.LayerNorm(embed_dim)
        self.da_conv = nn.Conv2d(embed_dim,in_channels, kernel_size=3,padding=1)
    def forward(self,x):
        # print('davit')
        #input:B,C,H,W
        B,C,H,W = x.shape
        x,newshape = self.patch_embeds(x)
        for blk in self.Block:
            x,_ = blk(x,newshape)
        out = self.da_norm(x)
        out = out.transpose(-1,-2).reshape(B,self.embed_dim,newshape[-2],newshape[-1])
        out = self.da_conv(out)
        # print(out.shape)
        return out


if __name__ == '__main__':
    Cha_model = ChannelBlock(
        dim=256,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path=0.1,
        norm_layer=nn.LayerNorm,
        ffn=True,
        cpe_act=False
    )
    Spa_model = SpatialBlock(
        dim=256,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path=0.1,
        norm_layer=nn.LayerNorm,
        ffn=True,
        cpe_act=False,
        window_size=8,
    )
    x = torch.randn((4,256,16,16))
    x = x.reshape(4,16*16,256)
    pa_size=(16,16)
    c_out, c_size = Cha_model(x, pa_size)
    print(c_out.shape,c_size)
    s_out,s_size = Spa_model(x,pa_size)
    print(s_out.shape,s_size)

    depths = (1, 1, 3, 1)
    drop_path_rate = 0.1
    attention_types = ('spatial', 'channel')
    architecture = [[index] * item for index, item in enumerate(depths)]
    print(architecture)
    print(len(list(itertools.chain(*architecture))))
    print(torch.linspace(0, drop_path_rate, 2 * len(list(itertools.chain(*architecture)))).shape)
    dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 2 * len(list(itertools.chain(*architecture))))]
    print(dpr)
    for block_id, block_param in enumerate(architecture):
        layer_offset_id = len(list(itertools.chain(*architecture[:block_id])))
        for attention_id, attention_type in enumerate(attention_types):
            print(attention_type)
            for layer_id, item in enumerate(block_param):
                print(layer_id , layer_offset_id, attention_id)
                d = dpr[2 * (layer_id + layer_offset_id) + attention_id]
                print(d)
        print('\n')