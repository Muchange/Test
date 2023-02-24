from functools import partial

import torch
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else ((val,) * length)


# MBConv
class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias=False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1')
        )

    def forward(self, x):
        return x * self.gate(x)


class MBConvResidual(nn.Module):
    def __init__(self, fn, hidden_dim,dim_out, dropout=0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)
        self.net = nn.Sequential(
        nn.Conv2d(hidden_dim, dim_out, 1, padding=1),
        nn.BatchNorm2d(dim_out),
        nn.GELU()
        )

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        out = out + x
        out = self.net(out)
        return out


class Dropsample(nn.Module):
    def __init__(self, prob=0):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        device = x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1), device=device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)


def MBConv(
        dim_in,
        dim_out,
        expansion_rate=4,
        shrinkage_rate=0.25,
        dropout=0.
):
    hidden_dim = int(expansion_rate * dim_out)
    net = nn.Sequential(
        nn.Conv2d(dim_in, hidden_dim, 3, padding=1),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        nn.Conv2d(dim_in, hidden_dim, 3, padding=1),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate=shrinkage_rate),
    )
    net = MBConvResidual(net, hidden_dim,dim_out,dropout=dropout)
    return net

# helper classes
class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # print(x.shape)
        return self.net(x)

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


# attention related classes
class Attention(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=32,
            dropout=0.,
            window_size=7
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attend = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(dropout)
        )
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Dropout(dropout)
        )

        # relative positional bias

        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)
        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing='ij'))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim=-1)
        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent=False)

    def forward(self, x):
        batch, height, width, window_height, window_width, _, device, h = *x.shape, x.device, self.heads
        # flatten
        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')
        # project for queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d ) -> b h n d', h=h), (q, k, v))
        # scale
        q = q * self.scale
        # sim
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        # add positional bias
        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')
        # attention
        attn = self.attend(sim)
        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # merge heads
        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1=window_height, w2=window_width)
        # combine heads out
        out = self.to_out(out)
        return rearrange(out, '(b x y) ... -> b x y ...', x=height, y=width)

class cbg_block(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels,dim_head,window_size,dropout=0.):
        super().__init__()
        w = window_size
        self.block = nn.Sequential(
            Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1 = w, w2 = w),  # block-like attention
            PreNormResidual(in_channels, Attention(dim = in_channels, dim_head = dim_head, dropout = dropout, window_size = w)),
            PreNormResidual(in_channels, FeedForward(dim = in_channels, dropout = dropout)),
            Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),

            Rearrange('b d (w1 x) (w2 y)-> b x y w1 w2 d', w1 = w, w2 = w),  # grid-like attention
            PreNormResidual(in_channels, Attention(dim = in_channels, dim_head = dim_head, dropout = dropout, window_size = w)),
            PreNormResidual(in_channels, FeedForward(dim = in_channels, dropout = dropout)),
            Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)'),
        )
    def forward(self, x):
        return self.block(x)



class CBG_block(nn.Module):
    def __init__(self, first_num_channels,num_channels, num_head,window_size, drop_path):
        super(CBG_block, self).__init__()

        self.down = Down_sample_conv(first_num_channels,num_channels)
        self.dw_conv = nn.Conv2d(num_channels, num_channels, kernel_size=3,padding=1,
                                 bias=True,groups = num_channels)
        self.linear = nn.Linear(num_channels ,num_channels, bias=True)
        self.act_gelu = nn.GELU()
        self.drop = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.GWX = cbg_block(in_channels = num_channels,dim_head = num_head,
                            window_size = window_size,dropout=drop_path)
        self.GWY = cbg_block(in_channels = num_channels,dim_head = num_head,
                            window_size = window_size,dropout=drop_path)
        self.out_conv = nn.Conv2d(num_channels, num_channels, kernel_size=3,padding=1,bias=True)

    def forward(self, x,y):
        # print('cgb')
        # print(x.shape,y.shape)
        y = self.down(y)
        # print(x.shape,y.shape)

        # n,c,h,w
        assert y.shape == x.shape
        shortcut_x = x
        shortcut_y = y
        
        # Get gating weights from X
        x = self.dw_conv(x)
        gx = self.GWX(x)

        # Get gating weights from Y
        y = self.dw_conv(y)
        gy = self.GWY(y)
        #print('-------gxy:',gy.shape)

        # Apply cross gating: X = X * GY, Y = Y * GX
        y = y * gx
        y = y.permute(0,2,3,1)
        y = self.linear(y)
        y = self.drop(y)
        y = y.permute(0,3,1,2)
        y = y + shortcut_y
       

        x = x * gy  # gating x using
        x = x.permute(0,2,3,1)
        x = self.linear(x)
        x = self.drop(x)
        x = x.permute(0,3,1,2)
        out = x + y + shortcut_x  # get all aggregated signals
        #n,c,h,w

        return out




if __name__ == '__main__':
    x = torch.randn((4,64,128,128))
    net = CBG_block(in_channels = 64,out_channels = 128,dim_head = 32,window_size = 8,dropout = 0.1,)
    out = net(x)
    print(out.shape)