""" Parts of the U-Net model """

from audioop import bias
import xxlimited
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

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

class dw_conv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels):
        super().__init__()
        hidden_dim = int(4 * in_channels)
        
        self.cross_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=7, padding=(7-1)//2,bias=bias, groups=hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            )
        self.conv_block_residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 1),
        )
    def forward(self, x):
        residual = x
        out = self.cross_conv(x)
        residual = self.conv_block_residual(residual)
        out = out + residual
        return out

class C_conv(nn.Module):
    def __init__(self, first_num_channels,num_channels):
        super(C_conv, self).__init__()
        hidden_dim = int(2 * num_channels)
        self.down = Down_sample_conv(first_num_channels,num_channels)

        # self.conv_block = nn.Sequential(
        #     LayerNorm(num_channels, eps=1e-6, data_format='channels_first'),
        #     nn.Conv2d(num_channels, num_channels, kernel_size=1,stride=1, padding=0),
        #     nn.GELU(),
        # )

        # self.cross_conv = nn.Sequential(
        #     nn.Conv2d(num_channels, hidden_dim, 1),
        #     nn.GELU(),
        #     nn.Conv2d(hidden_dim, hidden_dim, kernel_size=7, padding=(7-1)//2,bias=bias, groups=hidden_dim),
        #     LayerNorm(hidden_dim, eps=1e-6, data_format='channels_first'),
        #     nn.Conv2d(hidden_dim, num_channels, 1),
        #     nn.GELU(),
        #     )

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(num_channels),
            nn.GELU(),
            nn.Conv2d(num_channels, num_channels, 1),
        )
        self.cross_conv = dw_conv(num_channels)
        # self.bn = nn.BatchNorm2d(num_channels)
        # self.relu = nn.ReLU()
        # self.ln = LayerNorm(num_channels, eps=1e-6, data_format='channels_first')
        # self.gelu = nn.GELU()
        # self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=3,stride=1, padding=1)
        
       
           
    def forward(self, x,y):
        # print('cgb')
        # print(x.shape,y.shape)
        y = self.down(y)
        # print(x.shape,y.shape)
        # n,c,h,w
        assert y.shape == x.shape ,'cross shape no same'
        shortcut_x = x
        shortcut_y = y
        
        #print(img_x.shape)
        x = self.conv_block(x)
        gx = self.cross_conv(x)

        y = self.conv_block (y)
        gy = self.cross_conv(y)
        #print('-------gxy:',gy.shape)

        # Apply cross gating: X = X * GY, Y = Y * GX
        y = y * gx
        y = self.conv_block(y)
        # y = self.conv(self.bn(self.relu(y)))
        y = y + shortcut_y

        x = x * gy  # gating x using
        x = self.conv_block(x)
        # x = self.conv(self.bn(self.relu(x)))
        out = x + y + shortcut_x  # get all aggregated signals
        #n,c,h,w
        
        return out

