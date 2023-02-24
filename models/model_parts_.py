""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.efficientnet_blocks import SqueezeExcite, DepthwiseSeparableConv


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.double_conv(x)

class Down_sample_conv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
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



class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.conv_ch = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x1, x2):
        #print('---befor',x2.shape,x1.shape)
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        x1 = self.conv_ch(x1)
        #print('------after',x2.shape,x1.shape)
        out = torch.cat([x2, x1], dim=1)
        return self.conv(out)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

def block_images_einops(x, patch_size):
  """Image to patches."""
  batch, height, width, channels = x.shape
  grid_height = height // patch_size[0]
  grid_width = width // patch_size[1]
  
  x = einops.rearrange(
      x, "n (gh fh) (gw fw) c -> n  (gh gw) (fh fw) c",
      gh=grid_height, gw=grid_width, fh=patch_size[0], fw=patch_size[1])

  return x


def unblock_images_einops(x, grid_size, patch_size):
  """patches to images."""
  x = einops.rearrange(
      x, "n (gh gw) (fh fw) c -> n  (gh fh) (gw fw) c",
      gh=grid_size[0], gw=grid_size[1], fh=patch_size[0], fw=patch_size[1])
  return x

def trans_re(data):
    img = data.transpose(0,2,3,1)
    Shape = img.shape
    b,w,h,c = Shape
    img = img.reshape(b*w*h,c)
    return img,Shape

def re_trans(data,Shape):
    b,w,h,c = Shape
    img = data.reshape(b,w,h,c)
    img = img.transpose(0,3,1,2)
    return img




class HiT(nn.Module):
    def __init__(self, num_channels, drop, grid_size,feat_size):
        super(HiT, self).__init__()
        self.grid_size = grid_size
        self.LN = nn.LayerNorm([num_channels,feat_size,feat_size])
        self.Linear_in = nn.Linear(num_channels ,num_channels* 2, bias=True)
        self.linear_transpose = nn.Linear(grid_size[1]*grid_size[1], grid_size[1]*grid_size[1], bias=True)
        self.linear_out = nn.Linear(num_channels * 2,num_channels, bias=True)
        self.act_gelu = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        n,num_channels,h, w = x.shape

        # input projection
        x = self.LN(x)
        x = x.unsqueeze(-1).permute(0,4,2,3,1).contiguous().squeeze(1)
        x = self.Linear_in(x)
        x = self.act_gelu(x)
        #x = x.unsqueeze(-1).permute(0,4,2,3,1).contiguous().squeeze(1)
        #shape:n,h,w,c
        u, v = torch.chunk(x, 2, dim=-1)

        # Get grid MLP weights
        gh, gw = self.grid_size
        fh, fw = h // gh, w // gw

        u = block_images_einops(u, patch_size=(fh, fw))
        #u_shape:n,h,w,num_channels
        u = torch.transpose(u, -1, -3)
        u = self.linear_transpose(u)


        u = torch.transpose(u, -1, -3)
        u = unblock_images_einops(u, grid_size=(gh, gw), patch_size=(fh, fw))

        # Get Block MLP weights
        fh, fw = self.grid_size
        gh, gw = h // fh, w // fw
        v = block_images_einops(v, patch_size=(fh, fw))
        #u_shape:n,h,w,num_channels
        v = torch.transpose(v, -1, -2)
        v = self.linear_transpose(v)

        v = torch.transpose(v, -1, -2)
        v = unblock_images_einops(v, grid_size=(gh, gw), patch_size=(fh, fw))

        x = torch.cat([u, v], axis=-1)
        x = self.linear_out(x)

        x = self.drop(x)
        out = x.unsqueeze(1).permute(0,4,2,3,1).contiguous().squeeze(-1)
        return out

class CGB(nn.Module):
    def __init__(self, first_num_channels,num_channels, drop, grid_size,feat_size,
                    act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(CGB, self).__init__()
        self.grid_size = grid_size
        #print(num_channels)
        #print(feat_size)torch
        self.LN = norm_layer([num_channels,feat_size,feat_size])
        #self.linear = nn.Linear(num_channels, num_channels, bias=True)
        self.Conv = nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=True)
        self.act_gelu = act_layer()
        self.drop = nn.Dropout(drop)
        self.GWX = HiT(num_channels, drop, grid_size,feat_size)
        self.GWY = HiT(num_channels, drop, grid_size,feat_size)
        self.Down = Down_sample_conv(first_num_channels,num_channels)
    def forward(self, x,y):

        y = self.Down(y)
        #print(x.shape)
        #n,c,h,w
        assert y.shape == x.shape
        shortcut_x = x
        shortcut_y = y

        # Get gating weights from X
        x = self.LN(x)
        x = self.Conv(x)
        #shape: n,c,h,w
        #print(img_x.shape)
        x = self.act_gelu(x)
        gx = self.GWX(x)
        #print('-------gx:',gx.shape

        # Get gating weights from Y
        y = self.LN(y)       
        y = self.Conv(y)
        y = self.act_gelu(y)
        gy = self.GWY(y)
        #print('-------gxy:',gy.shape)

        # Apply cross gating: X = X * GY, Y = Y * GX
        y = y * gx
        #y_shape:n,c,h,w
        y = self.Conv(y)
        y = self.drop(y)
        y = y + shortcut_y

        x = x * gy  # gating x using 

        x = self.Conv(x)
        x = self.drop(x)
        x = x + y + shortcut_x  # get all aggregated signals
        out = x
        return out

