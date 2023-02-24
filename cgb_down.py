""" Parts of the U-Net model """

from audioop import bias
import xxlimited
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def _gelu_ignore_parameters(
        *args,
        **kwargs
) -> nn.Module:
    """ Bad trick to ignore the inplace=True argument in the DepthwiseSeparableConv of Timm.

    Args:
        *args: Ignored.
        **kwargs: Ignored.

    Returns:
        activation (nn.Module): GELU activation function.
    """
    activation = nn.GELU()
    return activation

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

class Up_sample_conv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_sample_conv = nn.Sequential(
           nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.down_sample_conv(x)


class HiT(nn.Module):
    def __init__(self, num_channels, drop_path, grid_size):
        super(HiT, self).__init__()
        self.grid_size = grid_size
        self.LN = nn.LayerNorm(num_channels)
        self.Linear_in = nn.Linear(num_channels ,num_channels* 2, bias=True)
        self.linear_transpose = nn.Linear(grid_size[1]*grid_size[1], grid_size[1]*grid_size[1], bias=True)
        self.linear_out = nn.Linear(num_channels * 2,num_channels, bias=True)
        self.act_gelu = nn.GELU()
        self.drop = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # print('Hit')
        #n,h,w,c
        n,h,w,num_channels = x.shape
        #
        # input projection
        x = self.LN(x)
        x = self.Linear_in(x)
        x = self.act_gelu(x)
        #x = x.unsqueeze(-1).permute(0,4,2,3,1).contiguous().squeeze(1)
        #shape:n,h,w,c*2
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
        #n,h,w,c
        return x

class CGB_(nn.Module):
    def __init__(self, first_num_channels,num_channels, drop_path, grid_size,norm_layer=nn.LayerNorm):
        super(CGB_, self).__init__()
        self.grid_size = grid_size
        # self.same = same
        #print(num_channels)
        #print(feat_size)torch
        self.LN = norm_layer(num_channels)
        self.linear = nn.Linear(num_channels ,num_channels, bias=True)
        self.act_gelu = nn.GELU()
        self.drop = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.GW = HiT(num_channels, drop_path, grid_size)
        # self.GWY = HiT(num_channels, drop_path, grid_size)
        # self.up = Up_sample_conv(first_num_channels,num_channels)
        self.down = Down_sample_conv(first_num_channels,num_channels)

    def forward(self, x,y):
        y = self.down(y)
        print(x.shape,y.shape)

        # n,c,h,w
        x = x.permute(0,2,3,1)
        y = y.permute(0, 2, 3, 1)
        #n,h,w,c

        assert y.shape == x.shape
        shortcut_x = x
        shortcut_y = y
        
        # Get gating weights from X
        x = self.LN(x)
        x = self.linear(x)
        #print(img_x.shape)
        x = self.act_gelu(x)
        gx = self.GW(x)

        #print('-------gx:',gx.shape

        # Get gating weights from Y
        y = self.LN(y)
        y = self.linear(y)
        y = self.act_gelu(y)
        gy = self.GW(y)
        #print('-------gxy:',gy.shape)

        # Apply cross gating: X = X * GY, Y = Y * GX
        y = y * gx
        y = self.linear(y)
        y = self.drop(y)
        y = y + shortcut_y
       

        x = x * gy  # gating x using
        x = self.linear(x)
        x = self.drop(x)
        x = x + y + shortcut_x  # get all aggregated signals
        #n,h,w,c
        out = x.permute(0,3,1,2)
        #n,c,h,w
        
        return out

