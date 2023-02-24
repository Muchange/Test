import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import match
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape

        # FIXME look at relaxing size constraints
        #assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        print(x.shape)
        print(self.proj(x).shape)
        print(self.proj(x).flatten(2).shape)
        x = self.proj(x).flatten(2).transpose(1, 2)
        #print(x.shape)
        return x
if __name__ == '__main__':
    x = torch.randn((4,256,16,16))
    Patch  = PatchEmbed(img_size=16, patch_size=4, in_chans=256, embed_dim=256*(match.pow((16/4),2)))
    out = Patch(x)
    print(out.shape)