import torch
import torch.nn.functional as F
from monai.data.utils import dense_patch_slices
from monai.inferers import SlidingWindowInferer


x = torch.randn(4,4,300,300)
sw = SlidingWindowInferer(roi_size, sw_batch_size=1, overlap=0.25)
print(sw.shape)