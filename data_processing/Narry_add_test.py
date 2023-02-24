import numpy as np
import torch
# print(torch.empty((4, 4, 3)))
patches = np.random.randint(0,5,(2,2,2,3))
imgs =np.zeros((4, 4, 3))
#print(imgs.shape,patches[1].shape)
#图像块之间可能有重叠，所以最后需要除以重复的次数求平均
weights = np.ones_like(imgs)

print(imgs)
print(patches)
imgs[0:2, 0:2,:] += patches[0]
weights[0:2, 0:2,:] += 1
#重叠部分求平均
print(imgs)
imgs = imgs/weights
print(np.unique(imgs))
