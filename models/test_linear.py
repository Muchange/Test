import torch 





import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import math

data = np.load('/media/xd/date/muzhaoshan/Muzhaoshan/LungSegData/02Normalized_subsetUR/1GMQX2WE.npy')
print(data.shape)
data = data.transpose(1,0,2)
data = data.astype(np.float32)
data = data[:,:,180:185]

#data = data.transpose(2,0,1)
#print(data.shape)
#data = data.transpose(1,2,0)

w,h,c = data.shape
num_channal = c
Lin = nn.Linear(c,num_channal,bias = True)
print(data.shape)
data  = data.reshape(w*h,c)
print(data.shape)
data = torch.from_numpy(data)
data = Lin(data)
data = data.detach().numpy()
print(data.shape)
data = data.reshape(w,h,num_channal)
print(data.shape)
plt.figure()
num_show = math.ceil(math.sqrt(num_channal))
print(num_show)
for i in range(num_channal):
    num = i
    x = data[:,:,num]
    plt.subplot(num_show,num_show,i+1)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x,cmap='gray')
plt.show()

