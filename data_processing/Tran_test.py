import torch 
import torch.nn as nn

x = torch.randn((4,32,128,256))
y = x.transpose(1,3)
print(y.shape)
y = x.permute(0,3,2,1)
y = y.reshape(y.shape[0]*y.shape[1],y.shape[2],y.shape[3])
print(y.shape)