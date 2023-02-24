from operator import rshift
import numpy as np

A = np.mat('0.1882353,0.8117647;0.4117647,0.5882353')
B = np.mat('-49,40').T
result = np.linalg.solve(A,B)
print(result)