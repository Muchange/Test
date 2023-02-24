import numpy as np

net_numpool = 5
weights = np.array([1 / (2 ** i) for i in range(net_numpool)])
mask = np.array([True] + [True if i < net_numpool else False for i in range(1, net_numpool)])
weights[~mask] = 0
weights = weights / weights.sum()


