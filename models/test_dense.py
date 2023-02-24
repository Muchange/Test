# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main file for the MAXIM model."""

import functools
from typing import Any, Sequence, Tuple

import einops
import torch
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
import math


weight_initializer = nn.initializers.normal(stddev=2e-2)
data = np.load('/media/xd/date/muzhaoshan/MCM_Transformer/data/train_data/agum_4_crop_256_patch_3/train/images/1.npy')

class De(nn.Module):
    @nn.compact
    def __call__(self,x):
        h, w, d = x.shape
        y = nn.Dense(d, use_bias=True,
                 )(x)
        return y
model = De()
input  = jnp.array(data)
h, w, d = input.shape
print(h, w, d)
print(input.shape[-1])
variables = model.init(jax.random.PRNGKey(42),input)
out = model.apply(variables,input)
#out = y(input)
print(out.shape)
plt.figure()
num_show = math.ceil(math.sqrt(d))
print(num_show)
for i in range(d):
    num = i
    x =out[:,:,num]
    plt.subplot(num_show,num_show,i+1)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x,cmap='gray')
plt.show()


