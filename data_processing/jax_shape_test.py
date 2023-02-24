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
data = np.random.randn(4,128,256,32)
input  = jnp.array(data)
n,h, w, c = input.shape
print(n,h, w, c)
print(input.shape[-1],input.shape[-2],input.shape[-3])



