
import os
import sys
import time
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable


def init_patch_square(patch_size):
    image_size = 64**2
    noise_size = image_size*patch_size
    noise_dim = int(noise_size**(0.5))
    patch = np.random.rand(noise_dim, noise_dim, 3)*255
    return patch
