
import os
import sys
import time
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable


def init_patch_square(image_size, patch_size):
    image_size = image_size**2
    noise_size = image_size*patch_size
    noise_dim = int(noise_size**(0.5))
    patch = np.random
    patch = np.zeros((1,noise_dim,noise_dim))
    patch[:,:,:] = 255.0/2
    return patch
