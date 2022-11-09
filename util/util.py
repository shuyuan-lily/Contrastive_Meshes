from __future__ import print_function
import torch
import numpy as np
import os

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

MESH_EXTENSIONS = [
    '.obj'
]

def is_mesh_file(filename):
    return any(filename.endswith(extension) for extension in MESH_EXTENSIONS)

def pad(input_arr, target_length, val=0, dim=1):
    shp = input_arr.shape
    npad = [(0, 0) for _ in range(len(shp))]
    npad[dim] = (0, target_length - shp[dim])
    return np.pad(input_arr, pad_width=npad, mode='constant', constant_values=val)