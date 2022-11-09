from mesh import Mesh
import util.mesh_utils
from util.mesh_utils import affine_transformation, getRotMat
import torch
import sys, os, copy, random
from random import randrange
import numpy as np
from os import listdir

x = np.array([1, 0, 0])
y = np.array([0, 1, 0])
z = np.array([0, 0, 1])

lst_of_dir = listdir('/home/shuyuan/shuyuan/MeshCNN/datasets/shrec_16')
lst_of_dir.remove('mean_std_cache.p')

for f in lst_of_dir:

    classes = f
    if not os.path.exists(f"/home/shuyuan/shuyuan/Contrastive_Meshes/transformed_meshes/{classes}"):
        os.mkdir(f"/home/shuyuan/shuyuan/Contrastive_Meshes/transformed_meshes/{classes}")

    obj_file_index_obj = listdir(f"/home/shuyuan/shuyuan/MeshCNN/datasets/shrec_16/{classes}/train/")[5]
    obj_file_name = f"/home/shuyuan/shuyuan/MeshCNN/datasets/shrec_16/{classes}/train/{obj_file_index_obj}" 
    export_file_name = f"/home/shuyuan/shuyuan/Contrastive_Meshes/transformed_meshes/{classes}/random_{obj_file_index_obj}"

    # Getting the list of directories
    path = f"/home/shuyuan/shuyuan/Contrastive_Meshes/transformed_meshes/{classes}/" 
    dir = os.listdir(path)
    
    # Checking if the list is empty or not
    if not len(dir) == 0:
        continue

    mesh = Mesh(obj_file_name)
    vertices = mesh.vertices

    # rotation
    R = getRotMat(axis=x, theta=random.randrange(-180, 180) * np.pi/180)
    R *= getRotMat(axis=y, theta=random.randrange(-180, 180) * np.pi/180)
    R *= getRotMat(axis=z, theta=random.randrange(-180, 180) * np.pi/180)
    # scale
    R *= random.randrange(7, 20)/10 * random.choice([1, -1])
    # some stretching
    S = np.diag([randrange(10, 20)/10, randrange(10, 20)/10, randrange(10, 20)/10])
    # some very simple shear
    A = np.eye(3)
    for i in range(2):
        zero_one_two = [0, 1, 2]
        row = random.choice(zero_one_two)
        zero_one_two.remove(row)
        col = random.choice(zero_one_two)
        A[row, col] += random.randrange(0, 10)/10
    # compose
    A = S @ R @ A 
    translated_mesh = affine_transformation(mesh, A)
    print(A)
        
    translated_mesh.export(export_file_name)