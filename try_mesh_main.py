from mesh import Mesh
import mesh_utils
from mesh_utils import affine_transformation, AffineTransform, getRotMat
import torch
import sys
import os
from os import listdir
import copy
import random
import numpy as np

current = os.path.dirname(os.path.realpath(__file__))

parent = os.path.dirname(current)
## print(parent)

#print(sys.path)
sys.path.append(parent)
#print(sys.path)

#print(current)
#print(parent)




A = torch.tensor([[1, 0, 0, 1],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0], 
                  [0, 0, 0, 1]]) ## translation
B = torch.tensor([[1, 0, 0], 
                  [0, 1, 0], 
                  [0, 0, 1]]) ## identity
C = torch.tensor([[1, 0, 0], 
                  [0, 2, 0], 
                  [0, 0, 1]]) ## streching



##print(affine_transformation(vertices, A))
##print(affine_transformation(vertices, B))





index = 253

classes = "cat"

def print_obj_file_path(index):
    obj_file_name = parent + f"/MeshCNN/shrec_16/{classes}/train/T" + f"{index}" + ".obj"
    print(obj_file_name)

lst_of_dir = listdir('/home/shuyuan/shuyuan/MeshCNN/datasets/shrec_16')
lst_of_dir.remove('mean_std_cache.p')
for f in lst_of_dir:

    classes = f

    obj_file_index_obj = listdir(f"/home/shuyuan/shuyuan/MeshCNN/datasets/shrec_16/{classes}/train/")[5]
    obj_file_name = f"/home/shuyuan/shuyuan/MeshCNN/datasets/shrec_16/{classes}/train/{obj_file_index_obj}" 
    print(obj_file_name)
    mesh = Mesh(obj_file_name)
    print(obj_file_name)

    vertices = mesh.vertices

    print(mesh)
    print("vertices:")
    print(mesh.vertices)
    print("faces:")
    print(mesh.faces)
    print("vertex_normals:")
    print(mesh.vertex_normals)
    print("face_normals:")
    print(mesh.face_normals)
    print(mesh.face_uvs)

    # streched_mesh = affine_transformation(mesh, C)
    # streched_mesh.export("T5_streched.obj")
    x = np.array([1, 0, 0])
    y = np.array([0, 1, 0])
    z = np.array([0, 0, 1])

    for i in range(4):
        # A = torch.randn(3, 3)
        # rotation
        A = getRotMat(axis=x, theta=random.randrange(-180, 180) * np.pi/180)
        A = getRotMat(axis=y, theta=random.randrange(-180, 180) * np.pi/180)
        A = getRotMat(axis=z, theta=random.randrange(-180, 180) * np.pi/180)
        # scale
        A *= random.randrange(-2, 2)
        # some stretching
        A[:, 0] *= random.randrange(1, 2) 
        A[:, 1] *= random.randrange(1, 2) 
        A[:, 2] *= random.randrange(1, 2) 
        # some very simple shear
        row = random.choice([0, 1, 2])
        col = random.choice([0, 1, 2])
        A[row, col] += random.randrange(0, 10)/10
        row = random.choice([0, 1, 2])
        col = random.choice([0, 1, 2])
        A[row, col] += random.randrange(0, 10)/10
        translated_mesh = affine_transformation(mesh, A)
        print(A)
        export_file_name = f"{classes}_random_{i}_{obj_file_index_obj}"
        translated_mesh.export(export_file_name)
