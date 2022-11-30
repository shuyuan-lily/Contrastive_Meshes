import torch
import numpy as np
# import torchvision
# import torchvision.transforms.functional as TF
import copy 

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


def homogeneous_coordinates(vertices):
    number, dimension = vertices.shape
    fourth_dim = torch.ones(number, device=device).unsqueeze(1)
    vertices = torch.cat([vertices, fourth_dim], dim=1)
    return vertices


def vertice_affine_transformation(vertices, A):
    # vertices = vertices.to(device)
    A = torch.from_numpy(A).float().to(device)
    if A.size() == torch.Size([4, 4]): ## translation involved
      hom_vertice = homogeneous_coordinates(vertices) 
      transformed_verts = torch.matmul(A, hom_vertice.T)
      transformed_verts = transformed_verts[:-1].T ## reverse homogeneous coordinate
    else: ## no translation
      transformed_verts = torch.matmul(A, vertices.T).T
    return transformed_verts


def affine_transformation(mesh, A):
    new_mesh = copy.deepcopy(mesh)
    new_mesh.vs = vertice_affine_transformation(mesh.vs, A)
    return new_mesh

class AffineTransform:
    '''Apply affine transformation by random parameters'''

    def __init__(self):
        self.affine = True # This line is some random crap

    def __call__(self, mesh):
        A = torch.randn(3, 3)
        return affine_transformation(mesh, A)


def getRotMat(axis, theta):
    """
    axis: np.array, normalized vector
    theta: radians
    """
    import math

    axis = axis / np.linalg.norm(axis)
    cprod = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
    rot = math.cos(theta) * np.identity(3) + math.sin(theta) * cprod + \
          (1 - math.cos(theta)) * np.outer(axis, axis)
    return rot