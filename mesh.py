import numpy as np
import torch
import mesh_utils
import kaolin
from mesh_utils import device

class Mesh:
    def __init__(self, obj_file_name):
        if ".obj" in obj_file_name:
            mesh = kaolin.io.obj.import_mesh(obj_file_name, with_normals=False)
        else:
            raise ValueError(f"{obj_file_name} is not readable by kaolin mesh reader.")
        self.vertices = mesh.vertices.to(device)
        self.faces = mesh.faces.to(device)
        self.faces = self.faces + 1 
        ## this above step looks weird... but it is intended for countering the effect
        ## of the mesh reader s.t. all the index was decreased by 1.
        self.vertex_normals = None
        self.face_normals = None
        self.face_uvs = None

        '''
        if mesh.vertex_normals is not None:
            self.vertex_normals = mesh.vertex_normals.to(device).float()
            self.vertex_normals = torch.nn.functional.normalize(self.vertex_normals)
    
        if mesh.face_normals is not None:
            self.face_normals = mesh.face_normals.to(device).float()
            self.face_normals = torch.nn.functional.normalize(self.face_normals)
        '''

    def export(self, filename):
        with open(filename, "w") as f:
            for vi, v in enumerate(self.vertices):
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                if self.vertex_normals is not None:
                    f.write(f"vn {self.vertex_normals[vi, 0]} {self.vertex_normals[vi, 1]} {self.vertex_normals[vi, 2]}\n")
            for face in self.faces:
                f.write(f"f {face[0]} {face[1]} {face[2]}\n")