import torch
import numpy as np
import os
from util.mesh_utils import getRotMat
from models.layers.mesh import Mesh
from models.layers.mesh_prepare import affine, scale_verts
from data.base_dataset import collate_fn
from options.train_options import TrainOptions
from data.classification_data import ClassificationData
from data.simclr_data import SimCLRData
if __name__ == '__main__':
    opt = TrainOptions().parse()
    
    # choose an object class
    object_class = 'alien'
    obj_file_path = os.path.join("datasets", 'shrec_16', object_class, 'train', 'T5.obj')
    
    mesh = Mesh(file=obj_file_path, opt=opt, hold_history=False, export_folder=opt.export_folder)
    mean = 1
    var = 0.1

    trials = 10

    for i in range(trials):
        
        # rotation
        axis = np.random.normal(loc=mean, scale=var, size=(3, 1))
        theta = np.random.uniform(0, 2 * np.pi)
        A = getRotMat(axis=axis, theta=theta)
        affine(mesh, A)

        # shear
        B = np.eye(3)
        row, col = np.random.choice(3, 2, replace=False)
        B[row][col] = np.random.normal(loc=0, scale=.4)
        affine(mesh, B) 
        
        # scale vert
        scale_verts(mesh, mean=1, var=.1)
        
        '''
        # some very simple shear
        row = random.choice([0, 1, 2])
        col = random.choice([0, 1, 2])
        A[row, col] += random.randrange(0, 10)/10
        row = random.choice([0, 1, 2])
        col = random.choice([0, 1, 2])
        A[row, col] += random.randrange(0, 10)/10
        '''
        
        export_file_name = f"{object_class}_random_{i}_T5.obj"
        mesh.export(export_file_name)