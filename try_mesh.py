import torch
import numpy as np
import os
from models.layers.mesh import Mesh
from data.base_dataset import collate_fn
from options.train_options import TrainOptions
from data.classification_data import ClassificationData
from data.simclr_data import SimCLRData

if __name__ == '__main__':
    opt = TrainOptions().parse()
    object_class = 'alien'
    obj_file_path = os.path.join("datasets", 'shrec_16', object_class, 'train', 'T5.obj')
    #npz_file_path = os.path.join("datasets", 'shrec_16', object_class, 'train', 'cache', 'T5_009.npz')
    mesh1 = Mesh(file=obj_file_path, opt=opt, hold_history=False, export_folder=opt.export_folder)
    #print(vars(mesh1))
    print('vs', mesh1.vs.shape)
    mesh2 = Mesh(file=obj_file_path, opt=opt, hold_history=False, export_folder=opt.export_folder)
    #print(vars(mesh2))
    print('vs', mesh2.vs.shape)
    print('gemm_edges(shape)', mesh2.gemm_edges.shape)
    print('features (shape):', mesh2.features.shape)
    print('edges_count', mesh2.edges_count)

    
    print("Are all the vertices the same?", np.all(mesh1.vs == mesh2.vs))
    # print(mesh1.ve[0])
    # print(mesh2.ve[0])
    mesh1.export("mesh1_experiment.obj")
    mesh2.export("mesh2_experiment.obj")