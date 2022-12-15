import os
import torch
from data.base_dataset import BaseDataset
from util.util import is_mesh_file, pad
from models.layers.mesh import Mesh

class SimCLRData(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device(f'cuda:{opt.gpu_ids[0]}') if opt.gpu_ids else torch.device('cpu')
        self.root = opt.dataroot
        self.dir = os.path.join(opt.dataroot)
        self.classes, self.class_to_idx = self.find_classes(self.dir)
        # self.paths = self.make_dataset(self.dir, opt.phase)
        self.paths = self.make_dataset_by_class(self.dir, self.class_to_idx, opt.phase)
        self.nclasses = len(self.classes) 
        self.size = len(self.paths)
        self.get_mean_std()

        # modify for network later.
        opt.nclasses = self.nclasses
        opt.input_nc = self.ninput_channels

    def __getitem__(self, index):
        # path = self.paths[index]
        # label = torch.tensor(0)
        path = self.paths[index][0]
        label = self.paths[index][1]

        # x_1 and x_2 each are augmented by rotation+shear, scale_vert, flip_edges, and slide_verts
        x_1 = Mesh(file=path, opt=self.opt, hold_history=False, export_folder=self.opt.export_folder)
        x_2 = Mesh(file=path, opt=self.opt, hold_history=False, export_folder=self.opt.export_folder)

        meta = {'meshes': (x_1, x_2)}

        # get edge features
        edge_features_1 = x_1.extract_features()
        edge_features_1 = pad(edge_features_1, self.opt.ninput_edges)
        edge_features_1 = (edge_features_1 - self.mean) / self.std 

        edge_features_2 = x_2.extract_features()
        edge_features_2 = pad(edge_features_2, self.opt.ninput_edges)
        edge_features_2 = (edge_features_2 - self.mean) / self.std 

        meta['edge_features'] = (edge_features_1, edge_features_2)
        meta['label'] = label
        return meta
    
    def __len__(self):
        return self.size

    # this is when the folders are organized by class...
    @staticmethod
    def find_classes(dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
        
    @staticmethod
    def make_dataset_by_class(dir, class_to_idx, phase):
        meshes = []
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if phase != 'all':
                        if is_mesh_file(fname) and (root.count(phase)==1):
                            path = os.path.join(root, fname)
                            item = (path, class_to_idx[target])
                            meshes.append(item)
                    else:
                        if is_mesh_file(fname):
                            path = os.path.join(root, fname)
                            item = (path, class_to_idx[target])
                            meshes.append(item)
        return meshes

    @staticmethod
    def make_dataset(path, phase):
        meshes = []
        assert os.path.isdir(path), "%s is not a valid directory" % path

        for root, _, fnames in sorted(os.walk(path)):
            for fname in sorted(fnames):
                if is_mesh_file(fname) and (root.count(phase)==1):
                    path = os.path.join(root, fname)
                    meshes.append(path)

        return meshes