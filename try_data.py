import os
import torch
from data.classification_data import ClassificationData
from util.util import is_mesh_file, pad
from data.simclr_data import SimCLRData
from data.segmentation_data import SegmentationData
from options.train_options import TrainOptions
from data.base_dataset import collate_fn



if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = SimCLRData(opt) 
    # dataset = ClassificationData(opt) 
    print(vars(dataset))
    print(len(dataset))
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads),
            collate_fn=collate_fn)
    print(len(dataloader))
    print(opt.batch_size)
    for i, data in enumerate(dataloader):
        if i * opt.batch_size >= opt.max_dataset_size:
            break
        print(data['meshes'].shape)
        print(data['edge_features'].shape)
        print(data['label'].shape)


    '''for root, _, fnames in sorted(os.walk(datapath)):
        for fname in sorted(fnames):
             if is_mesh_file(fname):
                path = os.path.join(root, fname)
                meshes.append(path)
    
    print(meshes)'''
    '''dir_opt = os.path.join(opt.dataroot)
    classes, class_to_idx = find_classes(dir_opt)
    
    directory = os.path.expanduser(datapath)
    for target in sorted(os.listdir(directory)):
        d = os.path.join(directory, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_mesh_file(fname) and (root.count(opt.phase)==1):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    meshes.append(item)
    print(meshes)'''

    print(len(dataloader))
    print(len(dataset))