import os
import shutil
from os import listdir

list_of_dir = listdir('datasets/shrec_16')
for d in list_of_dir:
    classes = d
    testpathdir = os.path.join('/home/shuyuan/shuyuan/MeshCNN/datasets/shrec_16/', classes, 'test')
    entries = listdir(testpathdir)

    for f in entries:
        fpath = os.path.join(testpathdir, f)
        if os.path.isfile(fpath) and f.endswith('.obj'):
            D = os.path.join('datasets/shrec_16', classes, 'test')
            print("Before copying file:")
            print(os.listdir(D))
            src = fpath
            dst = os.path.join('datasets/shrec_16', classes, 'test', f)
            path = shutil.copyfile(src,dst)
  
            print("After copying file:")
            print(os.listdir(D))
            
            # print path of the newly created duplicate file
            print("Path of the duplicate file is:")
            print(path)