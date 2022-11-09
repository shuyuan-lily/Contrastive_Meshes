import os
from os import listdir

for f in listdir('/home/shuyuan/shuyuan/MeshCNN/datasets/shrec_16'):
    print(f)

list_of = listdir('/home/shuyuan/shuyuan/MeshCNN/datasets/shrec_16')
list_of.remove('mean_std_cache.p')
print(list_of)