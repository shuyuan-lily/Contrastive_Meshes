import os
from os import listdir
import shutil

list_of_dir = listdir('datasets/shrec_16')

for d in list_of_dir:
    classes = d
    files = listdir(os.path.join('datasets/shrec_16', classes, 'train'))
    for f in files:
        if f.startswith('random_'):
            source = os.path.join('datasets/shrec_16', classes, 'train', f)
            newname = f.replace('random_', '').replace('.obj', '_random.obj')
            dest = os.path.join('datasets/shrec_16', classes, 'train', newname)
            try:
                os.rename(source, dest)
                print("Source path renamed to destination path successfully.")
            
            # If Source is a file 
            # but destination is a directory
            except IsADirectoryError:
                print("Source is a file but destination is a directory.")
            
            # If source is a directory
            # but destination is a file
            except NotADirectoryError:
                print("Source is a directory but destination is a file.")
            
            # For permission related errors
            except PermissionError:
                print("Operation not permitted.")
            
            # For other errors
            except OSError as error:
                print(error)


            