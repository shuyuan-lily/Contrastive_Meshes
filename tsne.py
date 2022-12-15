import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np

from options.test_options import TestOptions
from options.train_options import TrainOptions
from data import DataLoader
from models import create_model
from util.writer import Writer

tsne = TSNE()
    
def visualize_tsne(epoch=-1):
    print('Running T-SNE visualization')
    opt = TestOptions().parse()
    opt.phase = 'all'
    opt.serial_batches = True  
    dataset = DataLoader(opt)
    model = create_model(opt)
    writer = Writer(opt)
    # test
    writer.reset_counter()
    
    fig = plt.figure(figsize=(10, 10))
    plt.axis('off')
    sns.set_style('darkgrid')

    all_rep = np.zeros((1, opt.out_dim))
    all_labels = np.zeros((1,))

    for i, data in enumerate(dataset):
        print(i)
        model.set_input(data)
        reps, out = model.forward() 
        print('before', all_rep.shape)
        all_rep = np.vstack((all_rep, reps[0].cpu().data))
        print('after', all_rep.shape)
        labels = data['label']
        print(labels)
        print(type(labels))
        all_labels = np.hstack((all_labels, labels))
        

    all_rep = all_rep[1:]
    all_labels = all_labels[1:].astype(int)
    print('final shape', all_rep.shape)
    y_tsne = tsne.fit_transform(all_rep)
    print('y_tsne shape', y_tsne.shape)
    sns.scatterplot(x=y_tsne[:,0], y=y_tsne[:,1], hue=all_labels, legend='full', style=all_labels,  palette=sns.color_palette("Spectral", as_cmap=True))

    fname = 'tsne_train_last_layer.png'

    plt.legend(bbox_to_anchor=(1.1, 1.1))
    plt.savefig(fname)
    plt.close()

if __name__ == '__main__':
    visualize_tsne()