import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

tsne = TSNE()

def plot_vecs_n_labels(v, labels, fname):
    fig = plt.figure(figsize=(10, 10))
    plt.axis('off')
    sns.set_style('darkgrid')
    sns.scatterplot(v[:,0], v[:,1], hue=labels, legend='full', palette=sns.color_palette("bright", 5))
    plt.legend()
    plt.savefig(fname)
    plt.close()


    
def visualize_tsne(epoch=-1):
    print('Running T-SNE visualization')
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    dataset = DataLoader(opt)
    model = create_model(opt)
    writer = Writer(opt)
    # test
    writer.reset_counter()
    
    for i, data in enumerate(dataset):
        model.set_input(data)
        reps, out = model.forward() 
        x = data['edge_features']
        y_tsne = tsne.fit_transform(reps[0].cpu().data)
        labels = data['label']
        plot_vecs_n_labels(y_tsne,labels,'tsne_train_last_layer.png')


if __name__ == '__main__':
    visualize_tsne()