# Representation Learning of Meshes Based on SIMCLR Method

In this project, I explore the contrastive learning method proposed by Chen et al. in [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709) using the existing [MeshCNN](https://arxiv.org/abs/1809.05910) framework, a general-purpose deep neural network for deep learning on meshes.

This repository is adapted from the [MeshCNN codebase](https://github.com/ranahanocka/MeshCNN). 


## Getting Started

To clone this repo to your own working directory:
```
git clone https://github.com/shuyuan-lily/Contrastive_Meshes.git
cd Contrastive_Meshes
```
Set up your working environment with:
```
conda env create -f contrastive_meshes.yml
conda activate contrastive_meshes
```

## Usage

The experiments so far are mainly conducted on the [`shrec16`](http://shapenet.cs.stanford.edu/shrec16/) dataset for classification tasks. To download the dataset, run:
```
bash scripts/shrec/get_data.sh
```
Then, to run training:
```
bash scripts/shrec/train.sh
```

To view the visualization of training loss, test accuracy, and intermetiate model weights using TensorboardX, open another terminal and run the following command:

```
tensorboard dev upload --logdir runs
```
Here is an example of the training loss plot:

<img src="/docs/imgs/training_loss.png" width="450px"/> 

To run test on the latest updated network, run: 
```
bash scripts/shrec/test.sh
```

To view the [T-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding) visualization on the mesh embeddings (before the projection head, i.e., right after the global pooling step), run:

```
bash scripts/shrec/tsne.sh
```
## Configuration


## Evaluation

### Preliminary Evaluation

Representation learning differs from traditional classification or regression tasks, so there has not been an agreed or pre-defined means of testing model performance without reference to particular downstream tasks.

To preliminarily get a direct sense of network performance, I define the following evaluation metric, analogous to classification. 

> 1. Given a representation, find its nearest neighbor in feature space (that is, before the nonlinear projection head in the SimCLR architecture);
> 2. Check whether the nearest neighbor belongs to the same class with the original representation. If it does, then we have an instance that is analogous to the "correct label" in the case of traditional classification tasks.

Count all the instances where the representation is "correctly labeled" and obtain a global accuracy.


### Classification

Compute the global accuracy of classification task performed based on the representations obtained with the SimCLR representation learning. 

### Clustering

TODO: Representation learning and clustering have close connections. In a sense, the feature representations each occupy a "location" in the feature space, and so does the projected representations in the latent space. In SimCLR, we want the feature representations to be as close as possible for the meshes belonging to the same class, and as far as possible for the ones belonging to different classes. With clustering, there is a similar objective. Therefore, we can seek to utilize Silhouette Score in the case of K-means clustering tasks as an additional evaluation metric. 