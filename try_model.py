import torch
from models import create_model
from models.networks import define_classifier, MeshConvNet, MResConv, MeshEncoder
from data import DataLoader
from options.train_options import TrainOptions

# This code helps inspect architecture. 

if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = DataLoader(opt)
    dataset_size = len(dataset)
    print()
    print(f"dataset_size: {dataset_size}")
    # model = torchvision.models.resnet18() # for comparison
    model = create_model(opt)
    print()
    print(f"number of model attributes: {len(dir(model))}")
    #model.net.module._modules['fc1'] = torch.nn.Identity()
    print(model.net.module)

    for i, data in enumerate(dataset):
        model.set_input(data)
    #print(model.net.module.__class__.__name__)
    #print(model.net.module._modules['fc1'])


