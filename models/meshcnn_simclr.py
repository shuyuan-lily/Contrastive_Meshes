import torch.nn as nn
from . import create_model

class MeshCNNSimCLR(nn.Module):

    def __init__(self, base_model, opt, out_dim):
        super(MeshCNNSimCLR, self).__init__()
        self.model_dict = {"meshcnn": create_model(opt)}
        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
        except KeyError:
            raise Exception("Invalid backbone architecture. Check the config file and pass meshcnn.")
        else:
            return model
    
    def forward(self, x):
        return self.backbone(x)