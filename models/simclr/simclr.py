#from .. import models.create_base_encoder as create_base_encoder

from .. import create_base_encoder
import torch.nn as nn
## problematic code, do not use yet
class SimCLR(nn.Module):
    """
    Adopt the MeshCNN network as the base encoder
    """

    def __init__(self, base_model, out_dim, opt):
        super(SimCLR, self).__init__()

        self.model_dict = {'meshcnn': create_base_encoder(opt)} ## the only line that needs to be fixed
        self.base_encoder = self._get_basemodel(base_model)

        self.n_features = self.base_encoder.res[-1]
        
        self.projection_head = nn.Sequential(
            nn.Linear(self.n_features, self.n_features),
            nn.ReLU(),
            nn.Linear(self.n_features, out_dim)
        )
        
    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
        except KeyError:
            raise Exception("Invalid backbone architecture. Check the config file and pass meshcnn.")
        else:
            return model

    def forward(self, x_i, x_j):
        h_i = self.base_encoder(x_i)
        h_j = self.base_encoder(x_j)

        z_i = self.projection_head(h_i)
        z_j = self.projection_head(h_j)
        return h_i, h_j, z_i, z_j