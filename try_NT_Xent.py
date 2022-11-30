import torch
import torch.nn as nn
import torch.nn.functional as F
from models.NT_Xent import NT_Xent

loss = NT_Xent(16)