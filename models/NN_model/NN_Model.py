import torch
import torch.nn as nn
import torch.nn.functional as F

class NNModel(nn.Module):
    
    def __init__(self):
        super(NNModel, self).__init__()
        if not hasattr(self, 'name'):
            raise NotImplementedError("Chaque modèle doit avoir un attribut 'name'.")
        
    def forward(self, x):
        pass