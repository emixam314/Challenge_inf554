import torch
import torch.nn as nn
import torch.nn.functional as F
from .NN_Model import NNModel

class FeedforwardNeuralNetModelWithDropout(NNModel):
    def __init__(self, input_dim, hidden_dim, output_dim, p=0.8):
        self.name = "FeedforwardNeuralNetModelWithDropout"
        super().__init__()
        self.input_dim = input_dim
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim) 

        self.dropout = nn.Dropout(p=p, inplace=False)
        
        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = x.view(-1, self.input_dim)
       
        # Linear function
        out = self.fc1(x)

        # Non-linearity
        out = F.relu(out)

        out = self.dropout(out)

        # Take note here use a final sigmoid function so your loss should not go through sigmoid again as we are using BCE loss.
        out = self.fc2(out)
        out = F.sigmoid(out)
    
        return out