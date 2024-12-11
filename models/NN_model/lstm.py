import torch
import torch.nn as nn
import torch.nn.functional as F
from .NN_Model import NNModel

class lstm(NNModel):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        self.name = "LSTMModel"
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Fully connected layer (readout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initial hidden state (h0) and cell state (c0)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))

        # Take the last time step's output (last hidden state)
        out = out[:, -1, :]
        
        # Fully connected layer
        out = self.fc(out)

        # Optionally apply sigmoid or softmax, depending on your task
        out = torch.sigmoid(out)  # For binary classification

        return out
