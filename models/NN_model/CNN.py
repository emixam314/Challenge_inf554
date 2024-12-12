import torch
import torch.nn as nn
import torch.nn.functional as F
from .NN_Model import NNModel

class CNNModel(NNModel):
    def _init_(self, input_dim, num_classes):
        self.name = "CNNModel"
        super(CNNModel, self)._init_()
        self.input_dim = input_dim
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm1d(128)  # After the convolution layer
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(128 * (input_dim // 2), 64)
        self.fc2 = nn.Linear(64, num_classes)  # Adjust num_classes based on your output requirements (e.g., number of event categories)

    def forward(self, x):
        if x.dim() == 2:
        # Add a sequence dimension (e.g., shape (batch_size, num_features) to (batch_size, num_features, 1))
            x = x.unsqueeze(2)
        x = x.permute(0, 2, 1)  # Change shape from (batch_size, seq_len, feature_dim) to (batch_size, feature_dim, seq_len)
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        x = F.sigmoid(x)
        return x