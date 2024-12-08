import torch
from torch import nn
from NN_Model import NNModel


class AttentionModel(NNModel):
    def __init__(self, embedding_dim, hidden_dim, num_classes):
        self.name = "AttentionModel"
        super(AttentionModel, self).__init__()
        self.encoder = WordEncoder(embedding_dim, hidden_dim)
        self.attention = WordAttention(hidden_dim)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, embeddings):
        lstm_out = self.encoder(embeddings)
        context_vector, attention_weights = self.attention(lstm_out)
        output = self.classifier(context_vector)
        return output, attention_weights


class WordAttention(NNModel):
    def __init__(self, hidden_dim):
        self.name = "WordAttention"
        super(WordAttention, self).__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1)  # Map hidden states to a single attention score per word

    def forward(self, lstm_out):
        # lstm_out: (batch_size, seq_length, hidden_dim * 2)
        attention_scores = self.attention(lstm_out).squeeze(-1)  # (batch_size, seq_length)
        attention_weights = torch.softmax(attention_scores, dim=1)  # Normalize scores
        weighted_output = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)  # (batch_size, hidden_dim * 2)
        return weighted_output, attention_weights



class WordEncoder(NNModel):
    def __init__(self, embedding_dim, hidden_dim):
        self.name = "WordEncoder"
        super(WordEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, embeddings):
        # embeddings: (batch_size, seq_length, embedding_dim)
        lstm_out, _ = self.lstm(embeddings)
        return lstm_out  # (batch_size, seq_length, hidden_dim * 2)
