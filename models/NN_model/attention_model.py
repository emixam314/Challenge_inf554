import torch
from torch import nn
import torch.functional as F
from .NN_Model import NNModel


class AttentionModel(NNModel):
    def _init_(self, embedding_dim, hidden_dim, num_classes):
        self.name = 'attention model'
        super(AttentionModel, self)._init_()
        self.word_attention = WordAttention(embedding_dim)
        self.minute_attention = MinuteAttention(embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, tweet_embeddings):

         # Convert tweet_embeddings into a tensor
        # The structure should be (batch_size, num_words, embedding_dim)
        print('embedding'+tweet_embeddings[0][0], type(tweet_embeddings[0][0]))

        tweet_embeddings = [
        torch.stack([  # Stack tweets within a minute
            torch.stack([  # Stack words within a tweet
                torch.tensor(word, dtype=torch.float32) for word in tweet  # Words to tensor
            ]) for tweet in minute  # Process each tweet within a minute
        ]) for minute in tweet_embeddings  # Process each minute
        ]
       
        # Ensure tweet_embeddings is a tensor with shape (batch_size, num_words, embedding_dim)
        if tweet_embeddings.dim() != 3:
            raise ValueError("Expected tweet_embeddings to have 3 dimensions: (batch_size, num_words, embedding_dim).")

        # Apply word-level attention to each tweet (each word's embeddings in the tweet)
        tweet_embeddings = torch.stack([self.word_attention(tweet) for tweet in tweet_embeddings])  # Shape (batch_size, num_words, embedding_dim)

        # Apply minute-level attention to the tweets
        minute_embedding = self.minute_attention(tweet_embeddings)  # Shape (batch_size, embedding_dim)

        # Pass minute embedding through LSTM
        output, (hidden, _) = self.lstm(minute_embedding.unsqueeze(1))
        output = self.fc(hidden.squeeze(0))  # Final classification
        output = F.sigmoid(output)
        return output



class WordAttention(NNModel):
    def _init_(self, embedding_dim):
        self.name = 'word attention'
        super(WordAttention, self)._init_()
        self.attention = nn.Linear(embedding_dim, 1, bias=False)

    def forward(self, x):
        # x: Shape (batch_size, num_words, embedding_dim)
        weights = F.softmax(self.attention(x).squeeze(-1), dim=-1)  # Compute attention scores
        weighted = torch.bmm(weights.unsqueeze(1), x)  # Apply weights
        return weighted.squeeze(1)  # Shape (batch_size, embedding_dim)



class MinuteAttention(nn.Module):
    def _init_(self, embedding_dim):
        self.name = 'minute attention'
        super(MinuteAttention, self)._init_()
        self.attention = nn.Linear(embedding_dim, 1, bias=False)

    def forward(self, x):
        # x: Shape (batch_size, num_tweets, embedding_dim)
        weights = F.softmax(self.attention(x).squeeze(-1), dim=-1)  # Compute attention scores
        weighted = torch.bmm(weights.unsqueeze(1), x)  # Apply weights
        return weighted.squeeze(1)  # Shape (batch_size, embedding_dim)