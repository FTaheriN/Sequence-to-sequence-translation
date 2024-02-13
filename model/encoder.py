import torch
import torch.nn as nn


class EncoderLSTM(nn.Module):
    def __init__(self, embed_dim, hidden_dim, n_layers, embedding_weights):
        super(EncoderLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding.from_pretrained(embedding_weights, freeze=True)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, dropout=0.2, batch_first=True)


    def forward(self, input):
        embedded = self.embedding(input)
        output, hidden = self.lstm(embedded)

        return output, hidden