import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self,max_seq_length,embedding_dim):
        super().__init__()

        position = torch.arange(
            max_seq_length
        ).unsqueeze(1)

        dimension = torch.arange(
            0,
            embedding_dim,
            2

        )
        div_term = torch.exp(
            dimension * (-math.log(1000.0)/embedding_dim)
        )
        pe = torch.zeros(
            max_seq_length,
            embedding_dim
        )

        pe[:,0::2] = torch.sin(position*div_term)

        pe[:,1::2] = torch.cos(
            position *div_term
        )
        pe = pe.unsqueeze(0)

        self.register_buffer("pe",pe)

    def forward(self,x):
            sequence_length = x.size(1)

            return x + self.pe[:,:sequence_length]