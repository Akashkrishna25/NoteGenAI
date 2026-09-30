# import torch
# import torch.nn as nn
# import math


# class SelfAttention(nn.Module):

#     def __init__(self, embedding_dim):
#         super().__init__()

#         self.embedding_dim = embedding_dim

#         # Learnable projections
#         self.W_q = nn.Linear(
#             embedding_dim,
#             embedding_dim,
#             bias=False
#         )

#         self.W_k = nn.Linear(
#             embedding_dim,
#             embedding_dim,
#             bias=False
#         )

#         self.W_v = nn.Linear(
#             embedding_dim,
#             embedding_dim,
#             bias=False
#         )

#     def forward(self, x):

#         # x shape:
#         # [batch_size, sequence_length, embedding_dim]

#         Q = self.W_q(x)
#         K = self.W_k(x)
#         V = self.W_v(x)

#         # Q × K^T
#         scores = torch.matmul(
#             Q,
#             K.transpose(-2, -1)
#         )

#         # Scale
#         scores = scores / math.sqrt(
#             self.embedding_dim
#         )

#         # Convert scores to probabilities
#         attention_weights = torch.softmax(
#             scores,
#             dim=-1
#         )

#         # Weighted sum of values
#         output = torch.matmul(
#             attention_weights,
#             V
#         )

#         return output, attention_weights


#now we use modified self attention 
import torch
import torch.nn as nn
import math


class CausalSelfAttention(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.W_q = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        self.W_k = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        self.W_v = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

    def forward(self, x):

        # x:
        # [batch_size, sequence_length, embedding_dim]

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Calculate attention scores
        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        # Scale scores
        scores = scores / math.sqrt(
            self.embedding_dim
        )

        # Sequence length
        seq_len = x.size(1)

        # Create causal mask
        mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device
            )
        )

        # Block future tokens
        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        # Softmax
        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        # Weighted values
        output = torch.matmul(
            attention_weights,
            V
        )

        return output, attention_weights