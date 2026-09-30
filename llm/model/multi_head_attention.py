import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads
    ):
        super().__init__()

        assert embedding_dim % num_heads == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Q, K, V projections
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

        # Final projection
        self.W_o = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

    def forward(self, x):

        batch_size, seq_len, _ = x.shape

        # --------------------------------
        # 1. Create Q, K, V
        # --------------------------------

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # --------------------------------
        # 2. Split into multiple heads
        # --------------------------------

        Q = Q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        )

        K = K.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        )

        V = V.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        )

        # Move heads before sequence
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Shape:
        # [batch, heads, sequence, head_dim]

        # --------------------------------
        # 3. Attention scores
        # --------------------------------

        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        scores = scores / math.sqrt(
            self.head_dim
        )

        # --------------------------------
        # 4. Causal mask
        # --------------------------------

        mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device
            )
        )

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        # --------------------------------
        # 5. Softmax
        # --------------------------------

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        # --------------------------------
        # 6. Weighted values
        # --------------------------------

        output = torch.matmul(
            attention_weights,
            V
        )

        # --------------------------------
        # 7. Combine heads
        # --------------------------------

        output = output.transpose(1, 2)

        output = output.contiguous().view(
            batch_size,
            seq_len,
            self.embedding_dim
        )

        # --------------------------------
        # 8. Final projection
        # --------------------------------

        output = self.W_o(output)

        return output, attention_weights