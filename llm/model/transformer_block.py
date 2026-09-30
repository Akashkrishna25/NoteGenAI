import torch.nn as nn

from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward
from layer_norm import LayerNorm


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        hidden_dim
    ):
        super().__init__()

        self.layer_norm_1 = LayerNorm(
            embedding_dim
        )

        self.attention = MultiHeadAttention(
            embedding_dim,
            num_heads
        )

        self.layer_norm_2 = LayerNorm(
            embedding_dim
        )

        self.feed_forward = FeedForward(
            embedding_dim,
            hidden_dim
        )

    def forward(self, x):

        # -------------------------
        # Attention sub-layer
        # -------------------------

        normalized_x = self.layer_norm_1(x)

        attention_output, attention_weights = (
            self.attention(normalized_x)
        )

        # Residual connection
        x = x + attention_output

        # -------------------------
        # Feed-forward sub-layer
        # -------------------------

        normalized_x = self.layer_norm_2(x)

        feed_forward_output = (
            self.feed_forward(normalized_x)
        )

        # Residual connection
        x = x + feed_forward_output

        return x, attention_weights