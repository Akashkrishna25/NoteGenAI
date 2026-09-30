import torch
import torch.nn as nn

from embeddings import TokenEmbedding
from positional_encoding import PositionalEncoding
from transformer_block import TransformerBlock
from layer_norm import LayerNorm


class SmallLLM(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=64,
        num_heads=4,
        hidden_dim=256,
        num_layers=4,
        max_seq_length=128
    ):
        super().__init__()

        self.token_embedding = TokenEmbedding(
            vocab_size,
            embedding_dim
        )

        self.position_embedding = PositionalEncoding(
            max_seq_length,
            embedding_dim
        )

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = LayerNorm(
            embedding_dim
        )

        # Language Model Head
        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size,
            bias=False
        )

    def forward(self, token_ids):

        # -------------------------
        # Token embeddings
        # -------------------------

        x = self.token_embedding(
            token_ids
        )

        # -------------------------
        # Positional encoding
        # -------------------------

        x = self.position_embedding(x)

        attention_weights = []

        # -------------------------
        # Transformer blocks
        # -------------------------

        for block in self.transformer_blocks:

            x, weights = block(x)

            attention_weights.append(
                weights
            )

        # -------------------------
        # Final normalization
        # -------------------------

        x = self.final_norm(x)

        # -------------------------
        # Vocabulary prediction
        # -------------------------

        logits = self.lm_head(x)

        return logits, attention_weights