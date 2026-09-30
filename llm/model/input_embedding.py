import torch.nn as nn

from embeddings import TokenEmbedding
from positional_encoding import PositionalEncoding


class InputEmbedding(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        max_seq_length
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

    def forward(self, token_ids):

        token_vectors = self.token_embedding(
            token_ids
        )

        output = self.position_embedding(
            token_vectors
        )

        return output