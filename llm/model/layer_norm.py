import torch
import torch.nn as nn


class LayerNorm(nn.Module):

    def __init__(
        self,
        embedding_dim,
        epsilon=1e-5
    ):
        super().__init__()

        self.epsilon = epsilon

        self.gamma = nn.Parameter(
            torch.ones(embedding_dim)
        )

        self.beta = nn.Parameter(
            torch.zeros(embedding_dim)
        )

    def forward(self, x):

        mean = x.mean(
            dim=-1,
            keepdim=True
        )

        variance = x.var(
            dim=-1,
            keepdim=True,
            unbiased=False
        )

        normalized = (
            x - mean
        ) / torch.sqrt(
            variance + self.epsilon
        )

        return (
            self.gamma * normalized
            + self.beta
        )