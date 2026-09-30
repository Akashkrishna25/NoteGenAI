import torch

from feed_forward import FeedForward


batch_size = 1
sequence_length = 5
embedding_dim = 8
hidden_dim = 32


x = torch.randn(
    batch_size,
    sequence_length,
    embedding_dim
)


ffn = FeedForward(
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim
)


output = ffn(x)


print("Input shape:")
print(x.shape)

print("\nOutput shape:")
print(output.shape)