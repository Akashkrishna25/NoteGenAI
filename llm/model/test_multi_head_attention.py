import torch

from multi_head_attention import MultiHeadAttention


batch_size = 1
sequence_length = 5
embedding_dim = 8
num_heads = 2


x = torch.randn(
    batch_size,
    sequence_length,
    embedding_dim
)


attention = MultiHeadAttention(
    embedding_dim=embedding_dim,
    num_heads=num_heads
)


output, weights = attention(x)


print("Input:")
print(x.shape)

print("\nOutput:")
print(output.shape)

print("\nAttention weights:")
print(weights.shape)