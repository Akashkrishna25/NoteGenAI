import torch

from transformer_block import TransformerBlock


batch_size = 1
sequence_length = 5
embedding_dim = 8
num_heads = 2
hidden_dim = 32


x = torch.randn(
    batch_size,
    sequence_length,
    embedding_dim
)


block = TransformerBlock(
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    hidden_dim=hidden_dim
)


output, attention_weights = block(x)


print("Input shape:")
print(x.shape)

print("\nOutput shape:")
print(output.shape)

print("\nAttention shape:")
print(attention_weights.shape)