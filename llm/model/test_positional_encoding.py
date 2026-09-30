import torch
from positional_encoding import PositionalEncoding

batch_size = 1
sequence_length = 5
embedding_dim = 8

x = torch.randn(
    batch_size,
    sequence_length,
    embedding_dim
)
positional_encoding = PositionalEncoding(
    max_seq_length=100,
    embedding_dim=embedding_dim
)

output = positional_encoding(x)

print("Input Shape")
print(x.shape)

print("\n Output shape")
print(output.shape)

print("\n input")
print(input)

print("\n output:")
print(output)