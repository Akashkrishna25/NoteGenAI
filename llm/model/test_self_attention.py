# import torch

# from self_attention import SelfAttention


# batch_size = 1
# sequence_length = 5
# embedding_dim = 8


# # Position-aware input
# x = torch.randn(
#     batch_size,
#     sequence_length,
#     embedding_dim
# )


# attention = SelfAttention(
#     embedding_dim
# )


# output, weights = attention(x)


# print("Input shape:")
# print(x.shape)

# print("\nAttention weights shape:")
# print(weights.shape)

# print("\nOutput shape:")
# print(output.shape)

# print("\nAttention weights:")
# print(weights)

#we modied 

import torch

from self_attention import CausalSelfAttention


batch_size = 1
sequence_length = 5
embedding_dim = 8


x = torch.randn(
    batch_size,
    sequence_length,
    embedding_dim
)


attention = CausalSelfAttention(
    embedding_dim
)


output, weights = attention(x)


print("Input shape:")
print(x.shape)

print("\nAttention shape:")
print(weights.shape)

print("\nAttention weights:")
print(weights)