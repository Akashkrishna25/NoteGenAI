import torch 
from embeddings import TokenEmbedding

vocab_size = 100

embedding_dim = 8

embedding = TokenEmbedding(
    vocab_size,
    embedding_dim
)

token_ids = torch.tensor([
    [5,12,7,18]
])

vectors = embedding(token_ids)

print('Token IDs:')
print(token_ids)

print("\n Embedding Vector")
print(vectors)

print("\n shape")
print(vectors.shape)