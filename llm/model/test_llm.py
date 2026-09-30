import torch

from llm import SmallLLM


vocab_size = 1000

model = SmallLLM(
    vocab_size=vocab_size,
    embedding_dim=64,
    num_heads=4,
    hidden_dim=256,
    num_layers=4,
    max_seq_length=128
)


token_ids = torch.tensor([
    [5, 12, 7, 18, 9]
])


logits, attention = model(
    token_ids
)


print("Input:")
print(token_ids.shape)

print("\nLogits:")
print(logits.shape)

print("\nNumber of Transformer blocks:")
print(len(attention))