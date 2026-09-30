import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ============================================================
# PROJECT PATHS
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "../..")
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "llm",
    "model"
)

TRAINING_DIR = os.path.join(
    PROJECT_ROOT,
    "llm",
    "training"
)

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)


# Add required folders to Python path
sys.path.append(MODEL_DIR)
sys.path.append(TRAINING_DIR)


# ============================================================
# IMPORTS
# ============================================================

from llm import SmallLLM
from dataset import LanguageModelDataset


# ============================================================
# CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 32

EMBEDDING_DIM = 64
NUM_HEADS = 4
HIDDEN_DIM = 256
NUM_LAYERS = 4

BATCH_SIZE = 4

LEARNING_RATE = 0.001

EPOCHS = 20


# ============================================================
# FILE PATHS
# ============================================================

TOKEN_IDS_PATH = os.path.join(
    DATA_DIR,
    "token_ids.pt"
)

VOCABULARY_PATH = os.path.join(
    DATA_DIR,
    "vocabulary.pt"
)

MODEL_PATH = os.path.join(
    DATA_DIR,
    "notegen_model.pt"
)


# ============================================================
# LOAD VOCABULARY
# ============================================================

print("=" * 60)
print("              NoteGen AI Training")
print("=" * 60)

print("\nLoading vocabulary...")

vocabulary = torch.load(
    VOCABULARY_PATH,
    weights_only=False
)

vocab_size = len(vocabulary)

print(
    f"Vocabulary size: {vocab_size}"
)


# ============================================================
# LOAD TOKEN IDS
# ============================================================

print("\nLoading token IDs...")

token_ids = torch.load(
    TOKEN_IDS_PATH,
    weights_only=False
)

print(
    f"Total tokens: {len(token_ids)}"
)


# ============================================================
# CHECK DATA SIZE
# ============================================================

if len(token_ids) <= SEQUENCE_LENGTH:

    raise ValueError(
        "Training data is too small. "
        "Add more text to training.txt."
    )


# ============================================================
# CREATE DATASET
# ============================================================

dataset = LanguageModelDataset(
    token_ids=token_ids,
    sequence_length=SEQUENCE_LENGTH
)

print(
    f"Training samples: {len(dataset)}"
)


# ============================================================
# CREATE DATALOADER
# ============================================================

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ============================================================
# CREATE MODEL
# ============================================================

print("\nCreating model...")

model = SmallLLM(
    vocab_size=vocab_size,
    embedding_dim=EMBEDDING_DIM,
    num_heads=NUM_HEADS,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    max_seq_length=SEQUENCE_LENGTH
)


print("\nModel configuration:")
print(f"Vocabulary size : {vocab_size}")
print(f"Embedding dim   : {EMBEDDING_DIM}")
print(f"Attention heads : {NUM_HEADS}")
print(f"Hidden dim      : {HIDDEN_DIM}")
print(f"Layers          : {NUM_LAYERS}")
print(f"Sequence length : {SEQUENCE_LENGTH}")


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

print("\nStarting training...\n")


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for input_ids, target_ids in dataloader:

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        logits, _ = model(input_ids)

        # Reshape logits
        logits = logits.reshape(
            -1,
            vocab_size
        )

        # Reshape targets
        target_ids = target_ids.reshape(
            -1
        )

        # Calculate loss
        loss = criterion(
            logits,
            target_ids
        )

        # Backpropagation
        loss.backward()

        # Update parameters
        optimizer.step()

        total_loss += loss.item()

    average_loss = (
        total_loss / len(dataloader)
    )

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"| Loss: {average_loss:.4f}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

torch.save(
    model.state_dict(),
    MODEL_PATH
)


print("\n" + "=" * 60)
print("Training completed!")
print("=" * 60)

print(
    f"\nModel saved to:\n{MODEL_PATH}"
)