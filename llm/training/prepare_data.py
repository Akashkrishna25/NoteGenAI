import os
import sys
import torch


# ============================================================
# PROJECT PATHS
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "../..")
)

TOKENIZER_DIR = os.path.join(
    PROJECT_ROOT,
    "llm",
    "tokenizer"
)

DATA_RAW_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw"
)

DATA_PROCESSED_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)


# Add tokenizer directory to Python path
sys.path.append(TOKENIZER_DIR)


# ============================================================
# IMPORT TOKENIZER
# ============================================================

from tokenizer import SimpleTokenizer


# ============================================================
# FILE PATHS
# ============================================================

training_file = os.path.join(
    DATA_RAW_DIR,
    "training.txt"
)

token_ids_file = os.path.join(
    DATA_PROCESSED_DIR,
    "token_ids.pt"
)

vocabulary_file = os.path.join(
    DATA_PROCESSED_DIR,
    "vocabulary.pt"
)


# ============================================================
# CREATE PROCESSED DIRECTORY
# ============================================================

os.makedirs(
    DATA_PROCESSED_DIR,
    exist_ok=True
)


# ============================================================
# READ TRAINING DATA
# ============================================================

print("Loading training data...")

with open(
    training_file,
    "r",
    encoding="utf-8"
) as file:

    text = file.read()


print(f"Training text length: {len(text)} characters")


# ============================================================
# CREATE TOKENIZER
# ============================================================

tokenizer = SimpleTokenizer()

tokenizer.build_vocabulary(text)


print(
    f"Vocabulary size: {len(tokenizer.token_to_id)}"
)


# ============================================================
# ENCODE TEXT
# ============================================================

token_ids = tokenizer.encode(text)


print(
    f"Total tokens: {len(token_ids)}"
)


# ============================================================
# SAVE TOKEN IDs
# ============================================================

torch.save(
    token_ids,
    token_ids_file
)

print(
    f"Token IDs saved to:\n{token_ids_file}"
)


# ============================================================
# SAVE VOCABULARY
# ============================================================

torch.save(
    tokenizer.token_to_id,
    vocabulary_file
)

print(
    f"Vocabulary saved to:\n{vocabulary_file}"
)


# ============================================================
# COMPLETE
# ============================================================

print("\nData preparation completed successfully!")