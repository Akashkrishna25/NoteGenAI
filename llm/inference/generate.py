import sys
import os
import torch


# ============================================================
# 1. PROJECT PATHS
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "../..")
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "llm",
    "model"
)

TOKENIZER_DIR = os.path.join(
    PROJECT_ROOT,
    "llm",
    "tokenizer"
)

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)


# Add model and tokenizer folders to Python path
sys.path.append(MODEL_DIR)
sys.path.append(TOKENIZER_DIR)


# ============================================================
# 2. IMPORT MODEL AND TOKENIZER
# ============================================================

from llm import SmallLLM
from tokenizer import SimpleTokenizer


# ============================================================
# 3. LOAD VOCABULARY
# ============================================================

vocabulary_path = os.path.join(
    DATA_DIR,
    "vocabulary.pt"
)

vocabulary = torch.load(
    vocabulary_path,
    weights_only=False
)

tokenizer = SimpleTokenizer()

tokenizer.token_to_id = vocabulary
tokenizer.id_to_token = {
    index: token
    for token, index in vocabulary.items()
}


# ============================================================
# 4. MODEL CONFIGURATION
# ============================================================

vocab_size = len(vocabulary)

embedding_dim = 64
num_heads = 4
hidden_dim = 256
num_layers = 4
max_seq_length = 32


# ============================================================
# 5. CREATE MODEL
# ============================================================

model = SmallLLM(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    max_seq_length=max_seq_length
)


# ============================================================
# 6. LOAD TRAINED MODEL
# ============================================================

model_path = os.path.join(
    DATA_DIR,
    "notegen_model.pt"
)

model.load_state_dict(
    torch.load(
        model_path,
        map_location="cpu",
        weights_only=True
    )
)

model.eval()


# ============================================================
# 7. TOP-K FILTER
# ============================================================

def top_k_filter(logits, k=10):

    # Make sure k doesn't exceed vocabulary size
    k = min(k, logits.size(-1))

    values, indices = torch.topk(
        logits,
        k
    )

    filtered = torch.full_like(
        logits,
        float("-inf")
    )

    filtered.scatter_(
        -1,
        indices,
        values
    )

    return filtered


# ============================================================
# 8. TEXT GENERATION
# ============================================================

def generate(
    prompt,
    max_new_tokens=50,
    temperature=0.7,
    top_k=10
):

    # Convert prompt to token IDs
    input_ids = tokenizer.encode(prompt)

    if len(input_ids) == 0:
        raise ValueError(
            "Prompt produced no tokens."
        )

    input_ids = torch.tensor(
        [input_ids],
        dtype=torch.long
    )

    with torch.no_grad():

        for _ in range(max_new_tokens):

            # Keep only the latest context
            context = input_ids[:, -max_seq_length:]

            # Model prediction
            logits, _ = model(context)

            # Get logits for the last token
            logits = logits[:, -1, :]

            # Temperature
            if temperature <= 0:
                raise ValueError(
                    "Temperature must be greater than 0."
                )

            logits = logits / temperature

            # Top-K sampling
            logits = top_k_filter(
                logits,
                k=top_k
            )

            # Convert logits into probabilities
            probabilities = torch.softmax(
                logits,
                dim=-1
            )

            # Sample next token
            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            # Add token to sequence
            input_ids = torch.cat(
                [
                    input_ids,
                    next_token
                ],
                dim=1
            )

    # Convert token IDs back to text
    generated_text = tokenizer.decode(
        input_ids[0].tolist()
    )

    return generated_text


# ============================================================
# 9. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("             NoteGen AI - Text Generator")
    print("=" * 60)

    print("\nModel loaded successfully!")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Transformer layers: {num_layers}")
    print(f"Attention heads: {num_heads}")

    print("\nEnter a topic/prompt.")
    print("Type 'exit' to stop.\n")

    while True:

        prompt = input("Prompt: ")

        if prompt.lower() == "exit":
            print("\nExiting NoteGen AI...")
            break

        if not prompt.strip():
            print("Please enter a prompt.\n")
            continue

        try:

            output = generate(
                prompt=prompt,
                max_new_tokens=50,
                temperature=0.7,
                top_k=10
            )

            print("\nGenerated Text:")
            print("-" * 60)
            print(output)
            print("-" * 60)
            print()

        except Exception as e:

            print("\nError:")
            print(e)
            print()