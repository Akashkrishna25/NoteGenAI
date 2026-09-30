import os
import sys
import torch


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..")
)

MODEL_DIR = os.path.join(PROJECT_ROOT, "llm", "model")
TOKENIZER_DIR = os.path.join(PROJECT_ROOT, "llm", "tokenizer")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

sys.path.append(MODEL_DIR)
sys.path.append(TOKENIZER_DIR)

from llm import SmallLLM
from tokenizer import SimpleTokenizer


class LLMService:

    def __init__(self):

        vocabulary_path = os.path.join(
            DATA_DIR,
            "vocabulary.pt"
        )

        model_path = os.path.join(
            DATA_DIR,
            "notegen_model.pt"
        )

        vocabulary = torch.load(
            vocabulary_path,
            weights_only=False
        )

        self.tokenizer = SimpleTokenizer()

        self.tokenizer.token_to_id = vocabulary

        self.tokenizer.id_to_token = {
            index: token
            for token, index in vocabulary.items()
        }

        vocab_size = len(vocabulary)

        self.model = SmallLLM(
            vocab_size=vocab_size,
            embedding_dim=64,
            num_heads=4,
            hidden_dim=256,
            num_layers=4,
            max_seq_length=32
        )

        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location="cpu",
                weights_only=True
            )
        )

        self.model.eval()

    def generate(
        self,
        prompt,
        max_new_tokens=150
    ):

        input_ids = self.tokenizer.encode(prompt)

        input_ids = torch.tensor(
            [input_ids],
            dtype=torch.long
        )

        with torch.no_grad():

            for _ in range(max_new_tokens):

                context = input_ids[:, -32:]

                logits, _ = self.model(context)

                logits = logits[:, -1, :]

                next_token = torch.argmax(
                    logits,
                    dim=-1,
                    keepdim=True
                )

                input_ids = torch.cat(
                    [
                        input_ids,
                        next_token
                    ],
                    dim=1
                )

        return self.tokenizer.decode(
            input_ids[0].tolist()
        )