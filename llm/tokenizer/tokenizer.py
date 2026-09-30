import re


class SimpleTokenizer:

    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}

    def build_vocabulary(self, text):
        tokens = self.tokenize(text)

        unique_tokens = sorted(set(tokens))

        # Special tokens
        vocabulary = [
            "<PAD>",
            "<UNK>",
            "<BOS>",
            "<EOS>"
        ] + unique_tokens

        self.token_to_id = {
            token: index
            for index, token in enumerate(vocabulary)
        }

        self.id_to_token = {
            index: token
            for token, index in self.token_to_id.items()
        }

    def tokenize(self, text):
        text = text.lower()

        # Separate punctuation
        text = re.sub(r"([,.!?;:])", r" \1 ", text)

        tokens = text.split()

        return tokens

    def encode(self, text):
        tokens = self.tokenize(text)

        ids = []

        for token in tokens:
            token_id = self.token_to_id.get(
                token,
                self.token_to_id["<UNK>"]
            )

            ids.append(token_id)

        return ids

    def decode(self, ids):
        tokens = []

        for token_id in ids:
            token = self.id_to_token.get(
                token_id,
                "<UNK>"
            )

            tokens.append(token)

        return " ".join(tokens)