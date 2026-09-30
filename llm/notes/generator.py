from prompt import create_notes_prompt


class NotesGenerator:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_notes(
        self,
        topic,
        level="beginner",
        length="medium"
    ):

        prompt = create_notes_prompt(
            topic=topic,
            level=level,
            length=length
        )

        # Convert prompt into tokens
        token_ids = self.tokenizer.encode(
            prompt
        )

        return token_ids