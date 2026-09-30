from app.schemas.notes import NotesRequest
from app.services.llm_service import LLMService
from app.services.prompt_service import create_notes_prompt


llm_service = LLMService()


def generate_notes(request: NotesRequest):

    prompt = create_notes_prompt(
        topic=request.topic,
        level=request.level,
        length=request.length
    )

    generated_text = llm_service.generate(
        prompt,
        max_new_tokens=150
    )

    return generated_text