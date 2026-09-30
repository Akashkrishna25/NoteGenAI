from fastapi import APIRouter

from app.schemas.notes import (
    NotesRequest,
    NotesResponse
)

from app.services.notes_service import generate_notes


router = APIRouter()


@router.post(
    "/generate-notes",
    response_model=NotesResponse
)
def create_notes(request: NotesRequest):

    notes = generate_notes(request)

    return NotesResponse(
        topic=request.topic,
        level=request.level,
        length=request.length,
        notes=notes
    )