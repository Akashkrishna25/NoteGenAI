from pydantic import BaseModel


class NotesRequest(BaseModel):

    topic: str
    level: str = "beginner"
    length: str = "medium"


class NotesResponse(BaseModel):

    topic: str
    level: str
    length: str
    notes: str