from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.notes import router as notes_router


app = FastAPI(
    title="NoteGen AI",
    description="AI-powered study notes generator",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROUTES
# ============================================================

app.include_router(notes_router)


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():
    return {
        "message": "NoteGen AI is running!"
    }