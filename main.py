"""FastAPI app: wires routers. Shared state lives in deps.py."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import ai_export, analytics, canvas_lms, chat, concepts, courses, exams, exports_api, flashcards, github_io, notes, planner, practice, quiz, reviews, system

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for _m in (ai_export, analytics, canvas_lms, chat, concepts, courses, exams, exports_api, flashcards, github_io, notes, planner, practice, quiz, reviews, system):
    app.include_router(_m.router)
