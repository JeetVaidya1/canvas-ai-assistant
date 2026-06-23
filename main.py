"""FastAPI app: wires routers. Shared state lives in deps.py."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import analytics, chat, courses, exams, exports_api, notes, practice, quiz, system

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for _m in (analytics, chat, courses, exams, exports_api, notes, practice, quiz, system):
    app.include_router(_m.router)
