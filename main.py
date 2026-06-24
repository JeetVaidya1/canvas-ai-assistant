"""FastAPI app: wires routers. Shared state lives in deps.py."""
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from errors import install_error_handlers
from routers import ai_export, analytics, auth_api, canvas_lms, chat, concepts, courses, exams, exports_api, flashcards, github_io, notes, planner, practice, quiz, reviews, sharing, system, tutor

app = FastAPI()

# CORS: defaults to "*" for local dev / preview. In production set ALLOWED_ORIGINS
# to a comma-separated allowlist (e.g. the Vercel URL). Credentials can't be used
# with a wildcard origin, so only enable them once origins are pinned.
_origins_env = os.getenv("ALLOWED_ORIGINS", "*").strip()
_allowed_origins = ["*"] if _origins_env == "*" else [o.strip() for o in _origins_env.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=_allowed_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

install_error_handlers(app)

for _m in (ai_export, analytics, auth_api, canvas_lms, chat, concepts, courses, exams, exports_api, flashcards, github_io, notes, planner, practice, quiz, reviews, sharing, system, tutor):
    app.include_router(_m.router)
