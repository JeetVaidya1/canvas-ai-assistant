"""FastAPI app: wires routers. Shared state lives in deps.py."""
import logging
import os

from logging_config import setup_logging

# Configure logging before the routers/engines import so their module-level
# log records (and anything during request handling) use the JSON formatter.
setup_logging()

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from errors import install_error_handlers  # noqa: E402
from routers import ai_export, analytics, auth_api, canvas_lms, chat, concepts, courses, exams, exports_api, flashcards, github_io, notes, planner, practice, quiz, reviews, sharing, system, tutor  # noqa: E402

logger = logging.getLogger(__name__)

app = FastAPI()

# CORS: defaults to the local dev frontend. In production set ALLOWED_ORIGINS to
# a comma-separated allowlist (e.g. the Vercel URL). A wildcard ("*") must be
# opted into explicitly and is warned about — credentials can't be used with a
# wildcard origin, so only enable them once origins are pinned.
_DEFAULT_ORIGINS = "http://localhost:5173"
_origins_env = os.getenv("ALLOWED_ORIGINS", _DEFAULT_ORIGINS).strip()
_allowed_origins = ["*"] if _origins_env == "*" else [o.strip() for o in _origins_env.split(",") if o.strip()]
if _allowed_origins == ["*"]:
    logger.warning(
        "ALLOWED_ORIGINS is set to '*' (wildcard). Every website can call this "
        "API from a browser; set an explicit origin allowlist in production."
    )
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
