"""FastAPI app: wires middleware + routers. Shared state lives in deps.py."""
import logging

from core.config import get_settings
from logging_config import setup_logging

# Configure logging before the routers/engines import so their module-level
# log records (and anything during request handling) use the JSON formatter.
setup_logging()

logger = logging.getLogger(__name__)

# Fail fast, with a message that names exactly what's missing, instead of
# letting engines blow up later with opaque client errors.
_settings = get_settings()
_missing = _settings.missing_required()
if _missing:
    raise RuntimeError(
        "Cannot start: missing required environment variables: "
        + ", ".join(_missing)
        + ". Set them in the environment (or a .env file) and restart. "
        "See core/config.py for the full list of settings."
    )

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from core.middleware import RequestContextMiddleware  # noqa: E402
from errors import install_error_handlers  # noqa: E402
from routers import ai_export, analytics, auth_api, canvas_lms, chat, concepts, courses, exams, exports_api, flashcards, github_io, notes, planner, practice, quiz, reviews, sharing, system, tutor  # noqa: E402

app = FastAPI()

# CORS: defaults to the local dev frontend. In production set ALLOWED_ORIGINS to
# a comma-separated allowlist (e.g. the Vercel URL). A wildcard ("*") must be
# opted into explicitly and is warned about — credentials can't be used with a
# wildcard origin, so only enable them once origins are pinned.
_allowed_origins = _settings.allowed_origins_list()
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
# Added after CORS so it runs outermost: every response (including CORS
# preflights and error responses) carries X-Request-ID and gets an access log.
app.add_middleware(RequestContextMiddleware)

install_error_handlers(app)

for _m in (ai_export, analytics, auth_api, canvas_lms, chat, concepts, courses, exams, exports_api, flashcards, github_io, notes, planner, practice, quiz, reviews, sharing, system, tutor):
    app.include_router(_m.router)
