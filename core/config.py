"""Typed application settings (pydantic-settings).

Single source of truth for every environment variable the API reads at the
composition-root level. Modules call :func:`get_settings` (cached) instead of
scattering ``os.getenv`` defaults around the codebase.

Env vars are read case-insensitively; a local ``.env`` file is honoured but
real environment variables always win (same behaviour as the old
``load_dotenv(override=False)`` pattern).
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Populate os.environ from .env once, early. Several legacy modules
# (exports.py, rag/retrieval.py, notes_engine/persistence.py, ...) still read
# os.environ at call time; pydantic-settings alone would not populate it.
load_dotenv(override=False)

_TRUTHY = ("1", "true", "yes")


class Settings(BaseSettings):
    """All environment-driven configuration in one typed object."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Supabase (required) ---
    supabase_url: str = ""
    supabase_key: str = ""  # service-role key (bypasses RLS; server-side only)
    supabase_anon_key: str = ""  # anon key; used to verify user JWTs via GoTrue

    # --- AI providers (optional: the Claude OAuth path works without a key) ---
    anthropic_api_key: str = ""

    # --- HTTP / app behaviour ---
    allowed_origins: str = "http://localhost:5173"
    log_level: str = "INFO"
    # Kept as a raw string: the debug gate re-reads the env at request time by
    # design (tests toggle it at runtime), and values like "off" must not blow
    # up settings parsing. Truthy values: "1", "true", "yes".
    enable_debug_endpoints: str = ""
    prompt_cache_enabled: bool = True
    ai_rate_limit_per_minute: int = 20

    # --- model routing (engines keep their own defaults; documented here) ---
    model_default: str = ""
    model_complex: str = ""

    def allowed_origins_list(self) -> List[str]:
        """Parsed CORS allowlist. '*' yields the literal wildcard list."""
        raw = self.allowed_origins.strip()
        if raw == "*":
            return ["*"]
        return [origin.strip() for origin in raw.split(",") if origin.strip()]

    def debug_endpoints_enabled(self) -> bool:
        return self.enable_debug_endpoints.strip().lower() in _TRUTHY

    def missing_required(self) -> List[str]:
        """Names of required env vars that are unset/empty (for startup checks)."""
        required = {
            "SUPABASE_URL": self.supabase_url,
            "SUPABASE_KEY": self.supabase_key,
            "SUPABASE_ANON_KEY": self.supabase_anon_key,
        }
        return [name for name, value in required.items() if not value.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached accessor — the whole process shares one Settings instance."""
    return Settings()
