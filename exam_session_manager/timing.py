# timing.py - Timezone-aware time helpers for exam sessions
from datetime import datetime, timezone


def _utcnow() -> datetime:
    """Timezone-aware current UTC time (matches Postgres timestamptz reads)."""
    return datetime.now(timezone.utc)


def _utcnow_iso() -> str:
    return _utcnow().isoformat()


def _parse_dt(value: str) -> datetime:
    """Parse an ISO timestamp into a tz-aware datetime (assume UTC if naive)."""
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
