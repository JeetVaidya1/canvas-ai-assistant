"""exam_session_manager._parse_dt — ISO timestamp parsing, always tz-aware.

Per-question exam timing depends on this; a regression here would silently corrupt
elapsed-time math (the tz bug fixed back in Phase 3).
"""
from datetime import timezone

import pytest

from exam_session_manager import _parse_dt


@pytest.mark.unit
@pytest.mark.parametrize("value", [
    "2026-06-23T10:00:00Z",
    "2026-06-23T10:00:00+00:00",
    "2026-06-23T10:00:00",  # naive -> assumed UTC
])
def test_parse_dt_is_always_tz_aware(value):
    dt = _parse_dt(value)
    assert dt.tzinfo is not None
    assert dt.utcoffset() == timezone.utc.utcoffset(None)


@pytest.mark.unit
def test_parse_dt_preserves_wall_clock():
    dt = _parse_dt("2026-06-23T14:30:00Z")
    assert (dt.year, dt.month, dt.day, dt.hour, dt.minute) == (2026, 6, 23, 14, 30)
