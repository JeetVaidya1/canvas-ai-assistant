"""Debug endpoints (routers/system.py) must 404 unless ENABLE_DEBUG_ENDPOINTS is set."""
import pytest
from fastapi import HTTPException

from routers.system import require_debug_enabled


@pytest.mark.unit
def test_gate_blocks_by_default(monkeypatch):
    monkeypatch.delenv("ENABLE_DEBUG_ENDPOINTS", raising=False)
    with pytest.raises(HTTPException) as exc:
        require_debug_enabled()
    assert exc.value.status_code == 404


@pytest.mark.unit
@pytest.mark.parametrize("value", ["0", "false", "no", "", "off"])
def test_gate_blocks_for_disabled_values(monkeypatch, value):
    monkeypatch.setenv("ENABLE_DEBUG_ENDPOINTS", value)
    with pytest.raises(HTTPException) as exc:
        require_debug_enabled()
    assert exc.value.status_code == 404


@pytest.mark.unit
@pytest.mark.parametrize("value", ["1", "true", "yes", "TRUE"])
def test_gate_opens_when_enabled(monkeypatch, value):
    monkeypatch.setenv("ENABLE_DEBUG_ENDPOINTS", value)
    require_debug_enabled()  # must not raise
