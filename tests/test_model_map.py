"""Model-name routing — pure, no I/O. Asserts against the module's own constants
so it stays correct even if the env defaults change."""
import pytest

from providers.model_map import CLAUDE_FAST, CLAUDE_SMART, resolve_chat_model


@pytest.mark.unit
@pytest.mark.parametrize("name", [None, ""])
def test_missing_name_defaults_to_smart(name):
    assert resolve_chat_model(name) == CLAUDE_SMART


@pytest.mark.unit
@pytest.mark.parametrize("name", ["claude-opus-4-8", "claude-haiku-4-5", "CLAUDE-sonnet-4-6"])
def test_existing_claude_id_is_respected_verbatim(name):
    assert resolve_chat_model(name) == name


@pytest.mark.unit
@pytest.mark.parametrize("name", ["gpt-5-mini", "gpt-4o-mini", "some-haiku-tier", "gpt-5-fast", "GPT-5-MINI"])
def test_cheap_tier_routes_to_fast(name):
    assert resolve_chat_model(name) == CLAUDE_FAST


@pytest.mark.unit
@pytest.mark.parametrize("name", ["gpt-5", "gpt-4o", "gpt-5-vision", "text-davinci"])
def test_everything_else_routes_to_smart(name):
    assert resolve_chat_model(name) == CLAUDE_SMART
