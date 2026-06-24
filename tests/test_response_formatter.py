"""Response post-processing (response_formatter) — pure regex transforms."""
import pytest

from response_formatter import (
    add_natural_improvements,
    format_ai_response,
    remove_redundant_sections,
)


@pytest.mark.unit
def test_strips_emoji_bold_header():
    out = format_ai_response("🎯 **Explanation Guide**\n\nThe mitochondria is the powerhouse.")
    assert "Explanation Guide" not in out
    assert "powerhouse" in out


@pytest.mark.unit
def test_removes_feel_free_to_ask_conclusion():
    out = format_ai_response("Here is the answer.\n\nFeel free to ask if you need more help!")
    assert "Feel free to ask" not in out
    assert "Here is the answer." in out


@pytest.mark.unit
def test_collapses_excessive_blank_lines():
    out = remove_redundant_sections("Para one.\n\n\n\nPara two.")
    assert "\n\n\n" not in out


@pytest.mark.unit
def test_think_of_x_as_y_rewrite():
    out = add_natural_improvements("Think of electrons as tiny planets.")
    assert out == "electrons are essentially tiny planets."


@pytest.mark.unit
def test_plain_text_is_returned_trimmed():
    assert format_ai_response("  A simple answer.  ") == "A simple answer."


@pytest.mark.unit
def test_always_returns_str():
    assert isinstance(format_ai_response(""), str)
