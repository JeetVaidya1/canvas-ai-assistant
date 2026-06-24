"""Pure helpers from exports.py: URL/file slugs and ICS field escaping."""
import pytest

from exports import _ics_escape, _slug


@pytest.mark.unit
@pytest.mark.parametrize("raw, expected", [
    ("Intro to Biology 201!", "intro-to-biology-201"),
    ("  Spaced   Out  ", "spaced-out"),
    ("UPPER_case", "upper-case"),
    ("", "untitled"),
    ("***", "untitled"),
    (None, "untitled"),
])
def test_slug(raw, expected):
    assert _slug(raw) == expected


@pytest.mark.unit
@pytest.mark.parametrize("raw, expected", [
    ("a,b", "a\\,b"),
    ("a;b", "a\\;b"),
    ("a\nb", "a\\nb"),
    ("a\\b", "a\\\\b"),
    ("plain text", "plain text"),
])
def test_ics_escape(raw, expected):
    assert _ics_escape(raw) == expected


@pytest.mark.unit
def test_ics_escape_backslash_runs_first():
    # Order matters: a literal backslash must be doubled before other escapes,
    # otherwise the escape characters we add would themselves get escaped.
    assert _ics_escape("x;\\y") == "x\\;\\\\y"
