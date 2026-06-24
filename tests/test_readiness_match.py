"""readiness_engine._match_mastery — fuzzy topic→mastery matching (pure)."""
import pytest

from readiness_engine import _match_mastery


@pytest.mark.unit
def test_exact_match_case_insensitive():
    assert _match_mastery("Mitosis", {"mitosis": 80.0}) == 80.0


@pytest.mark.unit
def test_key_is_substring_of_topic():
    assert _match_mastery("Cell Division", {"division": 50.0}) == 50.0


@pytest.mark.unit
def test_topic_is_substring_of_key():
    assert _match_mastery("bio", {"biology": 40.0}) == 40.0


@pytest.mark.unit
def test_no_match_returns_none():
    assert _match_mastery("Physics", {"chemistry": 10.0}) is None


@pytest.mark.unit
def test_empty_mastery_returns_none():
    assert _match_mastery("anything", {}) is None
