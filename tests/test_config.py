"""Typed settings (core/config.py): env parsing, caching, startup validation.

Hermetic: conftest.py injects fake Supabase credentials into the environment
before any app import, and explicit constructor kwargs override env/.env.
"""
import pytest

from core.config import Settings, get_settings


@pytest.mark.unit
def test_get_settings_is_cached_singleton():
    assert get_settings() is get_settings()


@pytest.mark.unit
def test_settings_read_from_env():
    settings = get_settings()
    # Values injected by conftest.py before any app module import.
    assert settings.supabase_url == "https://fake-project.supabase.co"
    assert settings.supabase_key == "fake-service-role-key"
    assert settings.supabase_anon_key == "sb_publishable_fake"
    assert settings.missing_required() == []


@pytest.mark.unit
def test_missing_required_lists_all_unset_vars():
    settings = Settings(supabase_url="", supabase_key="", supabase_anon_key="")
    assert settings.missing_required() == [
        "SUPABASE_URL",
        "SUPABASE_KEY",
        "SUPABASE_ANON_KEY",
    ]


@pytest.mark.unit
def test_missing_required_reports_only_the_missing_one():
    settings = Settings(
        supabase_url="https://x.supabase.co",
        supabase_key="",
        supabase_anon_key="anon",
    )
    assert settings.missing_required() == ["SUPABASE_KEY"]


@pytest.mark.unit
def test_whitespace_only_values_count_as_missing():
    settings = Settings(supabase_url="   ", supabase_key="k", supabase_anon_key="a")
    assert settings.missing_required() == ["SUPABASE_URL"]


@pytest.mark.unit
def test_allowed_origins_list_parses_and_strips_csv():
    settings = Settings(allowed_origins=" https://a.com, https://b.com ,, ")
    assert settings.allowed_origins_list() == ["https://a.com", "https://b.com"]


@pytest.mark.unit
def test_allowed_origins_wildcard():
    assert Settings(allowed_origins="*").allowed_origins_list() == ["*"]


@pytest.mark.unit
@pytest.mark.parametrize("value,expected", [
    ("1", True),
    ("true", True),
    ("YES", True),
    ("", False),
    ("0", False),
    ("off", False),
    ("no", False),
])
def test_debug_endpoints_flag_parses_leniently(value, expected):
    assert Settings(enable_debug_endpoints=value).debug_endpoints_enabled() is expected


@pytest.mark.unit
def test_numeric_and_bool_fields_have_sane_defaults():
    settings = Settings(ai_rate_limit_per_minute=5)
    assert settings.ai_rate_limit_per_minute == 5
    assert Settings().prompt_cache_enabled in (True, False)  # parses without error
