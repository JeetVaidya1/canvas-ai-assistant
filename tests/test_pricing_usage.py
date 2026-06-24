"""Cost estimation (providers.pricing) + usage accumulation (usage_tracker)."""
from types import SimpleNamespace

import pytest

import usage_tracker
from providers.pricing import estimate_cost


@pytest.mark.unit
def test_cost_routes_by_model_tier():
    # haiku in=1.00/Mtok, out=5.00/Mtok
    haiku = estimate_cost("claude-haiku-4-5-20251001", input_tokens=1_000_000, output_tokens=0)
    assert haiku == pytest.approx(1.00, abs=1e-6)
    # sonnet out=15.00/Mtok
    sonnet = estimate_cost("claude-sonnet-4-6", input_tokens=0, output_tokens=1_000_000)
    assert sonnet == pytest.approx(15.00, abs=1e-6)


@pytest.mark.unit
def test_unknown_model_defaults_to_smart_tier():
    assert estimate_cost("mystery-model", input_tokens=1_000_000) == pytest.approx(3.00, abs=1e-6)


@pytest.mark.unit
def test_cache_reads_are_cheaper_than_fresh_input():
    fresh = estimate_cost("claude-sonnet-4-6", input_tokens=1_000_000)
    cached = estimate_cost("claude-sonnet-4-6", cache_read_tokens=1_000_000)
    assert cached < fresh
    assert cached == pytest.approx(fresh * 0.1, abs=1e-6)


@pytest.mark.unit
def test_usage_tracker_accumulates_and_snapshots():
    usage_tracker.reset()
    usage = SimpleNamespace(
        input_tokens=1000, output_tokens=500,
        cache_read_input_tokens=0, cache_creation_input_tokens=0,
    )
    usage_tracker.record("claude-haiku-4-5-20251001", usage)
    usage_tracker.record("claude-haiku-4-5-20251001", usage)

    snap = usage_tracker.snapshot()
    assert snap["total"]["calls"] == 2
    assert snap["total"]["input_tokens"] == 2000
    assert snap["total"]["output_tokens"] == 1000
    assert snap["total"]["cost_usd"] > 0
    usage_tracker.reset()


@pytest.mark.unit
def test_usage_record_is_resilient_to_missing_fields():
    usage_tracker.reset()
    # An object missing some attrs must not raise (observability is best-effort).
    event = usage_tracker.record("claude-sonnet-4-6", SimpleNamespace(input_tokens=10))
    assert event["input_tokens"] == 10
    assert event["output_tokens"] == 0
    usage_tracker.reset()
