"""Sliding-window rate limiter (rate_limit.SlidingWindowLimiter)."""
import pytest

from rate_limit import SlidingWindowLimiter


@pytest.mark.unit
def test_allows_up_to_the_limit_then_blocks():
    lim = SlidingWindowLimiter(max_requests=3, window_seconds=60)
    assert lim.check("u1", now=0)[0] is True
    assert lim.check("u1", now=1)[0] is True
    assert lim.check("u1", now=2)[0] is True
    allowed, retry_after = lim.check("u1", now=3)
    assert allowed is False
    assert retry_after > 0


@pytest.mark.unit
def test_window_slides_so_old_hits_expire():
    lim = SlidingWindowLimiter(max_requests=2, window_seconds=60)
    lim.check("u1", now=0)
    lim.check("u1", now=1)
    assert lim.check("u1", now=2)[0] is False
    # After the window passes, the early hits drop off and capacity returns.
    assert lim.check("u1", now=62)[0] is True


@pytest.mark.unit
def test_limits_are_per_key():
    lim = SlidingWindowLimiter(max_requests=1, window_seconds=60)
    assert lim.check("u1", now=0)[0] is True
    assert lim.check("u2", now=0)[0] is True  # different user unaffected
    assert lim.check("u1", now=1)[0] is False


@pytest.mark.unit
def test_retry_after_reflects_oldest_hit():
    lim = SlidingWindowLimiter(max_requests=1, window_seconds=60)
    lim.check("u1", now=10)
    allowed, retry_after = lim.check("u1", now=10)
    assert allowed is False
    # oldest hit at t=10, window 60 -> free at t=70, so ~60s from now.
    assert 59 <= retry_after <= 61
