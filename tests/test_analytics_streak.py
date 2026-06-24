"""learning_analytics.calculate_study_streak — consecutive study-day counting.

Instantiated via __new__ to skip the DB-touching __init__; the method itself is
pure (operates only on the passed-in interaction list + the current date).
"""
import pytest
from freezegun import freeze_time

from learning_analytics import LearningAnalyticsEngine


def _engine():
    return LearningAnalyticsEngine.__new__(LearningAnalyticsEngine)


def _ts(day: str) -> dict:
    return {"timestamp": f"{day}T10:00:00"}


@pytest.mark.unit
def test_empty_interactions_is_zero_streak():
    assert _engine().calculate_study_streak([]) == 0


@pytest.mark.unit
@freeze_time("2026-06-23")
def test_single_day_today_is_streak_one():
    assert _engine().calculate_study_streak([_ts("2026-06-23")]) == 1


@pytest.mark.unit
@freeze_time("2026-06-23")
def test_two_consecutive_days_is_streak_two():
    rows = [_ts("2026-06-23"), _ts("2026-06-22")]
    assert _engine().calculate_study_streak(rows) == 2


@pytest.mark.unit
@freeze_time("2026-06-23")
def test_gap_breaks_the_streak():
    # Today + two-days-ago, missing yesterday -> only today counts.
    rows = [_ts("2026-06-23"), _ts("2026-06-21")]
    assert _engine().calculate_study_streak(rows) == 1


@pytest.mark.unit
@freeze_time("2026-06-23")
@pytest.mark.xfail(
    reason="BUG (flagged for Phase 4): streak subtracts the running count instead "
    "of 1, so 3+ consecutive days under-counts. Locking in the observed value so a "
    "future fix is a deliberate, visible change.",
    strict=True,
)
def test_three_consecutive_days_should_be_three():
    rows = [_ts("2026-06-23"), _ts("2026-06-22"), _ts("2026-06-21")]
    assert _engine().calculate_study_streak(rows) == 3
