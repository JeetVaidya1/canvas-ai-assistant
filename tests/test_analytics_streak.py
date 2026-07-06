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
def test_three_consecutive_days_is_three():
    # Phase 4 fix: previously under-counted to 2 (subtracted the running streak
    # instead of stepping back one day).
    rows = [_ts("2026-06-23"), _ts("2026-06-22"), _ts("2026-06-21")]
    assert _engine().calculate_study_streak(rows) == 3


@pytest.mark.unit
@freeze_time("2026-06-23")
def test_longer_run_counts_all_consecutive_days():
    rows = [_ts(f"2026-06-{d:02d}") for d in (23, 22, 21, 20, 19)]
    assert _engine().calculate_study_streak(rows) == 5


@pytest.mark.unit
@freeze_time("2026-06-23")
def test_yesterday_grace_preserves_streak():
    # V3 grace: studied yesterday + before but not (yet) today -> streak
    # survives, anchored at yesterday. It shouldn't read 0 at breakfast.
    rows = [_ts("2026-06-22"), _ts("2026-06-21")]
    assert _engine().calculate_study_streak(rows) == 2


@pytest.mark.unit
@freeze_time("2026-06-23")
def test_two_days_idle_breaks_the_streak():
    # Last activity two days ago -> beyond the one-day grace window -> 0.
    rows = [_ts("2026-06-21"), _ts("2026-06-20")]
    assert _engine().calculate_study_streak(rows) == 0


@pytest.mark.unit
@freeze_time("2026-06-23")
def test_grace_plus_today_do_not_double_count():
    # Today + yesterday is still just a 2-day streak under the grace rule.
    rows = [_ts("2026-06-23"), _ts("2026-06-22")]
    assert _engine().calculate_study_streak(rows) == 2


@pytest.mark.unit
@freeze_time("2026-06-23")
def test_future_dated_rows_are_ignored():
    # Clock-skewed future rows must not break (or fake) the streak.
    rows = [_ts("2026-06-25"), _ts("2026-06-23"), _ts("2026-06-22")]
    assert _engine().calculate_study_streak(rows) == 2
