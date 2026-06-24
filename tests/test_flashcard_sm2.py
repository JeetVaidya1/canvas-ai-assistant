"""SM-2 spaced-repetition scheduling math (flashcard_engine.sm2).

This is the core algorithm that decides when every card resurfaces — exactly the
kind of pure logic that must not silently change during a refactor.
"""
from datetime import date, timedelta

import pytest
from freezegun import freeze_time

from flashcard_engine import DEFAULT_EASE, MIN_EASE, sm2


@pytest.mark.unit
@freeze_time("2026-06-23")
def test_failed_review_resets_repetitions_and_interval():
    out = sm2(grade=2, ease=2.5, interval=10, repetitions=4)
    assert out["repetitions"] == 0
    assert out["interval"] == 1
    # ease drops on a fail: 2.5 + (0.1 - 3*(0.08 + 3*0.02)) = 2.5 - 0.32
    assert out["ease"] == pytest.approx(2.18, abs=1e-3)
    assert out["due_date"] == (date(2026, 6, 23) + timedelta(days=1)).isoformat()


@pytest.mark.unit
@freeze_time("2026-06-23")
def test_first_successful_review_interval_is_one_day():
    out = sm2(grade=5, ease=DEFAULT_EASE, interval=0, repetitions=0)
    assert out["interval"] == 1
    assert out["repetitions"] == 1
    assert out["ease"] == pytest.approx(2.6, abs=1e-3)  # perfect grade => +0.1


@pytest.mark.unit
def test_second_successful_review_interval_is_six_days():
    out = sm2(grade=5, ease=DEFAULT_EASE, interval=1, repetitions=1)
    assert out["interval"] == 6
    assert out["repetitions"] == 2


@pytest.mark.unit
def test_third_plus_review_multiplies_interval_by_ease():
    # repetitions >= 2 => interval = round(interval * ease) = round(6 * 2.5) = 15
    out = sm2(grade=4, ease=2.5, interval=6, repetitions=2)
    assert out["interval"] == 15
    assert out["repetitions"] == 3
    # grade 4 => ease delta = 0.1 - 1*(0.08 + 1*0.02) = 0.0  => unchanged
    assert out["ease"] == pytest.approx(2.5, abs=1e-3)


@pytest.mark.unit
def test_ease_never_drops_below_floor():
    out = sm2(grade=0, ease=MIN_EASE, interval=1, repetitions=0)
    assert out["ease"] == MIN_EASE


@pytest.mark.unit
@pytest.mark.parametrize("raw_grade, treated_as_pass", [(10, True), (-3, False)])
def test_grade_is_clamped_to_0_5(raw_grade, treated_as_pass):
    out = sm2(grade=raw_grade, ease=2.5, interval=1, repetitions=0)
    # grade 10 clamps to 5 (pass -> reps advances); grade -3 clamps to 0 (fail -> reset)
    assert (out["repetitions"] == 1) is treated_as_pass
