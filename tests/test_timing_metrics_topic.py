"""track_exam_completion must key mastery by the REAL exam topic.

Regression for the positional-argument bug: update_learning_progress(user_id,
course_id, topic, score) put the topic into the `question` slot, so mastery was
keyed by extract_topic()'s keyword table (usually "general") instead of the
exam's actual topics. The fix passes topic= as an explicit keyword.
"""
import pytest

import learning_analytics
from exam_session_manager.timing_metrics import TimingMetricsMixin

pytestmark = pytest.mark.unit

SESSION = {
    "user_id": "u-exam",
    "course_id": "cs101",
    "exam_name": "Midterm 1",
}
RESULTS = {
    "percentage": 80,
    "time_metrics": {"time_used_seconds": 900},
    "topic_performance": {
        "Hashing": {"correct": 3, "total": 4},
        "B-Trees": {"correct": 0, "total": 2},
        "Empty":   {"correct": 0, "total": 0},   # zero-total topics are skipped
    },
}


class RecorderEngine:
    """Stands in for LearningAnalyticsEngine; records every call."""

    instances: list = []

    def __init__(self):
        self.interactions: list = []
        self.progress_calls: list = []
        RecorderEngine.instances.append(self)

    def track_interaction(self, **kwargs):
        self.interactions.append(kwargs)
        return True

    def update_learning_progress(self, *args, **kwargs):
        self.progress_calls.append((args, kwargs))


@pytest.fixture
def recorder(monkeypatch):
    RecorderEngine.instances = []
    monkeypatch.setattr(learning_analytics, "LearningAnalyticsEngine", RecorderEngine)
    return RecorderEngine


def test_topic_mastery_is_keyed_by_explicit_topic_kwarg(recorder):
    TimingMetricsMixin().track_exam_completion(dict(SESSION), dict(RESULTS))

    [engine] = recorder.instances
    # Overall exam interaction still tracked once.
    assert len(engine.interactions) == 1
    assert engine.interactions[0]["question_type"] == "exam"

    # Each non-empty topic updates mastery with topic= passed as a KEYWORD, so
    # it can never fall through to extract_topic()'s "general" bucket.
    assert len(engine.progress_calls) == 2
    by_topic = {kwargs["topic"]: kwargs for _args, kwargs in engine.progress_calls}
    assert set(by_topic) == {"Hashing", "B-Trees"}
    assert by_topic["Hashing"]["confidence"] == 0.75
    assert by_topic["B-Trees"]["confidence"] == 0.0
    # The topic must not be smuggled positionally into the question slot.
    for args, kwargs in engine.progress_calls:
        assert len(args) <= 2  # at most (user_id, course_id) positionally
        assert "topic" in kwargs and "confidence" in kwargs


def test_analytics_failure_is_swallowed(recorder, monkeypatch):
    class ExplodingEngine(RecorderEngine):
        def track_interaction(self, **kwargs):
            raise RuntimeError("analytics down")

    monkeypatch.setattr(learning_analytics, "LearningAnalyticsEngine", ExplodingEngine)
    # Must not raise — exam completion can never fail on analytics.
    TimingMetricsMixin().track_exam_completion(dict(SESSION), dict(RESULTS))
