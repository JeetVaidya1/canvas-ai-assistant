# timing_metrics.py - Per-session time metrics and analytics tracking
from typing import Dict, Any

from .timing import _utcnow, _parse_dt


class TimingMetricsMixin:
    """Elapsed-time, time-metric, and analytics helpers for :class:`ExamSessionManager`."""

    def calculate_time_metrics(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate time-related metrics"""
        try:
            start_time = session.get("start_time")
            time_limit = session["exam_data"]["time_limit"] * 60  # Convert to seconds

            if start_time:
                total_time_used = int((_utcnow() - _parse_dt(start_time)).total_seconds())
            else:
                total_time_used = 0

            return {
                "time_limit_seconds": time_limit,
                "time_used_seconds": total_time_used,
                "time_remaining_seconds": max(0, time_limit - total_time_used),
                "time_used_minutes": round(total_time_used / 60, 1),
                "time_efficiency": round((total_time_used / time_limit) * 100, 1) if time_limit > 0 else 0
            }

        except Exception as e:
            print(f"Time calculation error: {e}")
            return {"error": str(e)}

    def calculate_elapsed_time(self, session: Dict[str, Any]) -> int:
        """Calculate elapsed time in seconds"""
        try:
            start_time = session.get("start_time")
            if not start_time:
                return 0

            elapsed = (_utcnow() - _parse_dt(start_time)).total_seconds()
            return int(elapsed)

        except Exception as e:
            print(f"Elapsed time calculation error: {e}")
            return 0

    def calculate_question_time(self, session: Dict[str, Any], question_id: str) -> int:
        """Seconds elapsed since the student arrived at the current question.

        Reads ``current_question_start_time`` (set on start/navigate/save). Returns
        0 if it isn't set yet (e.g. legacy sessions created before this field).
        """
        start = session.get("current_question_start_time")
        if not start:
            return 0
        try:
            return max(0, int((_utcnow() - _parse_dt(start)).total_seconds()))
        except Exception as e:
            print(f"Question time calculation error: {e}")
            return 0

    def track_exam_completion(self, session: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Track exam completion for analytics"""
        try:
            # Update learning analytics
            from learning_analytics import LearningAnalyticsEngine
            analytics = LearningAnalyticsEngine()

            user_id = session["user_id"]
            course_id = session["course_id"]

            # Track overall exam performance
            analytics.track_interaction(
                user_id=user_id,
                course_id=course_id,
                question=f"Exam: {session['exam_name']}",
                answer=f"Score: {results.get('percentage', 0)}%",
                confidence=results.get('percentage', 0) / 100,
                response_time=results.get('time_metrics', {}).get('time_used_seconds', 0),
                question_type="exam"
            )

            # Track topic-specific performance
            for topic, performance in results.get('topic_performance', {}).items():
                if performance['total'] > 0:
                    topic_score = performance['correct'] / performance['total']
                    analytics.update_learning_progress(user_id, course_id, topic, topic_score)

        except Exception as e:
            print(f"Analytics tracking failed: {e}")
