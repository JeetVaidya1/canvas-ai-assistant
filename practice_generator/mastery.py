"""Mastery lookup mixin: reads per-topic mastery from Supabase.

Provides the lazily-initialised Supabase client and the topic-mastery lookup
used by adaptive difficulty routing.
"""
import os


class MasteryMixin:
    """Supabase-backed mastery lookups for adaptive difficulty."""

    def _get_supabase(self):
        if self._supabase is None:
            from supabase import create_client
            self._supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
        return self._supabase

    def lookup_topic_mastery(self, course_id: str, topic: str, user_id: str) -> float:
        """Return the user's mastery (0..1) for a topic, or 0.5 if unknown."""
        try:
            resp = (self._get_supabase().table("learning_progress")
                    .select("mastery_level")
                    .eq("user_id", user_id)
                    .eq("course_id", course_id)
                    .eq("topic", topic)
                    .limit(1)
                    .execute())
            if resp.data and resp.data[0].get("mastery_level") is not None:
                return float(resp.data[0]["mastery_level"])
        except Exception as e:  # noqa: BLE001
            print(f"Mastery lookup failed (defaulting to 0.5): {e}")
        return 0.5
