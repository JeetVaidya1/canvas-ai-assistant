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
        """Return the user's mastery (0..1) for a topic, or 0.5 if unknown.

        Fetches the user's rows for the course and matches via Course Brain
        label bridging (exact -> substring -> shared token) instead of an
        exact ``eq("topic", ...)``, so mastery written under legacy topic
        labels still routes adaptive difficulty for the new topic names.
        """
        try:
            import course_brain
            rows = (self._get_supabase().table("learning_progress")
                    .select("topic, mastery_level")
                    .eq("user_id", user_id)
                    .eq("course_id", course_id)
                    .execute().data or [])
            mastery = course_brain.match_mastery_rows(topic, rows)
            if mastery is not None:
                return float(mastery)
        except Exception as e:  # noqa: BLE001
            print(f"Mastery lookup failed (defaulting to 0.5): {e}")
        return 0.5
