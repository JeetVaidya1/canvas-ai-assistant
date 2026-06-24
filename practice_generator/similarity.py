"""Topic-similarity mixin: duplicate-detection helpers used when combining
topics from filename, content, and course-title sources.
"""


class SimilarityMixin:
    """Heuristic topic-duplication checks of varying strictness."""

    def topics_are_similar(self, topic1: str, topic2: str) -> bool:
        """Check if two topics are similar enough to be considered duplicates - LESS AGGRESSIVE"""
        # Make this much less aggressive
        t1_words = set(topic1.lower().split())
        t2_words = set(topic2.lower().split())

        # Only consider them similar if they share most words AND are reasonably similar in length
        shared_words = len(t1_words & t2_words)
        min_words = min(len(t1_words), len(t2_words))

        # Only mark as similar if 80%+ words match AND both topics are short
        if shared_words >= min_words * 0.8 and min_words <= 3:
            return True

        # Exact match check
        if topic1.lower().strip() == topic2.lower().strip():
            return True

        return False

    def topics_are_very_similar(self, topic1: str, topic2: str) -> bool:
        """Even more strict similarity check for content vs filename topics"""
        t1_clean = topic1.lower().strip()
        t2_clean = topic2.lower().strip()

        # Exact match
        if t1_clean == t2_clean:
            return True

        # One is contained in the other (for short topics)
        if len(t1_clean) <= 15 and len(t2_clean) <= 15:
            if t1_clean in t2_clean or t2_clean in t1_clean:
                return True

        # High word overlap for very short topics only
        t1_words = set(t1_clean.split())
        t2_words = set(t2_clean.split())

        if len(t1_words) <= 2 and len(t2_words) <= 2:
            shared = len(t1_words & t2_words)
            if shared >= max(len(t1_words), len(t2_words)) * 0.8:
                return True

        return False
