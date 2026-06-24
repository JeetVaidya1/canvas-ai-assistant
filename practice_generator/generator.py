"""``PracticeGenerator`` — composes topic, retrieval, generation, and mastery
mixins into the public practice-problem generator for any course subject.
"""
from providers import make_client

from .mastery import MasteryMixin
from .topics import TopicsMixin
from .similarity import SimilarityMixin
from .retrieval import RetrievalMixin
from .generation import GenerationMixin


class PracticeGenerator(MasteryMixin, TopicsMixin, SimilarityMixin, RetrievalMixin, GenerationMixin):
    """Generate practice problems from course materials - works for any subject"""

    def __init__(self):
        self.openai_client = make_client()
        self._supabase = None
