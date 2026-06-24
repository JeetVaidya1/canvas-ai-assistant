# practice_generator - COMPLETE ENHANCED VERSION for any course subject
#
# Facade package: preserves the original `practice_generator` module's public
# API so existing imports (`from practice_generator import PracticeGenerator`,
# etc.) keep working unchanged. Implementation is split across focused internal
# modules (schemas, difficulty, mastery, topics, retrieval, generation,
# generator) for cohesion and the project's file-size limits.
from dotenv import load_dotenv

from .schemas import PROBLEM_SCHEMA
from .difficulty import route_difficulty_by_mastery
from .generator import PracticeGenerator

# Preserve the original module-load side effect (env vars loaded at import).
load_dotenv()

__all__ = [
    "PROBLEM_SCHEMA",
    "route_difficulty_by_mastery",
    "PracticeGenerator",
]
