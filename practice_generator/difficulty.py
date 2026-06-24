"""Difficulty routing and time-estimate helpers for practice generation."""


def route_difficulty_by_mastery(mastery: float) -> str:
    """Adaptive difficulty: weaker topics get easier problems, mastered ones harder."""
    if mastery < 0.5:
        return "easy"
    if mastery <= 0.8:
        return "medium"
    return "hard"
