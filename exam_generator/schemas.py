# exam_generator/schemas.py - Shared difficulty guidance and structured-output schema
from typing import Any, Dict

# Difficulty guidance shared by exam generation (mirrors quiz/practice wording).
EXAM_DIFFICULTY_SPECS: Dict[str, str] = {
    "easy": "Recall, definitions, and single-concept understanding (Bloom: Remember/Understand).",
    "medium": "Application and analysis with multi-step reasoning across connected concepts (Bloom: Apply/Analyze).",
    "hard": "Synthesis, evaluation, rigor, edge cases, and formal/asymptotic analysis (Bloom: Evaluate/Create).",
}

# Schema for guaranteed-structure exam question generation. Supports both
# multiple-choice (with options) and free-response (calculation/short_answer/
# essay/proof/diagram) question types.
EXAM_QUESTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["multiple_choice", "calculation", "short_answer", "essay", "proof", "diagram"],
                    },
                    "question": {"type": "string", "description": "Complete, self-contained question text."},
                    "options": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "For multiple_choice only: four options prefixed 'A) '..'D) '. Null otherwise.",
                    },
                    "correct_answer": {
                        "type": "string",
                        "description": "For multiple_choice: the letter (A/B/C/D). Otherwise a short final answer or solution outline.",
                    },
                    "explanation": {"type": "string", "description": "Concise key reasoning (no hidden chain-of-thought)."},
                    "points": {"type": "integer"},
                    "time_estimate": {"type": "integer", "description": "Estimated minutes."},
                    "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
                    "topic": {"type": "string", "description": "Specific topic area."},
                },
                "required": ["type", "question", "correct_answer", "explanation", "points", "topic"],
            },
        }
    },
    "required": ["questions"],
}
