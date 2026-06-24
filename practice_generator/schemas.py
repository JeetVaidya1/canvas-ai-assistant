"""Structured-output schema for practice problem generation.

Defines the JSON schema used for guaranteed-structure tool-use calls so the
model returns parseable problem objects with no regex/JSON text parsing.
"""

# Schema for guaranteed-structure practice problem generation (no regex parsing).
PROBLEM_SCHEMA = {
    "type": "object",
    "properties": {
        "problems": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": "Four options, each prefixed 'A) '..'D) '.",
                    },
                    "correct_answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
                    "explanation": {"type": "string"},
                    "estimated_time": {"type": "string"},
                },
                "required": ["question", "options", "correct_answer", "explanation"],
            },
        }
    },
    "required": ["problems"],
}
