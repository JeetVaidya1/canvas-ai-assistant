# schemas.py - Schemas and grading constants for the exam answer judge
from typing import Any, Dict

# Schema for the AI answer judge (free-response grading).
ANSWER_JUDGE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["correct", "partial", "incorrect"]},
        "reason": {"type": "string", "description": "One or two sentences justifying the verdict."},
    },
    "required": ["verdict", "reason"],
}

# Fraction of a question's points awarded for each verdict.
VERDICT_CREDIT = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
