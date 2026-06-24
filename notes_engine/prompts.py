# notes_engine/prompts.py — Prompt builders, routing, and the flashcard schema
from typing import Dict, List, Tuple

from .config import (
    MODEL_DEFAULT,
    MODEL_COMPLEX,
    MAX_TOK_DEFAULT,
    MAX_TOK_COMPLEX,
)


def _build_combined_context(contents: Dict[str, str]) -> Tuple[str, List[str]]:
    if not contents:
        return "", []
    parts, tags = [], []
    for idx, (fname, text) in enumerate(contents.items(), 1):
        tag = f"[{idx}:{fname}]"
        tags.append(tag)
        parts.append(f"=== SOURCE {idx} {tag} ===\n{text}")
    return "\n\n".join(parts), tags

def _route(topic: str, total_words: int) -> Tuple[str, int]:
    complex_hint = any(k in (topic or "").lower() for k in [
        "proof","derive","thermodynamics","quantum","fourier","optimization","time complexity",
        "electromagnetism","organic synthesis","microeconomics","statistics","graph theory"
    ])
    long_doc = total_words > 8000
    if complex_hint or long_doc:
        return MODEL_COMPLEX, MAX_TOK_COMPLEX
    return MODEL_DEFAULT, MAX_TOK_DEFAULT

def _notes_instruction(style: str, topic: str, allow_gn: bool) -> str:
    focus = f"\nFOCUS: Give extra attention to “{topic}” where relevant." if topic else ""
    gn = ("If essential detail is missing, you may add brief general knowledge seamlessly (no label), "
          "but prefer course sources and do not invent citations.") if allow_gn else \
         ("If a detail is missing from sources, say so briefly; do not add general knowledge.")

    common_rules = f"""Rules:
- Prefer COURSE SOURCES; when you rely on them, cite inline like [1:file:page].
- {gn}
- No headings like “From your course”; just write the notes.
- No chain-of-thought.
{focus}"""

    if style == "summary":
        body = """Write a concise summary (≤ 500 words) a student can skim before an exam.

Structure:
## Overview
3–5 sentences on what this topic is and why it matters.
## Key Points
6–10 tight bullets covering the most important facts, definitions, and results
(cite with [i:file:page] when grounded).

Keep it short and high-signal; omit worked examples, mnemonics, and study plans."""
    elif style == "outline":
        body = """Write a structured outline using nested bullets only (no prose paragraphs).

- Use markdown nested bullets (indent with two spaces per level), 2–3 levels deep.
- Top-level bullets are the major topics; sub-bullets are key terms, facts, and results.
- Cite grounded points inline like [i:file:page].
- Be comprehensive but terse — fragments, not sentences. No worked examples or study plans."""
    else:  # "detailed" (default) — the full 11-section format
        body = """Write clean, exam-ready lecture notes a top student would keep.

Sections (in order, with concise content):
1. Overview — 3–5 bullets: what this topic is and why it matters.
2. Key terms — short, precise definitions (cite with [i:file:page] when grounded).
3. Core ideas — short paragraphs or bullets; integrate citations naturally.
4. Worked example(s) — compact, stepwise; emphasize why each step.
5. Figures described — describe any slide diagrams in words.
6. Formula box — list important equations; define symbols.
7. Pitfalls — common mistakes and how to avoid them.
8. Connections — how this ties to other course topics.
9. Mini Q&A — 3–5 short exam-style questions with crisp answers.
10. Mnemonics — a couple memory hooks.
11. Quick study plans — 30 / 60 / 120 min.

Keep tone friendly and precise; short paragraphs; tidy bullets.
Use markdown headings (##) for sections; bullets where helpful; avoid walls of text."""

    return f"\n{body}\n\n{common_rules}\n"


_FLASHCARD_SCHEMA = {
    "type": "object",
    "properties": {
        "flashcards": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"q": {"type": "string"}, "a": {"type": "string"}},
                "required": ["q", "a"],
            },
        }
    },
    "required": ["flashcards"],
}
