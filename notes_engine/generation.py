# notes_engine/generation.py — Core notes generation, QA polish, topic/flashcard extraction
from typing import Dict, List, Any, Optional

from .config import (
    openai_client,
    MODEL_DEFAULT,
    ALLOW_GENERAL_FILL,
    INCLUDE_FLASHCARDS,
    SYSTEM_STYLE,
)
from .helpers import _is_dict, _safe_json_obj, _prettify_notes, _nice_fallback_title
from .prompts import _build_combined_context, _route, _notes_instruction, _FLASHCARD_SCHEMA
from .retrieval import extract_content_from_files


def generate_detailed_notes(content_map: Dict[str, str], topic: str = "", style: str = "detailed") -> Dict[str, Any]:
    combined, _ = _build_combined_context(content_map)
    total_words = sum(len((v or "").split()) for v in content_map.values())
    model, max_tok = _route(topic, total_words)

    prompt = f"""You are generating polished course notes.

COURSE SOURCES (primary truth set, with inline tags like [1:file:page]):
{combined}

INSTRUCTIONS:
{_notes_instruction(style, topic, ALLOW_GENERAL_FILL)}

OUTPUT:
Return only the final notes content (markdown). Keep it smooth and readable; cite where grounded.
"""

    try:
        # Draft
        r = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_STYLE},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=max_tok
        )
        draft = (r.choices[0].message.content or "").strip()

        # QA / polish pass — keep structure, fix small issues. Style-aware so we
        # don't re-impose headings on an outline or pad a summary.
        if style == "outline":
            structure_rule = "- Keep the nested-bullet outline format; do NOT add prose paragraphs or '##' headings."
        elif style == "summary":
            structure_rule = "- Keep it concise (≤ 500 words); keep the '## Overview' and '## Key Points' sections."
        else:
            structure_rule = "- Ensure section headings use '## ' and are in the specified order."
        qa_prompt = f"""Edit the notes to improve clarity and flow without changing meaning.
{structure_rule}
- Keep paragraphs short; prefer tidy bullets.
- Leave inline source tags like [1:file:page] where used.
Return the improved notes only.
NOTES:
{draft}
"""
        r2 = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise academic editor. Final answer only."},
                {"role": "user", "content": qa_prompt}
            ],
            max_completion_tokens=max_tok
        )
        notes_content = _prettify_notes((r2.choices[0].message.content or "").strip())

        # Title (force a punchy, non-generic title)
        title_prompt = f"""Propose a punchy, descriptive title (6–12 words) for these notes.
Avoid generic section names like "Big picture overview", "Overview", "Key terms", etc.
Return strict JSON only: {{"title": "..."}}.
Notes excerpt:
{notes_content[:1200]}
"""
        t = openai_client.chat.completions.create(
            model=MODEL_DEFAULT,
            messages=[{"role": "user", "content": title_prompt}],
            max_completion_tokens=40
        )
        obj = _safe_json_obj(t.choices[0].message.content or "")
        suggested_title = (obj.get("title") or "").strip()

        banned = {"big picture overview", "overview", "key terms", "core concepts", "notes", "lecture notes"}
        if not suggested_title or suggested_title.lower() in banned:
            suggested_title = _nice_fallback_title(topic, list(content_map.keys()))

        # Inject H1 title at the very top if absent
        if not notes_content.lstrip().startswith("#"):
            notes_content = f"# {suggested_title}\n\n{notes_content}"

        # Topics (lightweight JSON extraction)
        topics = extract_topics_from_content(notes_content)

        word_count = len(notes_content.split())
        reading_time = f"{max(1, word_count // 200)} min"

        result: Dict[str, Any] = {
            "notes": notes_content,
            "suggested_title": suggested_title,
            "topics": topics[:8],
            "word_count": word_count,
            "reading_time": reading_time,
            "source_files": list(content_map.keys()),
        }

        if INCLUDE_FLASHCARDS:
            fc = _generate_flashcards(notes_content)
            if fc:
                result["flashcards"] = fc[:20]

        return result

    except Exception as e:
        print(f"❌ Notes generation failed: {e}")
        return {
            "notes": "Failed to generate notes. Please try again with different source materials.",
            "suggested_title": f"Error — Notes Generation Failed{': ' + topic if topic else ''}",
            "topics": [],
            "word_count": 0,
            "reading_time": "0 min",
            "source_files": list(content_map.keys())
        }

# ─────────────────────────────────────────────────────────────────────────────
# Topic + flashcard extraction
# ─────────────────────────────────────────────────────────────────────────────
def extract_topics_from_content(content: str) -> List[str]:
    try:
        base_prompt = """Extract 5–8 key topics from this content.
Return JSON only: {"topics": ["topic1","topic2","..."]}."""
        r = openai_client.chat.completions.create(
            model=MODEL_DEFAULT,
            messages=[{"role": "user", "content": f"{base_prompt}\n\nContent:\n{content[:2000]}"}],
            max_completion_tokens=120
        )
        obj = _safe_json_obj(r.choices[0].message.content or "")
        topics = obj.get("topics", [])
        topics = [t.strip() for t in topics if isinstance(t, str) and t.strip()]
        return topics[:8] if topics else ["General Topics"]
    except Exception as e:
        print(f"❌ Topic extraction failed: {e}")
        words = (content or "").lower().split()
        seeds = ['algorithm','data structures','function','method','process',
                 'theory','principle','theorem','model','equation','experiment']
        return [s.title() for s in seeds if s in " ".join(words)][:6] or ["General Topics"]


def _generate_flashcards(notes: str) -> Optional[List[Dict[str, str]]]:
    """Generate Q/A flashcards from notes via guaranteed-schema tool use (no regex)."""
    from providers import structured_call

    try:
        out = structured_call(
            [{
                "role": "user",
                "content": (
                    "Create 10 concise exam-relevant flashcards from these study notes. "
                    "Questions short; answers 1-3 sentences. Plain text, no markdown.\n\n"
                    f"NOTES:\n{notes[:8000]}"
                ),
            }],
            schema=_FLASHCARD_SCHEMA,
            tool_name="flashcards",
            model=MODEL_DEFAULT,
            max_tokens=2000,
        )
        cards = out.get("flashcards") if isinstance(out, dict) else None
        clean = [
            {"q": str(c["q"]).strip(), "a": str(c["a"]).strip()}
            for c in (cards or [])
            if _is_dict(c) and c.get("q") and c.get("a")
        ]
        return clean or None
    except Exception as e:
        print(f"❌ Flashcard generation failed: {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def generate_notes_from_files(course_id: str, file_names: List[str], topic: str = "", style: str = "detailed") -> Dict[str, Any]:
    try:
        print(f"📝 Generating {style} notes for {len(file_names)} files")
        print(f"📚 Files: {', '.join(file_names)}")
        if topic:
            print(f"🎯 Topic focus: {topic}")

        content = extract_content_from_files(course_id, file_names)

        if not content or all(not (v or "").strip() or str(v).startswith("Content not found") for v in content.values()):
            return {
                "status": "error",
                "message": "No content found in selected files",
                "notes": "Unable to generate notes: No content could be extracted from the selected files. Please ensure the files contain readable text content.",
                "suggested_title": "Error — No Content Found",
                "topics": [],
                "word_count": 0,
                "reading_time": "0 min",
                "source_files": file_names
            }

        result = generate_detailed_notes(content, topic, style)
        result["status"] = "success"
        print(f"✅ Generated {result['word_count']} word notes")
        return result

    except Exception as e:
        print(f"❌ Notes generation error: {e}")
        import traceback; traceback.print_exc()
        return {
            "status": "error",
            "message": str(e),
            "notes": "An error occurred while generating notes. Please try again.",
            "suggested_title": "Error — Generation Failed",
            "topics": [],
            "word_count": 0,
            "reading_time": "0 min",
            "source_files": file_names
        }
