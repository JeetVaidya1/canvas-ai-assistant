# notes_engine.py — Conversational, polished notes (RAG + QA polish)
import os
import uuid
import json
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from vector_store import VectorStore

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
vector_store = VectorStore()

# ── Config (env-overridable) ─────────────────────────────────────────────────
MODEL_DEFAULT      = os.getenv("MODEL_DEFAULT", "gpt-5-mini")
MODEL_COMPLEX      = os.getenv("MODEL_COMPLEX", "gpt-5")
EMBED_MODEL        = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-large")
RAG_TOP_K_FILE     = int(os.getenv("RAG_TOP_K_FILE", "60"))   # chunks per file
MAX_TOK_DEFAULT    = int(os.getenv("MAX_TOKENS_DEFAULT", "5000"))
MAX_TOK_COMPLEX    = int(os.getenv("MAX_TOKENS_COMPLEX", "7000"))
ALLOW_GENERAL_FILL = os.getenv("ALLOW_GENERAL", "true").lower() == "true"
INCLUDE_FLASHCARDS = os.getenv("INCLUDE_FLASHCARDS", "true").lower() == "true"

# Friendly, modern tone
SYSTEM_STYLE = (
    "You are a friendly, sharp university note-taker. Write like a great TA: "
    "clear, compact, and conversational—not stiff. Keep paragraphs short, "
    "use tidy bullets where helpful, and avoid filler. No chain-of-thought."
)

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def _is_dict(x) -> bool:
    return isinstance(x, dict)

def _safe_get(d: dict, k: str, default=None):
    try:
        return d.get(k, default)
    except Exception:
        return default

def _safe_json_obj(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                return json.loads(text[start:end+1])
        except Exception:
            return {}
    return {}

def _prettify_notes(notes: str) -> str:
    import re
    notes = re.sub(r'\n{3,}', '\n\n', notes)
    notes = re.sub(r'\*\*([^*]+)\*\*', r'\1', notes)
    return notes.strip()

def _nice_fallback_title(topic: str, files: List[str]) -> str:
    base = (topic or "").strip()
    if not base:
        cleaned = [f.rsplit(".",1)[0].replace("_"," ").replace("-", " ").title() for f in files[:2]]
        base = " • ".join(cleaned) if cleaned else "Course Notes"
    return f"{base} — Concepts, Implementations & Pitfalls"[:120]

# ─────────────────────────────────────────────────────────────────────────────
# Retrieval helpers
# ─────────────────────────────────────────────────────────────────────────────
def _sanitize_chunks(raw: Any, file_name: str = "") -> List[Dict[str, Any]]:
    """Filter out None/invalid rows and rows with empty content."""
    if not raw:
        return []
    clean: List[Dict[str, Any]] = []
    for c in raw:
        if not _is_dict(c):
            continue
        # Optional: if a file filter is intended, enforce it here too
        if file_name:
            doc = _safe_get(c, "doc_name") or _safe_get(c, "file") or ""
            if doc != file_name:
                # keep only exact file matches when we're in filename mode
                continue
        text = (_safe_get(c, "content") or "").strip()
        if not text:
            continue
        clean.append(c)
    return clean

def _get_file_chunks_by_meta(course_id: str, file_name: str) -> List[Dict[str, Any]]:
    try:
        if hasattr(vector_store, "query_by_metadata"):
            raw = vector_store.query_by_metadata(
                course_id,
                filters={"doc_name": file_name},
                top_k=RAG_TOP_K_FILE
            ) or []
            chunks = _sanitize_chunks(raw)  # metadata route shouldn't need filename check
        else:
            emb = openai_client.embeddings.create(model=EMBED_MODEL, input=[file_name])
            raw = vector_store.query(course_id, emb.data[0].embedding, top_k=RAG_TOP_K_FILE) or []
            chunks = _sanitize_chunks(raw, file_name=file_name)

        def _sort_key(c):
            return (
                _safe_get(c, "page") or _safe_get(c, "page_num") or 0,
                _safe_get(c, "chunk_id") or _safe_get(c, "index") or 0
            )
        chunks.sort(key=_sort_key)
        return chunks
    except Exception as e:
        print(f"❌ Retrieval failed for {file_name}: {e}")
        return []

def extract_content_from_files(course_id: str, file_names: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for fname in file_names:
        chunks = _get_file_chunks_by_meta(course_id, fname) or []
        if not chunks:
            out[fname] = f"Content not found for {fname}"
            continue
        texts: List[str] = []
        for c in chunks:
            if not _is_dict(c):
                continue
            t = (_safe_get(c, "content") or "").strip()
            if t:
                texts.append(t)
        out[fname] = "\n\n".join(texts) if texts else f"Content not found for {fname}"
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────
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
    return f"""
Write clean, exam-ready lecture notes a top student would keep.

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

At the very end, include: FLASHCARDS (JSON) with 10 items: [{{"q": "...","a":"..."}}, ...].

Rules:
- Prefer COURSE SOURCES; when you rely on them, cite inline like [1:file:page].
- {gn}
- Keep tone friendly and precise; short paragraphs; tidy bullets.
- No headings like “From your course”; just write the notes.
- No chain-of-thought.

Formatting:
- Use markdown headings (##) for sections; bullets where helpful; avoid walls of text.
{focus}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Core generation + QA polish
# ─────────────────────────────────────────────────────────────────────────────
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

        # QA / polish pass — keep structure, fix small issues
        qa_prompt = f"""Edit the notes to improve clarity and flow without changing meaning.
- Ensure section headings use '## ' and are in the specified order.
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
            fc = _extract_flashcards(notes_content)
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

def _extract_flashcards(notes: str) -> Optional[List[Dict[str, str]]]:
    import re
    m = re.search(r'FLASHCARDS\s*\(JSON\)\s*:\s*```?\s*json\s*(\[[\s\S]+?\])\s*```?', notes, re.IGNORECASE)
    if not m:
        m = re.search(r'FLASHCARDS\s*\(JSON\)\s*:\s*(\[[\s\S]+?\])', notes, re.IGNORECASE)
    if not m:
        return None
    try:
        arr = json.loads(m.group(1))
        clean = []
        for item in arr:
            if not _is_dict(item):
                continue
            q = (item.get("q") or "").strip()
            a = (item.get("a") or "").strip()
            if q and a:
                clean.append({"q": q, "a": a})
        return clean or None
    except Exception:
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

# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────
def save_notes_to_db(course_id: str, title: str, content: str, source_files: List[str],
                     topic: str = "", note_id: str = None) -> Dict[str, Any]:
    try:
        from supabase import create_client
        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_KEY")
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

        topics = extract_topics_from_content(content)
        word_count = len((content or "").split())
        reading_time = f"{max(1, word_count // 200)} min"

        note_data = {
            "course_id": course_id,
            "title": title,
            "content": content,
            "source_files": source_files,
            "topic_focus": topic,
            "topics": topics[:8],
            "word_count": word_count,
            "reading_time": reading_time,
            "updated_at": datetime.utcnow().isoformat()
        }

        if note_id:
            result = supabase.table("notes").update(note_data).eq("id", note_id).execute()
            saved_note = result.data[0] if result.data else None
        else:
            note_data["id"] = str(uuid.uuid4())
            note_data["created_at"] = datetime.utcnow().isoformat()
            result = supabase.table("notes").insert(note_data).execute()
            saved_note = result.data[0] if result.data else None

        if saved_note:
            return {"status": "success", "note": saved_note}
        else:
            return {"status": "error", "message": "Failed to save note to database"}

    except Exception as e:
        print(f"❌ Note saving error: {e}")
        return {"status": "error", "message": f"Database error: {str(e)}"}

def get_notes_from_db(course_id: str) -> List[Dict[str, Any]]:
    try:
        from supabase import create_client
        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_KEY")
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        res = supabase.table("notes").select("*").eq("course_id", course_id).order("updated_at", desc=True).execute()
        return res.data if res.data else []
    except Exception as e:
        print(f"❌ Notes retrieval error: {e}")
        return []

def delete_note_from_db(note_id: str) -> bool:
    try:
        from supabase import create_client
        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_KEY")
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        res = supabase.table("notes").delete().eq("id", note_id).execute()
        return len(res.data) > 0 if res.data else True
    except Exception as e:
        print(f"❌ Note deletion error: {e}")
        return False
