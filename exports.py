# exports.py — build downloadable artifacts (Notes PDF, Anki deck, study-plan iCal).
#
# Each builder is self-contained: it sources data from Supabase / course content
# and returns bytes, so the export endpoints work even though there is no stored
# planner/flashcard subsystem yet.
from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import os
import tempfile
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _supabase():
    from supabase import create_client

    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_KEY"]
    return create_client(url, key)


def _stable_id(seed: str) -> int:
    """Deterministic 31-bit id (genanki wants stable model/deck ids)."""
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) & 0x7FFFFFFF


def _course_title(course_id: str, supabase) -> str:
    try:
        res = supabase.table("courses").select("title").eq("course_id", course_id).limit(1).execute()
        if res.data:
            return res.data[0].get("title") or course_id
    except Exception:
        pass
    return course_id


def _course_context(course_id: str, supabase, max_chars: int = 12000) -> str:
    """Pull a sample of ingested chunk text to ground on-demand generation."""
    try:
        res = (
            supabase.table("embeddings")
            .select("content, doc_name, chunk_id")
            .eq("course_id", course_id)
            .order("chunk_id")
            .limit(120)
            .execute()
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Could not read course content: {exc}")

    rows = res.data or []
    if not rows:
        raise RuntimeError("No ingested content for this course yet.")

    out, total = [], 0
    for row in rows:
        piece = (row.get("content") or "").strip()
        if not piece:
            continue
        out.append(piece)
        total += len(piece)
        if total >= max_chars:
            break
    return "\n\n".join(out)


def _chat_json(prompt: str, system: str, max_tokens: int = 2000) -> Any:
    """One JSON-mode Claude call via the provider shim; returns parsed JSON."""
    from providers import make_client

    client = make_client()
    model = os.getenv("MODEL_COMPLEX", "claude-sonnet-4-6")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return json.loads(resp.choices[0].message.content)


# ---------------------------------------------------------------------------
# 1) Notes -> PDF
# ---------------------------------------------------------------------------
_PDF_CSS = """
@page { size: letter; margin: 2cm; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 11pt; color: #1a1a1a; line-height: 1.5; }
h1 { font-size: 20pt; color: #0e7490; border-bottom: 2px solid #0e7490; padding-bottom: 4px; }
h2 { font-size: 15pt; color: #155e75; margin-top: 18px; }
h3 { font-size: 12pt; color: #155e75; }
code { background: #f3f4f6; padding: 1px 4px; font-family: Courier, monospace; }
pre { background: #f3f4f6; padding: 8px; }
ul, ol { margin-left: 12px; }
hr { border: none; border-top: 1px solid #d1d5db; margin: 20px 0; }
.note-meta { color: #6b7280; font-size: 9pt; margin-bottom: 12px; }
"""


def build_notes_pdf(course_id: str) -> bytes:
    """Render a course's saved notes (or freshly generated ones) into a PDF."""
    import markdown as _md
    from xhtml2pdf import pisa

    supabase = _supabase()
    title = _course_title(course_id, supabase)

    from notes_engine import get_notes_from_db

    notes = get_notes_from_db(course_id)

    sections: List[str] = []
    if notes:
        for note in notes:
            meta = f"{note.get('word_count', 0)} words · {note.get('reading_time', '')}".strip(" ·")
            sections.append(f"# {note.get('title', 'Untitled')}\n\n_{meta}_\n\n{note.get('content', '')}")
    else:
        # No saved notes — generate a single notes doc from all course files.
        from notes_engine import generate_notes_from_files

        files = supabase.table("files").select("filename").eq("course_id", course_id).execute().data or []
        file_names = [f["filename"] for f in files]
        if not file_names:
            raise RuntimeError("This course has no files to build notes from.")
        generated = generate_notes_from_files(course_id, file_names)
        sections.append(
            f"# {generated.get('suggested_title', title + ' Notes')}\n\n{generated.get('notes', '')}"
        )

    body_html = _md.markdown("\n\n---\n\n".join(sections), extensions=["extra", "sane_lists"])
    html = (
        f"<html><head><meta charset='utf-8'><style>{_PDF_CSS}</style></head>"
        f"<body><h1>{title} — Study Notes</h1>{body_html}</body></html>"
    )

    buf = io.BytesIO()
    result = pisa.CreatePDF(src=io.StringIO(html), dest=buf, encoding="utf-8")
    if result.err:
        raise RuntimeError("PDF rendering failed.")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 2) Flashcards -> Anki .apkg
# ---------------------------------------------------------------------------
_FLASHCARD_SYSTEM = (
    "You are a study-tool generator. Produce concise, exam-relevant flashcards "
    "grounded strictly in the provided course material. Return JSON only."
)


def _generate_flashcards(course_id: str, supabase, count: int = 25) -> List[Dict[str, str]]:
    context = _course_context(course_id, supabase)
    prompt = (
        f"From the course material below, create {count} flashcards covering the most "
        f"important concepts. Return a JSON object: {{\"flashcards\": [{{\"q\": \"...\", "
        f"\"a\": \"...\"}}]}}. Questions short; answers 1-3 sentences.\n\n"
        f"COURSE MATERIAL:\n{context}"
    )
    data = _chat_json(prompt, _FLASHCARD_SYSTEM, max_tokens=3000)
    cards = data.get("flashcards") if isinstance(data, dict) else data
    cleaned = [
        {"q": str(c["q"]).strip(), "a": str(c["a"]).strip()}
        for c in (cards or [])
        if isinstance(c, dict) and c.get("q") and c.get("a")
    ]
    if not cleaned:
        raise RuntimeError("No flashcards could be generated for this course.")
    return cleaned


def build_flashcards_apkg(course_id: str) -> bytes:
    import genanki

    supabase = _supabase()
    title = _course_title(course_id, supabase)
    cards = _generate_flashcards(course_id, supabase)

    model = genanki.Model(
        _stable_id(f"model:{course_id}"),
        "Vindexa Basic",
        fields=[{"name": "Question"}, {"name": "Answer"}],
        templates=[
            {
                "name": "Card",
                "qfmt": "{{Question}}",
                "afmt": '{{FrontSide}}<hr id="answer">{{Answer}}',
            }
        ],
    )
    deck = genanki.Deck(_stable_id(f"deck:{course_id}"), f"{title} — Flashcards")
    for card in cards:
        deck.add_note(genanki.Note(model=model, fields=[card["q"], card["a"]]))

    # genanki needs a real file path (sqlite); write to a temp file then read back.
    with tempfile.NamedTemporaryFile(suffix=".apkg", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        genanki.Package(deck).write_to_file(tmp_path)
        with open(tmp_path, "rb") as fh:
            return fh.read()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# 3) Study plan -> iCal (.ics)
# ---------------------------------------------------------------------------
_PLANNER_SYSTEM = (
    "You are a study planner. Build a realistic day-by-day revision schedule grounded "
    "in the provided course material. Return JSON only."
)


def _generate_study_plan(course_id: str, supabase, days: int = 10) -> List[Dict[str, Any]]:
    context = _course_context(course_id, supabase, max_chars=8000)
    prompt = (
        f"Create a {days}-day study plan from the course material. Return JSON: "
        f"{{\"days\": [{{\"day\": 1, \"topics\": [\"...\"], \"duration_minutes\": 60, "
        f"\"type\": \"review|new|practice\"}}]}}. Order topics pedagogically.\n\n"
        f"COURSE MATERIAL:\n{context}"
    )
    data = _chat_json(prompt, _PLANNER_SYSTEM, max_tokens=2000)
    plan = data.get("days") if isinstance(data, dict) else data
    if not plan:
        raise RuntimeError("No study plan could be generated for this course.")
    return plan


def _ics_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def build_planner_ics(course_id: str) -> bytes:
    supabase = _supabase()
    title = _course_title(course_id, supabase)
    plan = _generate_study_plan(course_id, supabase)

    start = _dt.date.today()
    stamp = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Vindexa//Study Planner//EN",
        "CALSCALE:GREGORIAN",
        f"X-WR-CALNAME:{_ics_escape(title)} Study Plan",
    ]
    for idx, day in enumerate(plan):
        offset = int(day.get("day", idx + 1)) - 1
        date = start + _dt.timedelta(days=max(offset, 0))
        nxt = date + _dt.timedelta(days=1)
        topics = day.get("topics") or []
        kind = str(day.get("type", "review")).capitalize()
        mins = day.get("duration_minutes", 60)
        summary = f"{kind}: {', '.join(topics)[:60]}" if topics else f"{kind} session"
        desc = f"{title} — {kind} (~{mins} min)\\nTopics: {_ics_escape(', '.join(topics))}"
        lines += [
            "BEGIN:VEVENT",
            f"UID:{_stable_id(course_id)}-{idx}@vindexa",
            f"DTSTAMP:{stamp}",
            f"DTSTART;VALUE=DATE:{date.strftime('%Y%m%d')}",
            f"DTEND;VALUE=DATE:{nxt.strftime('%Y%m%d')}",
            f"SUMMARY:{_ics_escape(summary)}",
            f"DESCRIPTION:{desc}",
            "END:VEVENT",
        ]
    lines.append("END:VCALENDAR")
    return ("\r\n".join(lines) + "\r\n").encode("utf-8")
