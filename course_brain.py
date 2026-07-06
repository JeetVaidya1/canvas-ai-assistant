"""Course Brain — content-grounded topics that everything else keys on.

Replaces the legacy filename-regex topic extraction: topics are synthesized
per rebuild by sampling real chunk content per document into ONE
schema-guaranteed LLM call, then persisted to ``course_topics`` (migration
0012). Consumers read via :func:`get_topics` and bridge legacy free-text
mastery labels via the matching helpers below.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from supabase import create_client

from providers import structured_call

load_dotenv()
_supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

logger = logging.getLogger(__name__)

# Synthesis targets: ask the model for 8-15 topics; accept whatever survives cleaning.
MIN_TOPICS = 8
MAX_TOPICS = 15

# Content sampling bounds (keeps the digest a single bounded LLM call).
_MAX_DOCS = 12
_SAMPLES_PER_DOC = 3          # first, middle, last chunk of each document
_MAX_CHARS_PER_SAMPLE = 700

# Words too generic to count as a topic-identifying token when bridging labels.
_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "into", "part", "chapter", "week",
    "unit", "lecture", "lesson", "section", "module", "intro", "introduction",
    "overview", "notes", "basics", "basic", "course", "topics", "topic",
})


@dataclass(frozen=True)
class Topic:
    """One synthesized course topic (immutable)."""
    slug: str
    name: str
    description: str = ""
    doc_coverage: Tuple[Dict[str, Any], ...] = ()
    prereq_slugs: Tuple[str, ...] = ()
    position: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slug": self.slug,
            "name": self.name,
            "description": self.description,
            "doc_coverage": [dict(d) for d in self.doc_coverage],
            "prereq_slugs": list(self.prereq_slugs),
            "position": self.position,
        }


# ---- normalization / matching helpers --------------------------------------

def _norm(value: str) -> str:
    return (value or "").strip().casefold()


def slugify(name: str) -> str:
    """Kebab-case a topic name: 'Binary Search Trees' -> 'binary-search-trees'."""
    slug = re.sub(r"[^a-z0-9]+", "-", _norm(name))
    return slug.strip("-")


def _tokens(value: str) -> frozenset:
    """Significant tokens of a label (for bridging legacy mastery strings)."""
    words = re.findall(r"[a-z0-9]+", _norm(value))
    return frozenset(w for w in words if len(w) >= 3 and not w.isdigit()
                     and w not in _STOPWORDS)


def match_topic(name_or_label: str, topics: Sequence[Topic]) -> Optional[Topic]:
    """Match a free-text label to a Topic: exact name -> slug -> substring.

    All comparisons are on casefolded/stripped strings. Substring runs both
    ways ('Trees' matches 'Binary Search Trees' and vice versa); the most
    specific (longest-named) candidate wins.
    """
    query = _norm(name_or_label)
    if not query or not topics:
        return None
    for topic in topics:                                   # 1. exact name
        if _norm(topic.name) == query:
            return topic
    query_slug = slugify(name_or_label)
    for topic in topics:                                   # 2. slug
        if topic.slug == query_slug:
            return topic
    candidates = [t for t in topics                        # 3. substring both ways
                  if _norm(t.name) and (_norm(t.name) in query or query in _norm(t.name))]
    if candidates:
        return max(candidates, key=lambda t: len(t.name))
    return None


def match_mastery(topic: str, mastery: Dict[str, float]) -> Optional[float]:
    """Best mastery value for a topic label, bridging legacy label spellings.

    Tiers: exact (casefold) -> substring both ways -> shared significant token.
    The token tier is what lets old filename-era rows ('301 3 Excel') keep
    matching new Course Brain names ('Excel Formulas'). Returns None when
    nothing plausibly matches.
    """
    query = _norm(topic)
    if not query:
        return None
    normalized = {(_norm(k)): v for k, v in mastery.items() if _norm(k)}
    if query in normalized:
        return normalized[query]
    substr = [(k, v) for k, v in normalized.items() if k in query or query in k]
    if substr:
        return max(substr, key=lambda kv: len(kv[0]))[1]
    query_tokens = _tokens(topic)
    if query_tokens:
        scored = [(len(query_tokens & _tokens(k)), len(k), v)
                  for k, v in normalized.items() if query_tokens & _tokens(k)]
        if scored:
            return max(scored)[2]
    return None


def match_mastery_rows(topic: str, rows: Iterable[Dict[str, Any]]) -> Optional[float]:
    """Best ``mastery_level`` from learning_progress rows for a topic label."""
    mastery: Dict[str, float] = {}
    for row in rows or []:
        label = row.get("topic")
        if label and row.get("mastery_level") is not None:
            mastery[str(label)] = float(row["mastery_level"])
    return match_mastery(topic, mastery)


# ---- content sampling (plain metadata reads; no dummy-embedding queries) ----

def _list_docs(course_id: str) -> List[str]:
    rows = (_supabase.table("files").select("filename")
            .eq("course_id", course_id).execute().data or [])
    docs = [r["filename"] for r in rows if r.get("filename")]
    return docs[:_MAX_DOCS]


def _doc_chunks(course_id: str, doc_name: str) -> List[Dict[str, Any]]:
    rows = (_supabase.table("embeddings")
            .select("chunk_id, content, page, slide")
            .eq("course_id", course_id).eq("doc_name", doc_name)
            .execute().data or [])
    return sorted(rows, key=lambda r: r.get("chunk_id") or 0)


def _sample_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """First, middle, and last chunk — representative without reading everything."""
    if len(chunks) <= _SAMPLES_PER_DOC:
        return chunks
    indices = sorted({0, len(chunks) // 2, len(chunks) - 1})
    return [chunks[i] for i in indices]


def _page_range(chunks: List[Dict[str, Any]]) -> List[int]:
    pages = [r["page"] for r in chunks if isinstance(r.get("page"), int)]
    return [min(pages), max(pages)] if pages else []


def _build_digest(course_id: str) -> Tuple[str, Dict[str, List[int]]]:
    """Return (content digest for the LLM, observed page range per document)."""
    sections: List[str] = []
    doc_pages: Dict[str, List[int]] = {}
    for doc in _list_docs(course_id):
        chunks = _doc_chunks(course_id, doc)
        if not chunks:
            continue
        doc_pages[doc] = _page_range(chunks)
        samples = []
        for row in _sample_chunks(chunks):
            content = (row.get("content") or "").strip()
            if content:
                where = f"p.{row['page']}" if row.get("page") else f"chunk {row.get('chunk_id')}"
                samples.append(f"({where}) {content[:_MAX_CHARS_PER_SAMPLE]}")
        if samples:
            pages = doc_pages[doc]
            span = f" (pages {pages[0]}-{pages[1]})" if pages else ""
            sections.append(f"=== {doc}{span} ===\n" + "\n".join(samples))
    return "\n\n".join(sections), doc_pages


# ---- synthesis (one structured LLM call) + persistence ----------------------

TOPIC_SYNTHESIS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "topics": {
            "type": "array",
            "description": f"{MIN_TOPICS}-{MAX_TOPICS} core course topics, teaching order.",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Clean human title-case name, 1-4 words. No lecture/file numbering."},
                    "slug": {"type": "string", "description": "kebab-case of the name."},
                    "description": {"type": "string", "description": "Exactly one sentence describing the topic."},
                    "doc_coverage": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "doc": {"type": "string", "description": "A document filename from the provided list."},
                                "pages": {"type": "array", "items": {"type": "integer"}, "description": "[first_page, last_page] covering the topic, when known."},
                            },
                            "required": ["doc"],
                        },
                    },
                    "prereq_slugs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Slugs FROM THIS TOPIC SET that must be understood first. Keep acyclic.",
                    },
                    "position": {"type": "integer", "description": "0-based teaching order."},
                },
                "required": ["name", "slug", "description", "position"],
            },
        },
    },
    "required": ["topics"],
}


def _clean_coverage(raw: Any, doc_pages: Dict[str, List[int]]) -> Tuple[Dict[str, Any], ...]:
    """Keep only coverage entries naming real documents; backfill page ranges."""
    coverage: List[Dict[str, Any]] = []
    for entry in raw if isinstance(raw, list) else []:
        if not isinstance(entry, dict):
            continue
        doc = str(entry.get("doc") or "").strip()
        if doc not in doc_pages:
            continue
        pages = entry.get("pages")
        if not (isinstance(pages, list) and len(pages) == 2
                and all(isinstance(p, int) for p in pages)):
            pages = doc_pages[doc]
        if not any(c["doc"] == doc for c in coverage):
            coverage.append({"doc": doc, "pages": list(pages)})
    return tuple(coverage)


def _clean_topics(raw_topics: List[Dict[str, Any]],
                  doc_pages: Dict[str, List[int]]) -> List[Topic]:
    """Validate/normalize LLM output: dedupe slugs, scope prereqs, fix order."""
    staged: List[Dict[str, Any]] = []
    seen_slugs = set()
    for raw in raw_topics[:MAX_TOPICS]:
        if not isinstance(raw, dict):
            continue
        name = re.sub(r"\s+", " ", str(raw.get("name") or "").strip())
        slug = slugify(str(raw.get("slug") or "")) or slugify(name)
        if not name or not slug or slug in seen_slugs:
            continue
        seen_slugs.add(slug)
        try:
            position = int(raw.get("position", len(staged)))
        except (TypeError, ValueError):
            position = len(staged)
        staged.append({
            "slug": slug,
            "name": name,
            "description": str(raw.get("description") or "").strip(),
            "doc_coverage": _clean_coverage(raw.get("doc_coverage"), doc_pages),
            "raw_prereqs": [slugify(str(p)) for p in (raw.get("prereq_slugs") or [])],
            "position": position,
        })
    valid_slugs = {t["slug"] for t in staged}
    staged.sort(key=lambda t: t["position"])
    return [
        Topic(
            slug=t["slug"],
            name=t["name"],
            description=t["description"],
            doc_coverage=t["doc_coverage"],
            prereq_slugs=tuple(p for p in dict.fromkeys(t["raw_prereqs"])
                               if p in valid_slugs and p != t["slug"]),
            position=index,
        )
        for index, t in enumerate(staged)
    ]


def _persist_topics(course_id: str, topics: List[Topic]) -> None:
    """Idempotent rebuild: replace the course's rows wholesale."""
    _supabase.table("course_topics").delete().eq("course_id", course_id).execute()
    for topic in topics:
        _supabase.table("course_topics").insert({
            "course_id": course_id,
            **topic.to_dict(),
        }).execute()


def synthesize_topics(course_id: str) -> List[Topic]:
    """Synthesize, persist, and return the course's topics (one LLM call).

    Returns [] (persisting nothing) when the course has no ingested content;
    raises on LLM/synthesis failure so callers never cache garbage.
    """
    digest, doc_pages = _build_digest(course_id)
    if not digest:
        logger.info("Course Brain: no ingested content for course %s; skipping synthesis", course_id)
        return []

    doc_list = "\n".join(
        f"- {doc}" + (f" (pages {p[0]}-{p[1]})" if p else "")
        for doc, p in doc_pages.items()
    )
    prompt = (
        "You are building the 'Course Brain' for a study app: the definitive "
        f"topic map of one course. From the sampled materials below, synthesize "
        f"the {MIN_TOPICS}-{MAX_TOPICS} core topics a student must master.\n\n"
        "RULES:\n"
        "- name: clean human title-case, 1-4 words, real subject matter — never "
        "file numbering, lecture prefixes, or generic labels like 'Overview'.\n"
        "- slug: kebab-case of the name.\n"
        "- description: exactly one sentence a student would understand.\n"
        "- doc_coverage: which of the documents listed below teach the topic, "
        "with [first_page, last_page] from the samples where visible.\n"
        "- prereq_slugs: slugs from THIS topic set only that must come first; "
        "keep the graph acyclic and only add edges grounded in the material.\n"
        "- position: 0-based teaching order (foundations first).\n\n"
        f"DOCUMENTS:\n{doc_list}\n\n"
        f"CONTENT SAMPLES:\n{digest}"
    )
    out = structured_call(
        [{"role": "user", "content": prompt}],
        schema=TOPIC_SYNTHESIS_SCHEMA,
        tool_name="course_topics",
        model=os.getenv("MODEL_COMPLEX"),
        max_tokens=4000,
    )
    topics = _clean_topics(out.get("topics") or [], doc_pages)
    if not topics:
        raise RuntimeError(f"Topic synthesis produced no usable topics for course {course_id}")
    _persist_topics(course_id, topics)
    logger.info("Course Brain: synthesized %d topics for course %s", len(topics), course_id)
    return topics


def rebuild_topics_safely(course_id: str) -> None:
    """Background-task wrapper: rebuild after ingest, never raise."""
    try:
        synthesize_topics(course_id)
    except Exception:  # noqa: BLE001 — background task must not crash the app
        logger.exception("Course Brain background rebuild failed for course %s", course_id)


# ---- reads -------------------------------------------------------------------

def _row_to_topic(row: Dict[str, Any]) -> Topic:
    coverage = row.get("doc_coverage") or []
    prereqs = row.get("prereq_slugs") or []
    return Topic(
        slug=str(row.get("slug") or ""),
        name=str(row.get("name") or ""),
        description=str(row.get("description") or ""),
        doc_coverage=tuple(dict(c) for c in coverage if isinstance(c, dict)),
        prereq_slugs=tuple(str(p) for p in prereqs),
        position=int(row.get("position") or 0),
    )


def get_topics(course_id: str, auto_generate: bool = True) -> List[Topic]:
    """Read the course's topics; synthesize on first call when empty.
    Never raises: returns [] when there is no content or synthesis fails,
    so consumers can apply their own last-resort fallbacks."""
    try:
        rows = (_supabase.table("course_topics").select("*")
                .eq("course_id", course_id).execute().data or [])
    except Exception:  # noqa: BLE001
        logger.exception("Course Brain: topic read failed for course %s", course_id)
        rows = []
    if rows:
        topics = [_row_to_topic(r) for r in rows]
        return sorted((t for t in topics if t.slug and t.name),
                      key=lambda t: t.position)
    if not auto_generate:
        return []
    try:
        return synthesize_topics(course_id)
    except Exception:  # noqa: BLE001
        logger.exception("Course Brain: auto-synthesis failed for course %s", course_id)
        return []


def topic_names(course_id: str, auto_generate: bool = True) -> List[str]:
    """Convenience: ordered topic names (the shape legacy consumers expect)."""
    return [t.name for t in get_topics(course_id, auto_generate=auto_generate)]
