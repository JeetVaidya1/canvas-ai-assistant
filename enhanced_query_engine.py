# enhanced_query_engine.py — Hybrid RAG + GPT-5 Augmentation (world-class study mode)

import os
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from vector_store import VectorStore
from response_formatter import format_ai_response  # your post-formatter

# ── Init ──────────────────────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)
vector_store = VectorStore()

# ── Config (env-overridable) ─────────────────────────────────────────────────
MODEL_DEFAULT   = os.getenv("MODEL_DEFAULT", "gpt-5-mini")
MODEL_COMPLEX   = os.getenv("MODEL_COMPLEX", "gpt-5")
EMBED_MODEL     = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-large")
MAX_TOKENS_DEF  = int(os.getenv("MAX_TOKENS_DEFAULT", "3500"))
MAX_TOKENS_CX   = int(os.getenv("MAX_TOKENS_COMPLEX", "5500"))
TOP_K           = int(os.getenv("RAG_TOP_K", "8"))
ALLOW_GENERAL   = os.getenv("ALLOW_GENERAL", "true").lower() == "true"  # enable augmentation

# ── Lightweight classifier ───────────────────────────────────────────────────
def classify_question_simple(q: str) -> str:
    ql = q.lower()
    if any(w in ql for w in ['what is', 'define', 'definition', 'meaning']): return 'definition'
    if any(w in ql for w in ['compare', 'difference', 'versus', 'vs']):      return 'comparison'
    if any(w in ql for w in ['example', 'show me', 'demonstrate']):          return 'example'
    if any(w in ql for w in ['apply', 'use', 'implement', 'prove', 'derive', 'solve']): return 'application'
    if any(w in ql for w in ['how', 'explain', 'why', 'process']):           return 'explanation'
    return 'explanation'

# ── Prompt builders ──────────────────────────────────────────────────────────
def build_context(chunks: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """Create compact, labeled context and keep a map for inline source tags."""
    text_chunks, image_chunks, sources = [], [], []
    for i, r in enumerate(chunks):
        c = r.get("content", "")
        src = {
            "id": r.get("id") or f"src{i+1}",
            "file": r.get("file") or r.get("source") or "unknown",
            "page": r.get("page") or r.get("page_num") or None
        }
        sources.append(src)
        if "[IMAGE CONTENT" in c:
            image_chunks.append((i, c.strip(), src))
        else:
            text_chunks.append((i, c.strip(), src))

    context_parts = []
    if text_chunks:
        context_parts.append("=== TEXT FROM COURSE MATERIALS ===")
        for rank, (i, c, src) in enumerate(text_chunks[:5], 1):
            tag = f"[{rank}:{src['file']}{':' + str(src['page']) if src['page'] else ''}]"
            context_parts.append(f"Text {rank} {tag}: {c}")
    if image_chunks:
        context_parts.append("\n=== VISUAL CONTENT FROM COURSE MATERIALS ===")
        for rank, (i, c, src) in enumerate(image_chunks[:3], 1):
            tag = f"[V{rank}:{src['file']}{':' + str(src['page']) if src['page'] else ''}]"
            context_parts.append(f"Visual {rank} {tag}: {c}")

    return "\n".join(context_parts), sources

SYSTEM_STYLE = (
    "You are a world-class university tutor. Provide a single, polished final answer only—"
    "no hidden steps, no chain-of-thought. Prefer clarity over verbosity."
)

def user_prompt(question: str, qtype: str, context: str, allow_general: bool) -> str:
    """Hybrid grounding: prefer course; optionally augment if coverage is weak."""
    return f"""
STUDENT QUESTION:
{question}

COURSE CONTEXT (primary source of truth):
{context}

INSTRUCTIONS:
- First, answer strictly using COURSE CONTEXT. Cite inline with the provided tags like [1:file:page].
- If essential info is missing AND augmentation is allowed: seamlessly fill gaps with general knowledge (mark those lines with (GN)).
- Keep a smooth, conversational tone. Use short paragraphs and bullets when helpful.
- Include concrete examples if useful.
- Provide a brief 'Check your understanding' item if appropriate.

OUTPUT SECTIONS (use these exact headings if both appear):
1) From your course
2) General knowledge fill-ins (only if used)

CONSTRAINTS:
- Do NOT fabricate citations. Only use inline tags from the context for grounded claims.
- If you used general knowledge, add a 1–2 line note on what was missing from the course files.

AIM:
- Accurate, helpful, and easy to read.
- Final answer only (no scratch work).
"""

# ── Routing ──────────────────────────────────────────────────────────────────
def _route(question: str, qtype: str):
    long_query = len(question) > 240
    complex_hint = any(k in question.lower() for k in [
        "prove","derive","multi-step","step by step","design","optimize","time complexity",
        "algorithm","diagram","equation","trade-off","wavefunction","fourier","big-o"
    ])
    if qtype in ("definition","example") and not long_query and not complex_hint:
        return MODEL_DEFAULT, MAX_TOKENS_DEF, {"effort":"low"}, "medium"
    if qtype in ("comparison","explanation","application") and (long_query or complex_hint):
        return MODEL_COMPLEX, MAX_TOKENS_CX, {"effort":"high"}, "high"
    return MODEL_DEFAULT, MAX_TOKENS_DEF, {"effort":"medium"}, "high"

# ── Engine ───────────────────────────────────────────────────────────────────
def enhanced_ask_question(question: str, course_id: str) -> str:
    try:
        print(f"🤖 Hybrid RAG for: {question}")
        qtype = classify_question_simple(question)
        print(f"📝 Type: {qtype}")

        # Embedding
        emb = openai_client.embeddings.create(model=EMBED_MODEL, input=[question])
        qvec = emb.data[0].embedding

        # Vector search
        try:
            results = vector_store.query(course_id, qvec, top_k=TOP_K) or []
        except Exception:
            results = []

        if not results and not ALLOW_GENERAL:
            return ("I couldn’t find enough in your course files to answer this. "
                    "Upload relevant slides/notes and try again.")

        # Build context + source map
        context, sources = build_context(results) if results else ("(no matching course content)", [])

        # Route model
        model, max_tokens, reasoning_effort, verbosity = _route(question, qtype)

        # Compose messages
        messages = [
            {"role": "system", "content": SYSTEM_STYLE},
            {"role": "user", "content": user_prompt(question, qtype, context, ALLOW_GENERAL)}
        ]

        # Call GPT-5
        resp = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,                 # crisp + consistent
            max_tokens=max_tokens,
            extra_body={
                "reasoning": reasoning_effort,  # {"effort":"low|medium|high"}
                "verbosity": verbosity          # "low"|"medium"|"high"
            }
        )
        answer = resp.choices[0].message.content

        # Post-format to your house style (still smooth, keeps sections)
        final = format_ai_response(answer, qtype)

        print("✅ Answer generated (hybrid).")
        return final

    except Exception as e:
        print(f"❌ Engine error: {e}")
        import traceback; traceback.print_exc()
        return ("I hit an error while processing your question. "
                "Please try again or verify your uploads.")

# ── Optional: tiny helper for explicit “course-only” calls ───────────────────
def ask_course_only(question: str, course_id: str) -> str:
    """Force grounded-only mode for exams that require citing slides."""
    global ALLOW_GENERAL
    prev = ALLOW_GENERAL
    ALLOW_GENERAL = False
    try:
        return enhanced_ask_question(question, course_id)
    finally:
        ALLOW_GENERAL = prev
