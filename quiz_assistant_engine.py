# quiz_assistant_engine.py — GPT-5 world-class quiz assistant (hybrid RAG + strict JSON)
import os
import re
import json
from typing import Dict, List, Any, Optional, Tuple
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
RAG_TOP_K          = int(os.getenv("RAG_TOP_K", "6"))
MAX_TOK_DEFAULT    = int(os.getenv("MAX_TOKENS_DEFAULT", "1800"))
MAX_TOK_COMPLEX    = int(os.getenv("MAX_TOKENS_COMPLEX", "3000"))
ALLOW_GENERAL_FILL = os.getenv("ALLOW_GENERAL", "true").lower() == "true"   # Hybrid mode on/off

SYSTEM_STYLE = (
    "You are a rigorous but friendly university tutor. Provide a single final answer only—"
    "no chain-of-thought. Be concise, correct, and clearly grounded."
)

# ─────────────────────────────────────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────────────────────────────────────
def parse_quiz_question(raw_text: str) -> Dict[str, Any]:
    """Parse quiz question into a strict JSON schema (robust with fallbacks)."""
    schema_prompt = f"""
Return ONLY valid JSON for the quiz item below using this schema:
{{
  "question_text": "string",
  "question_type": "multiple_choice|true_false|short_answer|essay|fill_blank",
  "options": ["A", "B", "..."] or null,
  "topic": "likely academic topic",
  "difficulty": "easy|medium|hard"
}}

Quiz:
{raw_text}

Rules:
- If options exist, return them as a list of strings in their original order (no letters).
- If no options, set "options": null.
- JSON only. No backticks, no commentary.
"""
    try:
        r = openai_client.chat.completions.create(
            model=MODEL_DEFAULT,
            messages=[{"role": "user", "content": schema_prompt}],
            temperature=0.1,
            max_tokens=400,
            response_format={"type": "json_object"},
            extra_body={"reasoning": {"effort": "low"}, "verbosity": "low"}
        )
        return json.loads(r.choices[0].message.content)
    except Exception as e:
        print(f"❌ Parsing failed: {e}")
        # Heuristic fallback
        if any(marker in raw_text.lower() for marker in ['a)', 'b)', 'c)', 'd)', '(a)', '(b)', ' A)', ' B)']):
            return {
                "question_text": _strip_question_stem(raw_text),
                "question_type": "multiple_choice",
                "options": extract_options_heuristic(raw_text) or None,
                "topic": "unknown",
                "difficulty": "medium"
            }
        return {
            "question_text": raw_text.strip(),
            "question_type": "short_answer",
            "options": None,
            "topic": "unknown",
            "difficulty": "medium"
        }

def _strip_question_stem(text: str) -> str:
    # Try removing labeled options from the stem
    stem = re.split(r'(\n|\r|\r\n)[A-D]\)', text, maxsplit=1, flags=re.IGNORECASE)[0]
    return stem.strip()

def extract_options_heuristic(text: str) -> List[str]:
    """Extract multiple choice options using regex variants."""
    try:
        patterns = [
            r'^[A-D]\)\s*(.+?)(?=\n[A-D]\)|\Z)',             # "A) option"
            r'^\([A-D]\)\s*(.+?)(?=\n\([A-D]\)|\Z)',         # "(A) option"
            r'^[A-D]\.\s*(.+?)(?=\n[A-D]\.|$)',              # "A. option"
        ]
        for p in patterns:
            opts = re.findall(p, text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
            opts = [re.sub(r'\s+', ' ', o).strip() for o in opts if o.strip()]
            if opts:
                return opts
        return []
    except Exception:
        return []

# ─────────────────────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────────────────────
def find_relevant_materials(question: str, course_id: str) -> List[Dict]:
    """Find relevant course materials via embeddings."""
    try:
        emb = openai_client.embeddings.create(model=EMBED_MODEL, input=[question])
        return vector_store.query(course_id, emb.data[0].embedding, top_k=RAG_TOP_K) or []
    except Exception as e:
        print(f"❌ Materials search failed: {e}")
        return []

def _build_context(materials: List[Dict]) -> Tuple[str, List[str]]:
    """Build compact, tagged context + collect source tags for UI mapping."""
    if not materials:
        return "", []
    parts, tags = [], []
    parts.append("=== COURSE MATERIALS ===")
    for i, m in enumerate(materials[:3], 1):
        content = (m.get("content", "") or "").strip()
        snippet = (content[:900] + "…") if len(content) > 900 else content
        doc = m.get("doc_name") or m.get("file") or m.get("source") or "unknown"
        page = m.get("page") or m.get("page_num")
        tag = f"[{i}:{doc}{':' + str(page) if page else ''}]"
        tags.append(tag)
        parts.append(f"Source {i} {tag}: {snippet}")
    return "\n\n".join(parts), tags

# ─────────────────────────────────────────────────────────────────────────────
# Web/General knowledge fallback (no external fetch here; clearly labeled)
# ─────────────────────────────────────────────────────────────────────────────
def general_knowledge_answer(question: str, options: Optional[List[str]], topic: str) -> str:
    opts = ""
    if options:
        opts = "\nOptions:\n" + "\n".join([f"{chr(65+i)}. {o}" for i, o in enumerate(options)])
    prompt = f"""Answer a quiz question using general academic knowledge.

Question: {question}
{opts}
Topic: {topic or 'unknown'}

Return a short, authoritative explanation suitable for students. If MCQ, call out the correct letter and why others are wrong."""
    try:
        r = openai_client.chat.completions.create(
            model=MODEL_COMPLEX,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=900,
            extra_body={"reasoning": {"effort": "medium"}, "verbosity": "high"}
        )
        return "(GN) " + (r.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"❌ General knowledge failed: {e}")
        return "(GN) General knowledge explanation unavailable."

# ─────────────────────────────────────────────────────────────────────────────
# Core generation
# ─────────────────────────────────────────────────────────────────────────────
def _route_model(question_type: str, difficulty: str) -> Tuple[str, int, Dict[str, str], str]:
    complex_q = (question_type in ("essay", "application") or difficulty == "hard")
    if complex_q:
        return MODEL_COMPLEX, MAX_TOK_COMPLEX, {"effort": "high"}, "high"
    return MODEL_DEFAULT, MAX_TOK_DEFAULT, {"effort": "medium"}, "high"

def generate_quiz_response(parsed_question: Dict, materials: List[Dict], course_id: str) -> Dict:
    """Generate intelligent quiz response with hybrid grounding + strict JSON."""
    question_type = parsed_question.get("question_type", "short_answer")
    question_text = parsed_question.get("question_text", "").strip()
    options       = parsed_question.get("options")
    topic         = parsed_question.get("topic", "")
    difficulty    = parsed_question.get("difficulty", "medium")

    context, source_tags = _build_context(materials)
    has_context = bool(context and materials and any(len(m.get('content', '')) > 100 for m in materials))

    # Compose schema instruction
    schema = {
        "answer": "Direct answer. If MCQ, include the letter like 'C'.",
        "explanation": "Why it's correct. Use inline source tags like [1:doc:page] for grounded claims. Use (GN) for general knowledge lines.",
        "confidence": 0.0,
        "study_tips": ["..."],
        "why_others_are_wrong": {"A": "reason", "B": "reason"} if options else {},
        "requires_general_knowledge": False,
        "source_citations": source_tags
    }
    schema_text = json.dumps(schema, indent=2)

    # Instruction prompt
    instruction = []
    if question_type == "multiple_choice" and options:
        options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
        instruction.append(f"This is MCQ. Analyze each option, pick the best, and explain why others are wrong.\nOptions:\n{options_text}")
    elif question_type == "true_false":
        instruction.append('This is True/False. Answer "True" or "False" and justify briefly.')
    elif question_type == "fill_blank":
        instruction.append("This is fill-in-the-blank. Provide the exact missing term/phrase and a brief justification.")
    else:
        instruction.append("Provide a direct answer with concise justification.")

    # Hybrid grounding rules
    grounding_rules = f"""
GROUNDING:
- Prefer COURSE MATERIALS (use their inline tags like [1:doc:page]).
- If essential info is missing and augmentation is allowed, smoothly add general knowledge lines marked with (GN).
- Do NOT invent citations.
- Keep it succinct and student-friendly.
"""

    prompt = f"""You are an expert tutor. Output strict JSON only.

QUESTION:
{question_text}

COURSE CONTEXT:
{context or "(no matching course content)"}

INSTRUCTIONS:
{'\n'.join(instruction)}
{grounding_rules}

OUTPUT JSON (match this shape; fill values appropriately):
{schema_text}
"""

    model, max_tok, reasoning_effort, verbosity = _route_model(question_type, difficulty)

    try:
        r = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_STYLE},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=max_tok,
            response_format={"type": "json_object"},
            extra_body={"reasoning": reasoning_effort, "verbosity": verbosity}
        )
        obj = json.loads(r.choices[0].message.content)

        # If we had no context and augmentation is allowed, ensure GN path
        if not has_context and ALLOW_GENERAL_FILL:
            obj["requires_general_knowledge"] = True
            if not obj.get("explanation") or "(GN)" not in obj.get("explanation", ""):
                gn = general_knowledge_answer(question_text, options, topic)
                obj["explanation"] = (obj.get("explanation", "").strip() + "\n\n" + gn).strip()
            obj["confidence"] = min(0.75, float(obj.get("confidence", 0.7)))

        # Normalize fields
        obj.setdefault("study_tips", [])
        if options:
            obj.setdefault("why_others_are_wrong", {})
        obj.setdefault("source_citations", source_tags)
        obj.setdefault("requires_general_knowledge", not has_context)

        return {
            "status": "success",
            "answer": obj.get("answer", "Unable to determine"),
            "explanation": obj.get("explanation", ""),
            "confidence": float(obj.get("confidence", 0.8)),
            "question_type": question_type,
            "study_tips": obj.get("study_tips", []),
            "why_others_are_wrong": obj.get("why_others_are_wrong", {}),
            "similar_concepts": [],
            "estimated_time": get_time_estimate(question_type),
            "relevant_sources": obj.get("source_citations", source_tags),
            "requires_general_knowledge": obj.get("requires_general_knowledge", False)
        }

    except Exception as e:
        print(f"❌ Response generation error: {e}")
        # Hybrid fallback if parsing or generation failed
        gn = ""
        if ALLOW_GENERAL_FILL:
            gn = general_knowledge_answer(question_text, options, topic)
        return {
            "status": "success",
            "answer": "General knowledge fallback" if gn else "Answer based on limited course context",
            "explanation": gn or "Course context was limited and parsing failed.",
            "confidence": 0.6 if gn else 0.5,
            "question_type": question_type,
            "study_tips": [
                "Upload or link the specific slide/section covering this topic.",
                "Review key terms highlighted in class notes.",
                "Re-attempt similar questions to reinforce the concept."
            ],
            "why_others_are_wrong": {},
            "similar_concepts": [],
            "estimated_time": get_time_estimate(question_type),
            "relevant_sources": [],
            "requires_general_knowledge": bool(gn)
        }

# ─────────────────────────────────────────────────────────────────────────────
# Public entrypoint
# ─────────────────────────────────────────────────────────────────────────────
def assist_with_quiz_question(question_text: str, course_id: str, session_id: str = None) -> Dict:
    """Main entry point for quiz assistance."""
    try:
        print(f"🎯 Processing quiz question: {question_text[:100]}...")
        parsed = parse_quiz_question(question_text)
        print(f"📝 Parsed as {parsed.get('question_type')} ({parsed.get('difficulty')})")
        materials = find_relevant_materials(question_text, course_id)
        print(f"📚 Found {len(materials)} relevant materials")
        resp = generate_quiz_response(parsed, materials, course_id)
        print(f"✅ Generated response with {resp.get('confidence', 0):.0%} confidence")
        return resp
    except Exception as e:
        print(f"❌ Quiz assistance error: {e}")
        import traceback; traceback.print_exc()
        return {
            "status": "error",
            "answer": "Processing error occurred",
            "explanation": "Please try again or rephrase your question.",
            "confidence": 0.0,
            "question_type": "unknown",
            "study_tips": [],
            "similar_concepts": [],
            "estimated_time": "",
            "relevant_sources": []
        }

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def get_time_estimate(question_type: str) -> str:
    return {
        "multiple_choice": "2-3 minutes",
        "true_false": "1-2 minutes",
        "short_answer": "3-5 minutes",
        "essay": "15-30 minutes",
        "fill_blank": "2-4 minutes"
    }.get(question_type, "5-10 minutes")
