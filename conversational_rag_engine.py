# conversational_rag_engine.py - Conversational RAG (context-aware, smooth)

import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from vector_store import VectorStore

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)
vector_store = VectorStore()

# ── Config (env-overridable) ──────────────────────────────────────────────────
MODEL_DEFAULT      = os.getenv("MODEL_DEFAULT", "gpt-5-mini")
MODEL_COMPLEX      = os.getenv("MODEL_COMPLEX", "gpt-5")
EMBED_MODEL        = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-large")
RAG_TOP_K          = int(os.getenv("RAG_TOP_K", "12"))
ALLOW_GENERAL_FILL = os.getenv("ALLOW_GENERAL", "true").lower() == "true"
MAX_TOK_DEFAULT    = int(os.getenv("MAX_TOKENS_DEFAULT", "3500"))
MAX_TOK_COMPLEX    = int(os.getenv("MAX_TOKENS_COMPLEX", "5500"))

SYSTEM_STYLE = (
    "You are a friendly, sharp university tutor. Speak conversationally and natural, not stiff. "
    "Answer directly with short paragraphs (and compact bullets only when they help). "
    "Be precise, avoid filler, and end with a helpful follow-up question when useful. "
    "Do not reveal chain-of-thought."
)

class ConversationalRAGEngine:
    def __init__(self):
        self.openai_client = openai_client
        self.vector_store = vector_store

    # ── Conversation memory from Supabase (graceful fallback) ─────────────────
    def get_conversation_context(self, session_id: str, last_n_messages: int = 6) -> str:
        """Return compact recent dialog (Student/AI lines), or empty string."""
        try:
            from supabase import create_client
            SUPABASE_URL = os.getenv("SUPABASE_URL")
            SUPABASE_KEY = os.getenv("SUPABASE_KEY")
            if not SUPABASE_URL or not SUPABASE_KEY:
                return ""

            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            resp = (
                supabase.table("messages")
                .select("role, content")
                .eq("session_id", session_id)
                .order("timestamp", desc=True)
                .limit(last_n_messages)
                .execute()
            )
            messages = (resp.data or [])[::-1]  # chronological
            parts = []
            for m in messages:
                role = "Student" if m.get("role") == "user" else "AI"
                content = (m.get("content") or "")
                if len(content) > 300:
                    content = content[:300] + "..."
                parts.append(f"{role}: {content}")
            return "\n".join(parts)
        except Exception as e:
            print(f"Error getting conversation context: {e}")
            return ""

    # ── Lightweight classifier ────────────────────────────────────────────────
    def _classify(self, q: str) -> str:
        ql = q.lower()
        if any(w in ql for w in ['what is','define','definition','meaning']): return 'definition'
        if any(w in ql for w in ['compare','difference','versus','vs']):      return 'comparison'
        if any(w in ql for w in ['example','show me','demonstrate']):         return 'example'
        if any(w in ql for w in ['apply','use','implement','prove','derive','solve']): return 'application'
        if any(w in ql for w in ['how','explain','why','process']):           return 'explanation'
        return 'explanation'

    # ── Enhance the search query using recent chat ────────────────────────────
    def enhance_query_with_context(self, question: str, session_id: Optional[str]) -> str:
        convo = self.get_conversation_context(session_id, 6) if session_id else ""
        if not convo:
            return question

        prompt = f"""Create a concise search query for course-material retrieval.

Conversation (recent):
{convo}

Current question: {question}

Rules:
- Keep it short and specific (under 25 words).
- Include key terms from the conversation if relevant.
- No punctuation decoration. Return ONLY the query text.
"""
        try:
            r = self.openai_client.chat.completions.create(
                model=MODEL_DEFAULT,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=60
            )
            out = (r.choices[0].message.content or "").trim()
            print(f"🔍 Enhanced query: {out}")
            return out if out else question
        except Exception as e:
            print(f"Query enhancement failed: {e}")
            return question

    # ── Build grounded context with labeled tags ──────────────────────────────
    def _build_context(self, results: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        text_chunks, image_chunks, sources = [], [], []
        for i, r in enumerate(results):
            c = r.get("content", "").strip()
            src = {
                "id": r.get("id") or f"src{i+1}",
                "file": r.get("doc_name") or r.get("file") or r.get("source") or "unknown",
                "page": r.get("page") or r.get("page_num") or None
            }
            sources.append(src)
            if "[IMAGE CONTENT" in c:
                image_chunks.append((i, c, src))
            else:
                text_chunks.append((i, c, src))

        parts = []
        if text_chunks:
            parts.append("=== TEXT FROM COURSE MATERIALS ===")
            for rank, (_, c, src) in enumerate(text_chunks[:6], 1):
                tag = f"[{rank}:{src['file']}{':' + str(src['page']) if src['page'] else ''}]"
                parts.append(f"Text {rank} {tag}: {c}")
        if image_chunks:
            parts.append("\n=== VISUAL CONTENT FROM COURSE MATERIALS ===")
            for rank, (_, c, src) in enumerate(image_chunks[:3], 1):
                tag = f"[V{rank}:{src['file']}{':' + str(src['page']) if src['page'] else ''}]"
                parts.append(f"Visual {rank} {tag}: {c}")

        return "\n".join(parts), sources

    # ── Retrieval (conversation-aware) ────────────────────────────────────────
    def intelligent_retrieval(self, question: str, course_id: str, session_id: Optional[str] = None) -> List[Dict]:
        # Enhance query with conversation
        enhanced_query = self.enhance_query_with_context(question, session_id)

        # Embed + search
        try:
            emb = self.openai_client.embeddings.create(model=EMBED_MODEL, input=[enhanced_query])
            qvec = emb.data[0].embedding
            results = self.vector_store.query(course_id, qvec, top_k=RAG_TOP_K) or []
            print(f"📚 Found {len(results)} course material results")
        except Exception as e:
            print(f"Course material search failed: {e}")
            results = []

        # Optional: quick conversation-aware rerank (JSON indices)
        if session_id and results:
            convo = self.get_conversation_context(session_id, 6)
            if convo:
                try:
                    preview = "\n".join(
                        [f"{i+1}. {(r.get('content','')[:260] or '').strip()}..." for i, r in enumerate(results[:8])]
                    )
                    rerank_prompt = f"""Rank these 1..N by relevance to the student's question given the conversation.

Conversation:
{convo}

Question: {question}

Results:
{preview}

Return JSON list of indices most→least relevant, e.g. [3,1,2,...]. No comments."""
                    rr = self.openai_client.chat.completions.create(
                        model=MODEL_DEFAULT,
                        messages=[{"role": "user", "content": rerank_prompt}],
                        max_completion_tokens=80
                    )
                    order = json.loads((rr.choices[0].message.content or "[]").strip())
                    # Reorder
                    ranked = []
                    for idx in order:
                        if isinstance(idx, int) and 1 <= idx <= len(results):
                            ranked.append(results[idx-1])
                    for r in results:
                        if r not in ranked:
                            ranked.append(r)
                    results = ranked
                    print("🎯 Reranked by conversation relevance")
                except Exception as e:
                    print(f"Rerank failed, using baseline order: {e}")

        return results

    # ── Few-shot style priming (keeps tone consistently “ChatGPT-like”) ───────
    def _few_shots(self) -> List[Dict[str, str]]:
        return [
            {
                "role": "user",
                "content": "What is dynamic programming?"
            },
            {
                "role": "assistant",
                "content": (
                    "Dynamic programming (DP) is a way to solve problems by breaking them into overlapping subproblems "
                    "and caching the results so you don’t recompute work. You design a state (what uniquely identifies a subproblem), "
                    "a recurrence (how a state depends on smaller states), and an order to fill the table (top-down with memoization, "
                    "or bottom-up with iteration).\n\n"
                    "Typical steps:\n"
                    "• Define the state clearly (e.g., dp[i] = best answer up to i).\n"
                    "• Write the recurrence from smaller states to larger ones.\n"
                    "• Establish base cases, then fill in a consistent order.\n\n"
                    "Want me to walk through a quick example, like coin change or LIS?\n\n"
                    "Sources: course_notes_dp.pdf (pp. 2–5)"
                )
            },
            {
                "role": "user",
                "content": "Give me a quick refresher on Big-O for common operations."
            },
            {
                "role": "assistant",
                "content": (
                    "Here’s a quick cheat sheet:\n"
                    "• Arrays: index O(1), search O(n) unless sorted + binary search O(log n).\n"
                    "• Linked lists: access by index O(n), insert/delete at head O(1).\n"
                    "• Hash tables: expected O(1) insert/lookup/delete; worst-case O(n).\n"
                    "• Binary search trees: balanced O(log n) search/insert/delete; skewed can be O(n).\n\n"
                    "Want me to map these to examples from your notes?\n\n"
                    "Sources: cheatsheet_runtime.pdf (p. 1)"
                )
            }
        ]

    # ── Prompt for conversational + hybrid grounding ──────────────────────────
    def _response_prompt(self, question: str, context: str, convo: str, allow_general: bool) -> str:
        """
        Ask for a single, smooth reply that uses course context first, without rigid sections.
        Keep tag markers (e.g., [1:file:page]) inside CONTEXT so the model can cite, but instruct it
        to output one compact 'Sources:' line at the very end instead of inline tags.
        """
        return f"""You are helping a student.

STUDENT QUESTION:
{question}

RECENT CONVERSATION (may inform intent):
{convo or '(none)'}

COURSE CONTEXT (primary source — chunks labeled like [1:file:page]):
{context or '(no matching course content)'}

WRITE ONE NATURAL, CONCISE ANSWER:
- Prioritize facts from COURSE CONTEXT; weave them in smoothly (no rigid section headers).
- If crucial details are missing{" and you may briefly add general knowledge" if allow_general else ""}, blend it in naturally (no special labels).
- Prefer short paragraphs; use up to 3–5 tight bullets only if they improve clarity.
- Friendly, precise tone for a university student.
- Do NOT include tag markers like [1:file:page] in the body of the answer.
- End with exactly one compact line: "Sources: <file/page, file/page, ...>" listing any tags from COURSE CONTEXT you relied on. If none, omit the line.
- Return only the final answer.
"""

    # ── Routing to mini/full and token budgets ────────────────────────────────
    def _route(self, question: str, qtype: str):
        long_q = len(question) > 240
        complex_hint = any(k in question.lower() for k in [
            "prove","derive","multi-step","step by step","design","optimize","time complexity",
            "algorithm","diagram","equation","trade-off","fourier","wavefunction","big-o"
        ])
        if qtype in ("definition","example") and not long_q and not complex_hint:
            return MODEL_DEFAULT, MAX_TOK_DEFAULT
        if qtype in ("comparison","explanation","application") and (long_q or complex_hint):
            return MODEL_COMPLEX, MAX_TOK_COMPLEX
        return MODEL_DEFAULT, MAX_TOK_DEFAULT

    # ── Clean lightweight formatting ──────────────────────────────────────────
    def clean_response(self, response: str) -> str:
        # Remove any inline tag leftovers like [1:lecture.pdf:3]
        body = re.sub(r'\[\s*\d+\s*:[^\]]+\]', '', response)
        # Normalize whitespace
        body = re.sub(r'[ \t]+', ' ', body)
        body = re.sub(r'\n{3,}', '\n\n', body).strip()
        # Keep a single trailing "Sources:" line if present; move it to the end.
        parts = body.splitlines()
        src_lines = [i for i, line in enumerate(parts) if line.strip().lower().startswith('sources:')]
        sources_text = None
        if src_lines:
            sources_text = parts[src_lines[-1]].strip()
            parts = [p for i, p in enumerate(parts) if i not in src_lines]
        out = "\n".join(parts).strip()
        if sources_text:
            if out and not out.endswith('\n'):
                out += "\n\n"
            out += sources_text
        return out

    # ── Main: conversational answer generation ────────────────────────────────
    def generate_conversational_response(self, question: str, course_id: str, session_id: str = None) -> str:
        # Retrieval
        results = self.intelligent_retrieval(question, course_id, session_id)

        # Build context
        context, _ = self._build_context(results) if results else ("", [])
        convo = self.get_conversation_context(session_id, 6) if session_id else ""

        # Compose prompt
        qtype = self._classify(question)
        prompt = self._response_prompt(question, context, convo, ALLOW_GENERAL_FILL)
        model, max_tokens = self._route(question, qtype)

        try:
            # Few-shot messages to keep tone/style consistent
            few_shots = self._few_shots()

            messages = [{"role": "system", "content": SYSTEM_STYLE}]
            messages.extend(few_shots)  # style priming
            messages.append({"role": "user", "content": prompt})

            resp = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_tokens
            )
            answer = resp.choices[0].message.content or ""
            return self.clean_response(answer)
        except Exception as e:
            print(f"Response generation failed: {e}")
            return ("I'm having trouble right now. Please try again, or upload more relevant notes/slides "
                    "so I can ground the answer better.")

# Public entrypoint
def conversational_ask_question(question: str, course_id: str, session_id: str = None) -> str:
    engine = ConversationalRAGEngine()
    return engine.generate_conversational_response(question, course_id, session_id)
