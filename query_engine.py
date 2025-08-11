# query_engine.py — GPT-5 aligned (safe upgrades, non-breaking)
import os, json, re, time, random
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI
from vector_store import VectorStore

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_FAST  = os.getenv("MODEL_DEFAULT", "gpt-5-mini")   # classify/rerank
MODEL_GEN   = os.getenv("MODEL_COMPLEX", "gpt-5")        # final answer
EMBED_MODEL = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
RAG_TOP_K   = int(os.getenv("RAG_TOP_K", "12"))
ALLOW_GENERAL = os.getenv("ALLOW_GENERAL", "true").lower() == "true"
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "16000"))  # guardrail for prompt size

openai_client = OpenAI(api_key=OPENAI_API_KEY)
vector_store = VectorStore()

def _retry_chat(model: str, messages: List[Dict[str, Any]], temperature: float = 0.1, max_tokens: int = 2000, attempts: int = 3):
    """Tiny retry wrapper with jitter for flaky upstream errors."""
    last_err = None
    for i in range(attempts):
        try:
            return openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            last_err = e
            # backoff with jitter
            time.sleep((2 ** i) + random.uniform(0, 0.25))
    raise last_err or RuntimeError("chat completion failed")

def _truncate(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    # try to cut on a boundary
    cut = s.rfind("\n\n", 0, int(limit * 0.95))
    if cut == -1:
        cut = s.rfind(". ", 0, int(limit * 0.95))
    return s[: (cut if cut != -1 else limit)].rstrip() + "\n\n[context trimmed]"

def _mk_citation_tags(results: List[Dict[str, Any]], max_tags: int = 3) -> str:
    """Create short inline source tags like [1: doc p12] [2: notes slide3]."""
    tags = []
    for i, r in enumerate(results[:max_tags], 1):
        doc = r.get("doc_name") or "Document"
        page = r.get("page")
        slide = r.get("slide")
        which = f"p{page}" if page else (f"slide{slide}" if slide else "")
        tag = f"[{i}: {doc}" + (f" {which}]" if which else "]")
        tags.append(tag)
    return " ".join(tags)

class AdvancedRAGEngine:
    def __init__(self):
        self.openai_client = openai_client
        self.vector_store = vector_store

    def classify_query_type(self, question: str) -> str:
        prompt = f"""Classify this student question into one category:
DEFINITION, EXPLANATION, EXAMPLE, COMPARISON, APPLICATION, ANALYSIS, SYNTHESIS, EVALUATION

Question: "{question}"
Return just the category."""
        try:
            r = _retry_chat(
                MODEL_FAST,
                [{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=20
            )
            return (r.choices[0].message.content or "EXPLANATION").strip().upper()
        except Exception as e:
            print(f"classify warn: {e}")
            return "EXPLANATION"

    def expand_query(self, question: str, query_type: str) -> List[str]:
        prompt = f"""Generate 3 academic search reformulations:
1) factual, 2) conceptual, 3) applied.
Original: "{question}"
Return JSON list of 3 strings."""
        try:
            r = _retry_chat(
                MODEL_FAST,
                [{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            q = json.loads(r.choices[0].message.content.strip())
            return [question] + [str(x) for x in q][:3]
        except Exception as e:
            print(f"expand warn: {e}")
            return [question]

    def hybrid_search(self, queries: List[str], course_id: str, top_k: Optional[int] = None) -> List[Dict]:
        top_k = top_k or RAG_TOP_K
        all_results, seen = [], set()
        for q in queries:
            try:
                emb = self.openai_client.embeddings.create(model=EMBED_MODEL, input=[q]).data[0].embedding
                results = self.vector_store.query(course_id, emb, top_k=5) or []
                for r in results:
                    c = r.get("content", "")
                    if c and c not in seen:
                        r["query_used"] = q
                        r["relevance_score"] = r.get("similarity", 0.0)
                        all_results.append(r)
                        seen.add(c)
            except Exception as e:
                print(f"vector query warn: {e}")
        return all_results[:top_k]

    def rerank_results(self, question: str, results: List[Dict]) -> List[Dict]:
        if len(results) <= 1:
            return results
        docs_text = ""
        for i, r in enumerate(results):
            docs_text += f"Document {i+1}: {r.get('content','')[:500]}...\n\n"
        prompt = f"""Question: "{question}"
Rank these documents by relevance (most→least).
Docs:
{docs_text}
Return JSON list of numbers 1..{len(results)}."""
        try:
            r = _retry_chat(
                MODEL_FAST,
                [{"role":"user","content":prompt}],
                temperature=0,
                max_tokens=200
            )
            order = json.loads(r.choices[0].message.content.strip())
            reranked = []
            for idx in order:
                try:
                    n = int(idx)
                except:
                    continue
                if 1 <= n <= len(results):
                    reranked.append(results[n-1])
            for item in results:
                if item not in reranked:
                    reranked.append(item)
            return reranked
        except Exception as e:
            print(f"rerank warn: {e}")
            return results

    def create_enhanced_context(self, question: str, query_type: str, results: List[Dict]) -> str:
        if not results:
            return "No relevant course materials found."
        parts = [f"=== COURSE MATERIALS FOR: {question} ===\n"]
        for i, r in enumerate(results[:8], 1):
            content = (r.get("content","") or "").strip()
            doc = r.get("doc_name","Unknown")
            page = r.get("page")
            slide = r.get("slide")
            tag = f" [p{page}]" if page else (f" [slide{slide}]" if slide else "")
            parts.append(f"📄 Source {i}: {doc}{tag}")
            parts.append(f"Content: {content}")
            parts.append("---")
        joined = "\n".join(parts)
        return _truncate(joined, MAX_CONTEXT_CHARS)

    def generate_student_optimized_prompt(self, question: str, query_type: str, context: str) -> str:
        if query_type == "DEFINITION":
            instruction = "Provide a clear definition (plain + technical), key traits, and why it matters."
        elif query_type == "EXPLANATION":
            instruction = "Explain step-by-step, underlying principles, analogies, and misconceptions."
        elif query_type == "EXAMPLE":
            instruction = "Give 2–3 concrete examples, explain why each fits, connect to theory."
        elif query_type == "COMPARISON":
            instruction = "Compare (similarities/differences), when to use each, brief examples."
        elif query_type == "APPLICATION":
            instruction = "Show how to apply: steps, scenarios, tips, pitfalls."
        else:
            instruction = "Analyze components, relationships, implications, broader connections."
        return f"""You are an expert tutor.

QUESTION TYPE: {query_type}
STUDENT QUESTION: {question}

INSTRUCTIONS: {instruction}

COURSE MATERIALS:
{context}

GUIDELINES:
- Be clear and conversational
- Use course materials first; note gaps if any
- Include examples/analogies where helpful
- Use light structure (bullets ok), end with a brief takeaway
Provide your response:"""

    def _general_knowledge_answer(self, question: str) -> str:
        """Optional GN fallback if no course content is found."""
        prompt = f"""The student has no usable course materials for this question.
Give your best academically sound answer with a short rationale and a 2–3 bullet summary.

Question: {question}"""
        try:
            r = _retry_chat(
                MODEL_GEN,
                [{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=900
            )
            return r.choices[0].message.content
        except Exception as e:
            print(f"GN fallback warn: {e}")
            return "I couldn’t find enough in your course materials. Try uploading more relevant files."

    def ask_question(self, question: str, course_id: str) -> str:
        try:
            qtype = self.classify_query_type(question)
            queries = self.expand_query(question, qtype)
            results = self.hybrid_search(queries, course_id, top_k=RAG_TOP_K)

            if not results:
                if ALLOW_GENERAL:
                    return self._general_knowledge_answer(question)
                return "I couldn’t find enough in your course materials. Try uploading more relevant files."

            reranked = self.rerank_results(question, results)
            context = self.create_enhanced_context(question, qtype, reranked)
            prompt = self.generate_student_optimized_prompt(question, qtype, context)

            resp = _retry_chat(
                MODEL_GEN,
                [{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000
            )
            answer = resp.choices[0].message.content

            # Light inline provenance to build trust (first 2–3 sources)
            tags = _mk_citation_tags(reranked, max_tags=3)
            if tags:
                answer = f"{answer}\n\n— Sources: {tags}"

            return answer

        except Exception as e:
            print(f"RAG error: {e}")
            return "I hit an error while answering. Please try again."

advanced_rag_engine = AdvancedRAGEngine()

def ask_question(question: str, course_id: str) -> str:
    # returns a string (no tuple)
    return advanced_rag_engine.ask_question(question, course_id)
