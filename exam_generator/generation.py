# exam_generator/generation.py - Practice exam assembly, context sampling, and question validation
import os
from datetime import datetime
from typing import Any, Dict, List

from providers import structured_call

from .schemas import EXAM_DIFFICULTY_SPECS, EXAM_QUESTION_SCHEMA


class GenerationMixin:
    """Generate practice exams from sampled course content and grounded retrieval."""

    def generate_practice_exam(self, course_id: str, exam_specs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a practice exam based on course materials and past paper analysis"""
        try:
            print(f"🎯 Generating practice exam for course: {course_id}")

            # Get course materials for context (diverse across all files)
            course_content = self.get_course_content_sample(course_id)

            # Generate questions based on specifications
            questions = self.generate_exam_questions(
                course_content,
                exam_specs,
                course_id
            )

            exam_data = {
                "id": f"exam_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "name": exam_specs.get("name", f"Practice Exam - {datetime.now().strftime('%Y-%m-%d')}"),
                "course_id": course_id,
                "questions": questions,
                "time_limit": exam_specs.get("time_limit", 120),
                "total_points": sum(q.get("points", 0) for q in questions),
                "question_count": len(questions),
                "difficulty": exam_specs.get("difficulty", "medium"),  # respect requested difficulty
                "created_at": datetime.now().isoformat(),
                "instructions": self.generate_exam_instructions(exam_specs)
            }
            print(f"✅ Generated exam with {len(questions)} questions")
            return {"status": "success", "exam": exam_data}
        except Exception as e:
            print(f"❌ Exam generation failed: {e}")
            return {"status": "error", "message": str(e)}

    def get_course_content_sample(self, course_id: str) -> str:
        """Build a diverse, multi-document context sample for the exam generator.

        Strategy:
        1) Pull many rows from the embeddings table for this course.
        2) Group by document and pick a representative (longest) chunk per doc.
        3) Concatenate a capped number of documents to keep the prompt compact.
        4) Fallback to vector store with several seed queries if DB access fails.
        """
        try:
            from supabase import create_client
            SUPABASE_URL = os.getenv("SUPABASE_URL")
            SUPABASE_KEY = os.getenv("SUPABASE_KEY")
            sb = create_client(SUPABASE_URL, SUPABASE_KEY)

            resp = sb.table("embeddings") \
                     .select("doc_name, content") \
                     .eq("course_id", course_id) \
                     .limit(1000) \
                     .execute()

            rows = resp.data or []
            if not rows:
                raise RuntimeError("No embeddings found for this course")

            # Group by document
            by_doc: Dict[str, List[str]] = {}
            for r in rows:
                doc = (r.get("doc_name") or "unknown").strip()
                content = (r.get("content") or "").strip()
                if not content:
                    continue
                by_doc.setdefault(doc, []).append(content)

            # Pick one strong chunk per document (prefer the longest)
            parts: List[str] = []
            for doc in sorted(by_doc.keys(), key=lambda d: len(by_doc[d]), reverse=True):
                chunks = by_doc[doc]
                best = max(chunks, key=len)
                if len(best) < 80 and len(chunks) > 1:
                    best = sorted(chunks, key=len, reverse=True)[0]
                parts.append(f"From {doc}: {best[:800]}")
                if len(parts) >= 12:
                    break

            context = "\n\n---\n\n".join(parts).strip()
            return context if context else "No course content available"

        except Exception as e:
            print(f"Course content sampling error (DB path). Falling back to vector store: {e}")
            try:
                seeds = [
                    "overview of course",
                    "key definitions",
                    "main theorems and proofs",
                    "worked examples",
                    "common pitfalls",
                    "summary"
                ]
                seen_docs = set()
                parts = []
                for q in seeds:
                    emb = self.openai_client.embeddings.create(
                        model=os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-large"),
                        input=[q]
                    )
                    vec = emb.data[0].embedding
                    hits = self.vector_store.query(course_id, vec, top_k=6) or []
                    for h in hits:
                        doc = (h.get("doc_name") or "unknown").strip()
                        if doc in seen_docs:
                            continue
                        content = (h.get("content") or "").strip()
                        if not content or len(content) < 80:
                            continue
                        parts.append(f"From {doc}: {content[:800]}")
                        seen_docs.add(doc)
                        if len(parts) >= 12:
                            break
                    if len(parts) >= 12:
                        break

                return "\n\n---\n\n".join(parts) if parts else "Limited course content available"
            except Exception as e2:
                print(f"Vector-store fallback also failed: {e2}")
                return "Limited course content available"

    def generate_exam_questions(self, course_content: str, exam_specs: Dict[str, Any], course_id: str) -> List[Dict[str, Any]]:
        """Generate exam questions, respecting the requested difficulty and types.

        Uses guaranteed-schema tool use (structured_call) and grounds questions in
        retrieved course material. Multiple-choice is supported when requested.
        """
        try:
            question_count = exam_specs.get("question_count", 10)

            # Respect the requested difficulty. "mixed"/blank => let the model vary
            # difficulty per question; a concrete level => target that level.
            requested_difficulty = (exam_specs.get("difficulty") or "medium").lower()
            mixed = requested_difficulty in ("mixed", "", "any")
            target_difficulty = "medium" if mixed else requested_difficulty
            if target_difficulty not in EXAM_DIFFICULTY_SPECS:
                target_difficulty = "medium"

            # Respect requested question types, including multiple_choice.
            allowed_types = exam_specs.get("question_types") or [
                "calculation", "short_answer", "essay", "proof", "multiple_choice"
            ]
            allow_mc = "multiple_choice" in allowed_types

            # Add grounded retrieval on top of the broad multi-document sample.
            grounded = self._grounded_exam_context(course_id, exam_specs)
            context = (grounded + "\n\n" + course_content) if grounded else course_content

            difficulty_line = (
                "Vary difficulty across easy/medium/hard for good coverage."
                if mixed else
                f"Target {target_difficulty.upper()} difficulty: {EXAM_DIFFICULTY_SPECS[target_difficulty]}"
            )
            mc_line = (
                "- Multiple-choice questions are allowed: when type is 'multiple_choice', provide exactly four "
                "options prefixed 'A) '..'D) ' and set correct_answer to the letter (A/B/C/D)."
                if allow_mc else
                "- Do NOT create multiple_choice questions."
            )

            prompt = f"""Generate {question_count} exam questions based STRICTLY on the course materials below.

COURSE MATERIALS (multi-document sample):
{context[:6000]}

EXAM SPECIFICATIONS:
- {difficulty_line}
- Allowed question types: {allowed_types}
- Subject (if known): {exam_specs.get('subject', 'Academic')}

CONSTRAINTS:
{mc_line}
- Each problem must be self-contained, unambiguous, and grounded in the materials.
- Include realistic point values and per-question time estimates (minutes).
- For calculations/proofs, give a concise solution outline in correct_answer (no hidden chain-of-thought).
- Set each question's "difficulty" to its actual level."""

            out = structured_call(
                [{"role": "user", "content": prompt}],
                schema=EXAM_QUESTION_SCHEMA,
                tool_name="exam_questions",
                model=os.getenv("MODEL_COMPLEX"),
                max_tokens=4000,
            )
            questions = out.get("questions", []) if isinstance(out, dict) else []

            cleaned = self.validate_and_clean_questions(
                questions, default_difficulty=target_difficulty, allow_mc=allow_mc
            )
            return cleaned or self.create_fallback_questions(exam_specs)

        except Exception as e:
            print(f"Question generation error: {e}")
            return self.create_fallback_questions(exam_specs)

    def _grounded_exam_context(self, course_id: str, exam_specs: Dict[str, Any]) -> str:
        """Retrieve grounded context for the exam subject/topics via the canonical
        hybrid + reranked retriever. Best-effort; returns '' on any failure."""
        try:
            from rag.retrieval import retrieve
            query = (exam_specs.get("subject")
                     or " ".join(exam_specs.get("topics", []))
                     or "key concepts, definitions, theorems, and worked examples")
            rows = retrieve(query, course_id, top_k=6)
            parts = []
            for r in rows:
                doc = r.get("doc_name", "unknown")
                page = r.get("page") or r.get("slide")
                head = f"From {doc}" + (f" (p.{page})" if page else "")
                parts.append(f"{head}: {(r.get('content') or '').strip()[:700]}")
            return "\n\n---\n\n".join(parts)
        except Exception as e:  # noqa: BLE001
            print(f"Grounded exam context failed (non-fatal): {e}")
            return ""

    def validate_and_clean_questions(self, questions: List[Dict[str, Any]],
                                     default_difficulty: str = "medium",
                                     allow_mc: bool = True) -> List[Dict[str, Any]]:
        """Validate and clean generated questions, preserving MC + options."""
        valid_questions = []
        for i, q in enumerate(questions):
            try:
                if not q.get("question") or len(q.get("question", "")) < 10:
                    continue
                q_type = q.get("type", "short_answer")
                options = q.get("options")
                # Drop MC only if it wasn't requested; otherwise keep options intact.
                if q_type == "multiple_choice" and not allow_mc:
                    q_type = "short_answer"
                    options = None
                difficulty = q.get("difficulty") or default_difficulty
                if difficulty not in EXAM_DIFFICULTY_SPECS:
                    difficulty = default_difficulty
                cleaned_q = {
                    "id": q.get("id", f"q{i+1}"),
                    "type": q_type,
                    "question": q.get("question", "").strip(),
                    "options": options if q_type == "multiple_choice" else None,
                    "points": max(1, int(q.get("points", 4))),
                    "time_estimate": max(2, int(q.get("time_estimate", 5))),
                    "difficulty": difficulty,
                    "topic": q.get("topic", "General"),
                    "correct_answer": q.get("correct_answer"),
                    "explanation": q.get("explanation", ""),
                    "solution_steps": q.get("solution_steps"),
                }
                valid_questions.append(cleaned_q)
            except Exception as e:
                print(f"Question validation error: {e}")
                continue
        return valid_questions[:20]

    def create_fallback_questions(self, exam_specs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create basic fallback questions if generation fails (hard, non-MC)"""
        return [
            {
                "id": "fallback_1",
                "type": "short_answer",
                "question": "Prove that for any connected, undirected graph G=(V,E), the number of edges in every spanning tree is |V|-1.",
                "correct_answer": "A spanning tree connects all |V| vertices with no cycles; acyclicity implies exactly |V|-1 edges.",
                "explanation": "Show that a tree on n vertices has n-1 edges by induction; any extra edge would create a cycle.",
                "points": 5,
                "time_estimate": 8,
                "difficulty": "hard",
                "topic": "Graph Theory",
                "solution_steps": ["Base case n=1", "Induction on adding a vertex", "Cyclicity argument"]
            }
        ]

    def generate_exam_instructions(self, exam_specs: Dict[str, Any]) -> List[str]:
        """Generate appropriate exam instructions"""
        instructions = [
            "Read all questions carefully before beginning.",
            "Answer all questions to the best of your ability.",
            "Show your work for calculation and proof problems.",
            "Manage your time effectively across all questions."
        ]
        if exam_specs.get("time_limit"):
            instructions.insert(0, f"You have {exam_specs['time_limit']} minutes to complete this exam.")
        if "calculation" in (exam_specs.get("question_types") or []):
            instructions.append("Include proper units in your final answers for calculation problems.")
        if "essay" in (exam_specs.get("question_types") or []):
            instructions.append("For essay questions, provide detailed explanations with supporting evidence.")
        return instructions
