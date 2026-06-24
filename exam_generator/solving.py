# exam_generator/solving.py - RAG grounding and vision-based exam question solving
import os
import json
from typing import Any, Dict, List, Optional


class SolvingMixin:
    """Solve individual exam questions using vision + RAG-grounded context."""

    def _rag_context(self, course_id: str, query: str, top_k: int = 8) -> str:
        """Retrieve relevant course snippets for grounding."""
        try:
            emb = self.openai_client.embeddings.create(
                model=os.getenv("EMBEDDINGS_MODEL","text-embedding-3-large"),
                input=[query]
            )
            vec = emb.data[0].embedding
            hits = self.vector_store.query(course_id, vec, top_k=top_k) or []
            chunks = []
            seen = set()
            for h in hits:
                txt = (h.get("content") or "").strip()
                if txt and txt not in seen:
                    chunks.append(txt[:800])
                    seen.add(txt)
                if len(chunks) >= top_k:
                    break
            return "\n\n---\n\n".join(chunks)
        except Exception as e:
            print(f"RAG error: {e}")
            return ""

    def solve_question_with_vision(self,
                                   course_id: str,
                                   question_text: str,
                                   file_bytes: Optional[bytes] = None,
                                   pages: Optional[List[int]] = None,
                                   want_hint: bool = False) -> Dict[str, Any]:
        """
        Use GPT-5/4o vision + RAG to solve a question. Supports diagrams via PDF page images.
        Returns dict with: {'answer','steps','choice','units','used_pages'}
        """
        try:
            vision_model = os.getenv("VISION_MODEL", "gpt-5-vision")
            text_model = os.getenv("TEXT_MODEL", "gpt-5")

            # Build RAG context
            context = self._rag_context(course_id, question_text, top_k=8)

            # Prepare image attachments
            image_blocks = []
            used_pages = []
            if file_bytes:
                pp_text = self.extract_text_from_pdf(file_bytes)
                if not pp_text or len(pp_text.strip()) < 80:
                    pp_text = self.ocr_pdf_text(file_bytes, pages=pages)

                if pages:
                    used_pages = pages
                b64s = self.pdf_pages_to_images(file_bytes, pages=pages)
                for b in b64s:
                    image_blocks.append({
                        "type": "image_url",
                        "image_url": { "url": f"data:image/png;base64,{b}" }
                    })

            task = "Give a helpful hint only (no final numeric/letter answer)" if want_hint else "Provide the final answer"
            user_content = [
                {"type":"text","text":
                 f"""Solve the exam question below. Use the images and context if helpful.

Question:
{question_text}

COURSE CONTEXT (RAG snippets):
{context[:6000]}

Instructions:
- {task}.
- If numeric, include units and show concise steps (no hidden chain-of-thought).
- If MC, return a 'choice' key like "A"/"B"... and also the final reasoning.
- Keep steps clear and compact.
- Return strict JSON with keys:
  {{
    "final_answer": "...",       # or "" if hint mode
    "steps": ["...", "..."],
    "choice": "A|B|C|D|null",
    "units": "m|s|N|...|null"
  }}
                 """}
            ]
            user_content.extend(image_blocks)

            resp = self.openai_client.chat.completions.create(
                model=vision_model,
                messages=[
                    {"role":"system","content":"You are a precise exam solver. Return JSON only, no extra prose."},
                    {"role":"user","content": user_content}
                ],
                max_completion_tokens=1200
            )

            raw = (resp.choices[0].message.content or "").strip()
            try:
                obj = json.loads(raw)
            except Exception:
                obj = {"final_answer": raw, "steps": [], "choice": None, "units": None}

            obj.setdefault("choice", None)
            obj.setdefault("units", None)
            obj["used_pages"] = used_pages
            return {"status":"success", "solution": obj}

        except Exception as e:
            print(f"Vision solve error: {e}")
            return {"status":"error", "message": str(e)}
