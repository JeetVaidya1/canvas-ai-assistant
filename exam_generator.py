# exam_generator.py - Advanced exam generation from past papers and course materials
import os
import json
import re
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI
from vector_store import VectorStore
import pdfplumber
import io
import base64
from pdf2image import convert_from_bytes
try:
    import pytesseract
    OCR_OK = True
except Exception:
    OCR_OK = False

load_dotenv()

class ExamGenerator:
    """Generate practice exams from past papers and course materials"""
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.vector_store = VectorStore()
    
    def analyze_past_paper(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Analyze an uploaded past paper to extract question patterns and structure"""
        try:
            print(f"📄 Analyzing past paper: {filename}")
            
            # Extract text from PDF
            text_content = self.extract_text_from_pdf(file_bytes)
            
            if not text_content:
                return {"error": "Could not extract text from PDF"}
            
            # Use AI to analyze the exam structure
            exam_analysis = self.ai_analyze_exam_structure(text_content, filename)
            
            # Extract individual questions if possible
            questions = self.extract_questions_from_text(text_content)
            
            return {
                "status": "success",
                "filename": filename,
                "analysis": exam_analysis,
                "extracted_questions": questions,
                "content_preview": text_content[:1000] + "..." if len(text_content) > 1000 else text_content
            }
            
        except Exception as e:
            print(f"❌ Past paper analysis failed: {e}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def extract_text_from_pdf(self, file_bytes: bytes) -> str:
        """Extract text content from PDF past paper"""
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                text_parts = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                return "\n\n".join(text_parts)
        except Exception as e:
            print(f"PDF extraction error: {e}")
            return ""
    
    def ai_analyze_exam_structure(self, exam_text: str, filename: str) -> Dict[str, Any]:
        """Use AI to analyze exam structure and patterns"""
        try:
            prompt = f"""
            Analyze this exam paper and extract key structural information:
            
            EXAM TEXT:
            {exam_text[:3000]}...
            
            Return a JSON object with the following structure:
            {{
                "exam_type": "midterm|final|quiz|assignment",
                "subject": "detected subject area",
                "total_questions": number,
                "question_types": ["multiple_choice", "calculation", "short_answer", "essay", "diagram", "proof"],
                "time_limit": estimated_minutes,
                "point_distribution": {{"type": points}},
                "topics_covered": ["topic1", "topic2", ...],
                "difficulty_level": "easy|medium|hard|mixed",
                "exam_format": "structured|free_form|mixed",
                "special_instructions": "any special notes about format"
            }}
            
            Focus on:
            - Question numbering patterns
            - Point values mentioned
            - Topic areas covered
            - Types of questions (MC, calculations, etc.)
            - Any time limits mentioned
            - Difficulty indicators
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=1500,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            print(f"AI analysis error: {e}")
            return {
                "exam_type": "unknown",
                "subject": "unknown",
                "total_questions": 0,
                "question_types": ["multiple_choice"],
                "time_limit": 120,
                "difficulty_level": "medium"
            }
    
    def extract_questions_from_text(self, exam_text: str) -> List[Dict[str, Any]]:
        """Extract individual questions from exam text"""
        try:
            # Look for common question patterns
            question_patterns = [
                r'^(\d+)\.\s*(.+?)(?=^\d+\.|$)',  # "1. Question text"
                r'^Question\s+(\d+)[:.]\s*(.+?)(?=^Question\s+\d+|$)',  # "Question 1: text"
                r'^(\d+)\)\s*(.+?)(?=^\d+\)|$)',  # "1) Question text"
            ]
            
            questions = []
            
            for pattern in question_patterns:
                matches = re.findall(pattern, exam_text, re.MULTILINE | re.DOTALL)
                if matches:
                    for i, (num, text) in enumerate(matches):
                        if len(text.strip()) > 20:  # Reasonable question length
                            question_data = self.parse_individual_question(text.strip(), int(num))
                            if question_data:
                                questions.append(question_data)
                    break  # Use first successful pattern
            
            # If no structured questions found, try AI extraction
            if not questions:
                questions = self.ai_extract_questions(exam_text)
            
            return questions[:20]  # Limit to 20 questions
            
        except Exception as e:
            print(f"Question extraction error: {e}")
            return []
    
    def parse_individual_question(self, question_text: str, question_num: int) -> Optional[Dict[str, Any]]:
        """Parse an individual question to determine type and extract components"""
        try:
            # Detect question type
            question_type = self.detect_question_type(question_text)
            
            # Extract options for multiple choice
            options = []
            if question_type == "multiple_choice":
                options = self.extract_mc_options(question_text)
            
            # Estimate points (look for point indicators)
            points = self.extract_points(question_text)
            
            # Estimate time
            time_estimate = self.estimate_question_time(question_text, question_type)
            
            return {
                "id": f"extracted_{question_num}",
                "type": question_type,
                "question": self.clean_question_text(question_text),
                "options": options if options else None,
                "points": points,
                "time_estimate": time_estimate,
                "difficulty": self.estimate_difficulty(question_text),
                "topic": self.extract_topic(question_text)
            }
            
        except Exception as e:
            print(f"Question parsing error: {e}")
            return None
    
    def detect_question_type(self, text: str) -> str:
        """Detect the type of question based on content"""
        text_lower = text.lower()
        
        # Multiple choice indicators
        if re.search(r'\b[a-e]\)', text_lower) or re.search(r'\([a-e]\)', text_lower):
            return "multiple_choice"
        
        # Calculation indicators
        if any(word in text_lower for word in ['calculate', 'compute', 'find', 'determine', 'solve', 'value']):
            if any(word in text_lower for word in ['equation', 'formula', 'units', 'kg', 'meter', 'newton', 'joule']):
                return "calculation"
        
        # Essay indicators
        if any(phrase in text_lower for phrase in ['explain', 'describe', 'discuss', 'analyze', 'compare', 'contrast']):
            if len(text) > 200:  # Longer questions likely essays
                return "essay"
            else:
                return "short_answer"
        
        # Proof indicators
        if any(word in text_lower for word in ['prove', 'show that', 'demonstrate', 'derive']):
            return "proof"
        
        # Diagram indicators
        if any(word in text_lower for word in ['draw', 'sketch', 'diagram', 'graph', 'plot']):
            return "diagram"
        
        # Default to short answer
        return "short_answer"
    
    def extract_mc_options(self, text: str) -> List[str]:
        """Extract multiple choice options"""
        options = []
        patterns = [
            r'[A-E]\)\s*([^\n]+)',
            r'\([A-E]\)\s*([^\n]+)',
            r'[A-E]\.\s*([^\n]+)'
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches and len(matches) >= 2:
                options = [match.strip() for match in matches]
                break
        return options[:5]
    
    def extract_points(self, text: str) -> int:
        """Extract point value from question text"""
        point_patterns = [
            r'\[(\d+)\s*point?s?\]',
            r'\((\d+)\s*point?s?\)',
            r'(\d+)\s*point?s?',
            r'\[(\d+)\s*mark?s?\]',
            r'\((\d+)\s*mark?s?\)'
        ]
        for pattern in point_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        if len(text) > 500:
            return 5
        elif len(text) > 200:
            return 3
        else:
            return 2
    
    def estimate_question_time(self, text: str, question_type: str) -> int:
        """Estimate time needed for question in minutes"""
        base_times = {
            "multiple_choice": 2,
            "calculation": 7,   # harder bias
            "short_answer": 6,  # harder bias
            "essay": 18,
            "proof": 15,
            "diagram": 10
        }
        base_time = base_times.get(question_type, 6)
        if len(text) > 500:
            base_time += 3
        elif len(text) > 300:
            base_time += 1
        return base_time
    
    def estimate_difficulty(self, text: str) -> str:
        """Estimate question difficulty"""
        text_lower = text.lower()
        hard_words = ['derive', 'prove', 'analyze', 'synthesize', 'evaluate', 'complex', 'advanced', 'optimize', 'asymptotic', 'rigorous']
        if any(word in text_lower for word in hard_words):
            return "hard"
        easy_words = ['define', 'list', 'identify', 'state', 'basic', 'simple']
        if any(word in text_lower for word in easy_words):
            return "easy"
        return "hard"  # default bias to harder
    
    def extract_topic(self, text: str) -> str:
        """Extract topic/subject from question text"""
        text_lower = text.lower()
        physics_topics = {
            'mechanics': ['force', 'motion', 'velocity', 'acceleration', 'momentum'],
            'thermodynamics': ['heat', 'temperature', 'entropy', 'gas', 'thermal'],
            'electromagnetism': ['electric', 'magnetic', 'current', 'voltage', 'field'],
            'waves': ['wave', 'frequency', 'amplitude', 'oscillation', 'sound'],
            'optics': ['light', 'reflection', 'refraction', 'lens', 'mirror']
        }
        cs_topics = {
            'algorithms': ['algorithm', 'complexity', 'sorting', 'searching', 'big-o', 'asymptotic'],
            'data structures': ['array', 'list', 'tree', 'graph', 'stack', 'queue', 'hash'],
            'programming': ['code', 'function', 'variable', 'loop', 'recursion']
        }
        all_topics = {**physics_topics, **cs_topics}
        for topic, keywords in all_topics.items():
            if any(keyword in text_lower for keyword in keywords):
                return topic.title()
        return "General"
    
    def clean_question_text(self, text: str) -> str:
        """Clean and format question text"""
        cleaned = re.sub(r'\[?\d+\s*point?s?\]?', '', text, flags=re.IGNORECASE)
        cleaned = re.sub(r'\[?\d+\s*mark?s?\]?', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'^\d+\.\s*', '', cleaned)
        cleaned = re.sub(r'Question\s+\d+[:.]\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    
    def ai_extract_questions(self, exam_text: str) -> List[Dict[str, Any]]:
        """Use AI to extract questions when regex fails (bias to harder, non-MC)"""
        try:
            prompt = f"""
            Extract up to 10 individual NON-multiple-choice questions from this exam text.
            Prefer hard 'calculation', 'short_answer', 'proof', or 'essay' questions suitable for upper-division exams.
            
            EXAM TEXT:
            {exam_text[:4000]}
            
            Return JSON with:
            {{
              "questions": [
                {{
                  "id": "q1",
                  "type": "calculation|short_answer|essay|proof|diagram",
                  "question": "clean question text without numbering",
                  "options": null,
                  "points": estimated_points,
                  "time_estimate": estimated_minutes,
                  "difficulty": "hard",
                  "topic": "subject area"
                }}
              ]
            }}
            """
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=3000,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            return result.get("questions", [])
        except Exception as e:
            print(f"AI question extraction error: {e}")
            return []
    
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
                "difficulty": exam_specs.get("difficulty", "hard"),  # default bias to hard
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
        """Generate exam questions using AI based on course content and specifications (hard bias, avoid MC)."""
        try:
            question_count = exam_specs.get("question_count", 10)

            # Make it harder by default
            requested_difficulty = exam_specs.get("difficulty", "hard")
            effective_difficulty = "hard" if requested_difficulty in ("mixed", "", None) else requested_difficulty

            # Avoid multiple choice by default; if FE explicitly wants MC, it can still pass it.
            incoming_types = exam_specs.get("question_types", ["calculation", "short_answer", "essay", "proof"])
            effective_types = [t for t in incoming_types if t != "multiple_choice"]
            if not effective_types:
                effective_types = ["calculation", "short_answer", "proof"]

            # Build detailed prompt
            prompt = f"""
            Generate {question_count} HARD exam questions based on the course materials below.
            Prefer non-multiple-choice questions (calculation, short_answer, proof, essay).
            Require multi-step reasoning, edge cases, rigor, and where relevant, asymptotic or formal analysis.

            COURSE MATERIALS (multi-document sample):
            {course_content[:6000]}

            EXAM SPECIFICATIONS:
            - Difficulty target: {effective_difficulty} (lean hard)
            - Allowed types only: {effective_types}
            - Time limit (whole exam): {exam_specs.get('time_limit', 120)} minutes
            - Subject (if known): {exam_specs.get('subject', 'Academic')}

            Constraints:
            - DO NOT create multiple_choice questions.
            - Each problem should be self-contained and unambiguous.
            - Include point values and realistic time estimates.
            - For calculations/proofs, include a concise solution outline (not full chain-of-thought).

            Return JSON with:
            {{
              "questions": [
                {{
                  "id": "q1",
                  "type": "calculation|short_answer|essay|proof|diagram",
                  "question": "Complete question text",
                  "options": null,
                  "correct_answer": "short final answer or brief solution outline",
                  "explanation": "Key reasoning or steps (concise, no hidden chain-of-thought)",
                  "points": point_value,
                  "time_estimate": minutes,
                  "difficulty": "hard",
                  "topic": "specific topic area",
                  "solution_steps": ["step1", "step2"] or null
                }}
              ]
            }}
            """

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=4000,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            questions = result.get("questions", [])

            # Validate and clean questions
            cleaned = self.validate_and_clean_questions(questions)

            # Final enforcement: no MC + hard bias
            for q in cleaned:
                if q.get("type") == "multiple_choice":
                    q["type"] = "short_answer"
                    q.pop("options", None)
                q["difficulty"] = "hard"

                # Slightly bump points/time to reflect hardness if not set
                q["points"] = max(q.get("points", 0), 4)
                q["time_estimate"] = max(q.get("time_estimate", 0), 6)

            return cleaned
            
        except Exception as e:
            print(f"Question generation error: {e}")
            return self.create_fallback_questions(exam_specs)
    
    def validate_and_clean_questions(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and clean generated questions"""
        valid_questions = []
        for i, q in enumerate(questions):
            try:
                if not q.get("question") or len(q.get("question", "")) < 10:
                    continue
                cleaned_q = {
                    "id": q.get("id", f"q{i+1}"),
                    "type": q.get("type", "short_answer"),
                    "question": q.get("question", "").strip(),
                    "points": max(1, int(q.get("points", 4))),              # harder default
                    "time_estimate": max(3, int(q.get("time_estimate", 6))), # harder default
                    "difficulty": q.get("difficulty", "hard"),
                    "topic": q.get("topic", "General"),
                    "correct_answer": q.get("correct_answer"),
                    "explanation": q.get("explanation", ""),
                    "solution_steps": q.get("solution_steps")
                }
                # Never keep MC options (we’re avoiding MC)
                if cleaned_q["type"] == "multiple_choice":
                    cleaned_q["type"] = "short_answer"
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
    
    def save_past_paper_analysis(self, course_id: str, analysis: Dict[str, Any]) -> bool:
        """Save past paper analysis for future reference"""
        try:
            from supabase import create_client
            SUPABASE_URL = os.getenv("SUPABASE_URL")
            SUPABASE_KEY = os.getenv("SUPABASE_KEY")
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            supabase.table("past_paper_analyses").insert({
                "course_id": course_id,
                "filename": analysis.get("filename"),
                "analysis_data": analysis,
                "created_at": datetime.now().isoformat()
            }).execute()
            return True
        except Exception as e:
            print(f"Failed to save analysis: {e}")
            return False

    def _b64_png(self, pil_image) -> str:
        import io, base64
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def pdf_pages_to_images(self, file_bytes: bytes, pages: Optional[List[int]] = None) -> List[str]:
        """Return a list of base64 PNG strings for requested pages (1-based)."""
        try:
            images = convert_from_bytes(file_bytes, fmt="png")
            if pages:
                idxs = [p-1 for p in pages if 1 <= p <= len(images)]
                images = [images[i] for i in idxs]
            return [self._b64_png(im) for im in images]
        except Exception as e:
            print(f"PDF->image error: {e}")
            return []

    def ocr_pdf_text(self, file_bytes: bytes, pages: Optional[List[int]] = None) -> str:
        """OCR fallback when pdfplumber text is empty or partial."""
        if not OCR_OK:
            return ""
        try:
            images = convert_from_bytes(file_bytes, fmt="png")
            if pages:
                idxs = [p-1 for p in pages if 1 <= p <= len(images)]
                images = [images[i] for i in idxs]
            out = []
            for im in images:
                try:
                    out.append(pytesseract.image_to_string(im))
                except Exception:
                    pass
            return "\n\n".join([t for t in out if t and t.strip()])
        except Exception as e:
            print(f"OCR error: {e}")
            return ""

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
