# exam_generator/analysis.py - Past-paper analysis and AI-assisted structure/question extraction
import os
import json
from datetime import datetime
from typing import Any, Dict, List


class AnalysisMixin:
    """Analyze uploaded past papers and persist the resulting analysis."""

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
