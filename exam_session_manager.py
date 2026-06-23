# exam_session_manager.py - Handle active exam sessions, timing, and scoring
import os
import json
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()


def _utcnow() -> datetime:
    """Timezone-aware current UTC time (matches Postgres timestamptz reads)."""
    return datetime.now(timezone.utc)


def _utcnow_iso() -> str:
    return _utcnow().isoformat()


def _parse_dt(value: str) -> datetime:
    """Parse an ISO timestamp into a tz-aware datetime (assume UTC if naive)."""
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

# Schema for the AI answer judge (free-response grading).
ANSWER_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["correct", "partial", "incorrect"]},
        "reason": {"type": "string", "description": "One or two sentences justifying the verdict."},
    },
    "required": ["verdict", "reason"],
}

# Fraction of a question's points awarded for each verdict.
VERDICT_CREDIT = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


class ExamSessionManager:
    """Manage active exam sessions, timing, scoring, and persistence"""
    
    def __init__(self):
        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_KEY")
        self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    def create_exam_session(self, user_id: str, course_id: str, exam_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new exam session"""
        try:
            session_id = str(uuid.uuid4())
            
            session_data = {
                "id": session_id,
                "user_id": user_id,
                "course_id": course_id,
                "exam_name": exam_data.get("name", "Practice Exam"),
                "exam_data": exam_data,
                "status": "created",
                "current_question": 0,
                "user_answers": {},
                "start_time": None,
                "end_time": None,
                "time_remaining": exam_data.get("time_limit", 120) * 60,  # Convert to seconds
                "is_paused": False,
                "created_at": _utcnow_iso(),
                "updated_at": _utcnow_iso()
            }
            
            # Save to database
            result = self.supabase.table("exam_sessions").insert(session_data).execute()
            
            if result.data:
                print(f"✅ Created exam session: {session_id}")
                return {"status": "success", "session": result.data[0]}
            else:
                return {"status": "error", "message": "Failed to create session"}
                
        except Exception as e:
            print(f"❌ Session creation failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def start_exam_session(self, session_id: str) -> Dict[str, Any]:
        """Start an exam session (begin timing)"""
        try:
            # Get current session
            session_result = self.supabase.table("exam_sessions").select("*").eq("id", session_id).execute()
            
            if not session_result.data:
                return {"status": "error", "message": "Session not found"}
            
            session = session_result.data[0]
            
            if session["status"] != "created":
                return {"status": "error", "message": "Session already started or completed"}
            
            # Update session to started; start the per-question timer too.
            now_iso = _utcnow_iso()
            updated_data = {
                "status": "active",
                "start_time": now_iso,
                "current_question_start_time": now_iso,
                "updated_at": now_iso
            }
            
            result = self.supabase.table("exam_sessions").update(updated_data).eq("id", session_id).execute()
            
            if result.data:
                print(f"▶️ Started exam session: {session_id}")
                return {"status": "success", "session": result.data[0]}
            else:
                return {"status": "error", "message": "Failed to start session"}
                
        except Exception as e:
            print(f"❌ Session start failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def pause_exam_session(self, session_id: str) -> Dict[str, Any]:
        """Pause/unpause an exam session"""
        try:
            session_result = self.supabase.table("exam_sessions").select("*").eq("id", session_id).execute()
            
            if not session_result.data:
                return {"status": "error", "message": "Session not found"}
            
            session = session_result.data[0]
            
            if session["status"] != "active":
                return {"status": "error", "message": "Session not active"}
            
            # Toggle pause state
            new_pause_state = not session["is_paused"]
            
            updated_data = {
                "is_paused": new_pause_state,
                "updated_at": _utcnow_iso()
            }
            
            # If pausing, calculate remaining time
            if new_pause_state and session["start_time"]:
                elapsed_seconds = self.calculate_elapsed_time(session)
                original_time_limit = session["exam_data"]["time_limit"] * 60
                remaining_time = max(0, original_time_limit - elapsed_seconds)
                updated_data["time_remaining"] = remaining_time
            
            result = self.supabase.table("exam_sessions").update(updated_data).eq("id", session_id).execute()
            
            action = "⏸️ Paused" if new_pause_state else "▶️ Resumed"
            print(f"{action} exam session: {session_id}")
            
            return {"status": "success", "session": result.data[0] if result.data else None}
                
        except Exception as e:
            print(f"❌ Session pause failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def save_answer(self, session_id: str, question_id: str, answer: str) -> Dict[str, Any]:
        """Save an answer to a question"""
        try:
            session_result = self.supabase.table("exam_sessions").select("*").eq("id", session_id).execute()
            
            if not session_result.data:
                return {"status": "error", "message": "Session not found"}
            
            session = session_result.data[0]
            
            if session["status"] not in ["active"]:
                return {"status": "error", "message": "Session not active"}
            
            # Time spent = seconds since the student arrived at this question.
            # Accumulate across re-saves of the same question.
            elapsed = self.calculate_question_time(session, question_id)
            prev_spent = session.get("user_answers", {}).get(question_id, {}).get("time_spent", 0)

            now_iso = _utcnow_iso()
            user_answers = session.get("user_answers", {})
            user_answers[question_id] = {
                "answer": answer,
                "timestamp": now_iso,
                "time_spent": prev_spent + elapsed
            }

            updated_data = {
                "user_answers": user_answers,
                # Reset the per-question timer so time isn't double-counted.
                "current_question_start_time": now_iso,
                "updated_at": now_iso
            }
            
            result = self.supabase.table("exam_sessions").update(updated_data).eq("id", session_id).execute()
            
            return {"status": "success", "saved": True}
                
        except Exception as e:
            print(f"❌ Answer save failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def navigate_to_question(self, session_id: str, question_index: int) -> Dict[str, Any]:
        """Navigate to a specific question"""
        try:
            session_result = self.supabase.table("exam_sessions").select("*").eq("id", session_id).execute()
            
            if not session_result.data:
                return {"status": "error", "message": "Session not found"}
            
            session = session_result.data[0]
            exam_data = session["exam_data"]
            
            if question_index < 0 or question_index >= len(exam_data["questions"]):
                return {"status": "error", "message": "Invalid question index"}
            
            now_iso = _utcnow_iso()
            updated_data = {
                "current_question": question_index,
                # Start timing the newly-displayed question.
                "current_question_start_time": now_iso,
                "updated_at": now_iso
            }

            result = self.supabase.table("exam_sessions").update(updated_data).eq("id", session_id).execute()

            return {"status": "success", "current_question": question_index}
                
        except Exception as e:
            print(f"❌ Navigation failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def submit_exam(self, session_id: str) -> Dict[str, Any]:
        """Submit and score the exam"""
        try:
            session_result = self.supabase.table("exam_sessions").select("*").eq("id", session_id).execute()
            
            if not session_result.data:
                return {"status": "error", "message": "Session not found"}
            
            session = session_result.data[0]
            
            if session["status"] == "completed":
                return {"status": "error", "message": "Exam already submitted"}
            
            # Calculate final score
            scoring_result = self.calculate_final_score(session)
            
            # Update session to completed
            updated_data = {
                "status": "completed",
                "end_time": _utcnow_iso(),
                "final_score": scoring_result,
                "updated_at": _utcnow_iso()
            }
            
            result = self.supabase.table("exam_sessions").update(updated_data).eq("id", session_id).execute()
            
            # Track analytics
            self.track_exam_completion(session, scoring_result)

            # Closed loop: seed review items for missed questions.
            self._seed_exam_mistakes(session, scoring_result)

            print(f"✅ Submitted exam session: {session_id}")
            return {
                "status": "success", 
                "session": result.data[0] if result.data else None,
                "results": scoring_result
            }
                
        except Exception as e:
            print(f"❌ Exam submission failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def calculate_final_score(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate the final score and detailed results"""
        try:
            exam_data = session["exam_data"]
            user_answers = session.get("user_answers", {})
            questions = exam_data["questions"]
            course_id = session.get("course_id") or ""

            total_points = 0
            earned_points = 0
            correct_count = 0
            question_results = []
            topic_performance = {}
            explained = 0
            EXPLAIN_CAP = 8  # bound grounded-explanation LLM calls per submission
            
            for question in questions:
                q_id = question["id"]
                total_points += question.get("points", 0)
                
                user_answer_data = user_answers.get(q_id, {})
                user_answer = (user_answer_data.get("answer") or "").strip()
                correct_answer = (question.get("correct_answer") or "").strip()
                possible = question.get("points", 0)

                # Grade with partial credit where appropriate.
                grade = self.grade_response(question, user_answer, correct_answer)
                verdict = grade["verdict"]
                is_correct = verdict == "correct"
                points_earned = round(possible * VERDICT_CREDIT.get(verdict, 0.0), 2)
                earned_points += points_earned

                if is_correct:
                    correct_count += 1

                # Track topic performance (points-based, so partial credit counts).
                topic = question.get("topic", "General")
                if topic not in topic_performance:
                    topic_performance[topic] = {"correct": 0, "total": 0, "points_earned": 0, "points_possible": 0}

                topic_performance[topic]["total"] += 1
                topic_performance[topic]["points_possible"] += possible
                topic_performance[topic]["points_earned"] += points_earned
                if is_correct:
                    topic_performance[topic]["correct"] += 1

                # Grounded "explain my mistake" for wrong answers (capped).
                mistake_explanation = ""
                mistake_source = {"doc_name": None, "page": None}
                if not is_correct and user_answer and course_id and explained < EXPLAIN_CAP:
                    try:
                        import mistake_engine
                        grounded = mistake_engine.explain_mistake(
                            course_id, question.get("question", ""), topic,
                            user_answer, correct_answer,
                        )
                        mistake_explanation = grounded.get("explanation") or ""
                        mistake_source = grounded.get("source") or mistake_source
                        explained += 1
                    except Exception as e:  # noqa: BLE001
                        print(f"exam explain_mistake failed: {e}")

                question_results.append({
                    "question_id": q_id,
                    "question": question["question"],
                    "user_answer": user_answer,
                    "correct_answer": correct_answer,
                    "is_correct": is_correct,
                    "verdict": verdict,
                    "grade_reason": grade.get("reason", ""),
                    "points_earned": points_earned,
                    "points_possible": possible,
                    "topic": topic,
                    "difficulty": question.get("difficulty", "medium"),
                    "explanation": question.get("explanation", ""),
                    "mistake_explanation": mistake_explanation,
                    "mistake_source": mistake_source,
                    "time_spent": user_answer_data.get("time_spent", 0)
                })
            
            # Calculate percentages and grades
            percentage = (earned_points / total_points * 100) if total_points > 0 else 0
            letter_grade = self.calculate_letter_grade(percentage)
            
            # Calculate time metrics
            time_metrics = self.calculate_time_metrics(session)
            
            return {
                "total_questions": len(questions),
                "correct_answers": correct_count,
                "total_points": total_points,
                "earned_points": earned_points,
                "percentage": round(percentage, 1),
                "letter_grade": letter_grade,
                "question_results": question_results,
                "topic_performance": topic_performance,
                "time_metrics": time_metrics,
                "completion_date": _utcnow_iso()
            }
            
        except Exception as e:
            print(f"❌ Score calculation failed: {e}")
            return {"error": str(e)}
    
    def _seed_exam_mistakes(self, session: Dict[str, Any], scoring_result: Dict[str, Any]) -> None:
        """Seed the spaced-repetition review queue from missed exam questions."""
        try:
            import review_engine
            user_id = session.get("user_id") or "anonymous"
            course_id = session.get("course_id") or ""
            for qr in scoring_result.get("question_results", []):
                if qr.get("verdict") == "correct":
                    continue
                review_engine.seed_from_mistake(
                    user_id=user_id,
                    course_id=course_id,
                    concept=qr.get("topic") or "general",
                    prompt=qr.get("question") or "",
                    answer=qr.get("correct_answer") or "",
                    explanation=qr.get("explanation") or qr.get("grade_reason") or "",
                    source="exam",
                )
        except Exception as e:  # noqa: BLE001  must never break submission
            print(f"exam review seeding failed: {e}")

    def grade_response(self, question: Dict[str, Any], user_answer: str, correct_answer: str) -> Dict[str, Any]:
        """Grade one response, returning {verdict, reason}.

        - multiple_choice: exact letter match (correct/incorrect).
        - calculation: numerical comparison with tolerance (correct/incorrect).
        - short_answer/essay/proof/diagram: AI judge with partial credit.
        """
        question_type = question.get("type", "short_answer")

        if not user_answer:
            return {"verdict": "incorrect", "reason": "No answer provided."}

        if question_type == "multiple_choice":
            ok = self._normalize_letter(user_answer) == self._normalize_letter(correct_answer)
            return {"verdict": "correct" if ok else "incorrect",
                    "reason": "Selected the correct option." if ok else f"Correct option was {correct_answer}."}

        if question_type == "calculation":
            ok = self.compare_numerical_answers(user_answer, correct_answer)
            return {"verdict": "correct" if ok else "incorrect",
                    "reason": "Numerically matches the expected answer." if ok else "Does not match the expected value."}

        # Free-response: use the AI judge.
        return self.judge_text_answer(
            question.get("question", ""), user_answer, correct_answer,
            question.get("explanation", "")
        )

    def _normalize_letter(self, text: str) -> str:
        """Reduce an MC answer to its option letter (handles 'B', 'b)', 'B) foo')."""
        t = (text or "").strip().upper()
        return t[0] if t and t[0] in "ABCD" else t

    def judge_text_answer(self, question: str, user_answer: str, correct_answer: str,
                          explanation: str = "") -> Dict[str, Any]:
        """AI judge for free-response answers via guaranteed-schema tool use."""
        try:
            from providers import structured_call
            reference = correct_answer or explanation or "(no reference provided)"
            prompt = (
                "You are grading a student's exam answer. Decide if it is correct, partially "
                "correct, or incorrect relative to the reference. Award 'partial' when the answer "
                "captures some but not all key ideas. Judge meaning, not exact wording.\n\n"
                f"QUESTION:\n{question}\n\n"
                f"REFERENCE ANSWER:\n{reference}\n\n"
                f"STUDENT ANSWER:\n{user_answer}\n\n"
                "Return your verdict and a one-sentence reason."
            )
            out = structured_call(
                [{"role": "user", "content": prompt}],
                schema=ANSWER_JUDGE_SCHEMA,
                tool_name="answer_judge",
                model=os.getenv("MODEL_DEFAULT"),
                max_tokens=300,
            )
            verdict = out.get("verdict", "incorrect") if isinstance(out, dict) else "incorrect"
            if verdict not in VERDICT_CREDIT:
                verdict = "incorrect"
            return {"verdict": verdict, "reason": out.get("reason", "") if isinstance(out, dict) else ""}
        except Exception as e:  # noqa: BLE001  fall back to lenient keyword overlap
            print(f"AI answer judge failed, falling back to keyword overlap: {e}")
            ok = self.compare_text_answers(user_answer, correct_answer)
            return {"verdict": "correct" if ok else "incorrect",
                    "reason": "Graded by keyword overlap (AI judge unavailable)."}

    def compare_numerical_answers(self, user_answer: str, correct_answer: str) -> bool:
        """Compare numerical answers with tolerance"""
        try:
            import re
            
            # Extract numbers from answers
            user_nums = re.findall(r'-?\d+\.?\d*', user_answer)
            correct_nums = re.findall(r'-?\d+\.?\d*', correct_answer)
            
            if not user_nums or not correct_nums:
                return user_answer.lower().strip() == correct_answer.lower().strip()
            
            # Compare primary numbers with 5% tolerance
            user_val = float(user_nums[0])
            correct_val = float(correct_nums[0])
            
            tolerance = abs(correct_val * 0.05)  # 5% tolerance
            return abs(user_val - correct_val) <= tolerance
            
        except:
            # Fall back to string comparison
            return user_answer.lower().strip() == correct_answer.lower().strip()
    
    def compare_text_answers(self, user_answer: str, correct_answer: str) -> bool:
        """Compare text answers with keyword matching"""
        try:
            user_words = set(user_answer.lower().split())
            correct_words = set(correct_answer.lower().split())
            
            # Remove common words
            common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
            user_words -= common_words
            correct_words -= common_words
            
            if not correct_words:
                return True  # If no key words, accept any answer
            
            # Calculate overlap
            overlap = len(user_words & correct_words)
            overlap_ratio = overlap / len(correct_words)
            
            # Accept if 60% of key words are present
            return overlap_ratio >= 0.6
            
        except:
            return False
    
    def calculate_letter_grade(self, percentage: float) -> str:
        """Convert percentage to letter grade"""
        if percentage >= 90:
            return "A"
        elif percentage >= 80:
            return "B"
        elif percentage >= 70:
            return "C"
        elif percentage >= 60:
            return "D"
        else:
            return "F"
    
    def calculate_time_metrics(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate time-related metrics"""
        try:
            start_time = session.get("start_time")
            time_limit = session["exam_data"]["time_limit"] * 60  # Convert to seconds

            if start_time:
                total_time_used = int((_utcnow() - _parse_dt(start_time)).total_seconds())
            else:
                total_time_used = 0
            
            return {
                "time_limit_seconds": time_limit,
                "time_used_seconds": total_time_used,
                "time_remaining_seconds": max(0, time_limit - total_time_used),
                "time_used_minutes": round(total_time_used / 60, 1),
                "time_efficiency": round((total_time_used / time_limit) * 100, 1) if time_limit > 0 else 0
            }
            
        except Exception as e:
            print(f"Time calculation error: {e}")
            return {"error": str(e)}
    
    def calculate_elapsed_time(self, session: Dict[str, Any]) -> int:
        """Calculate elapsed time in seconds"""
        try:
            start_time = session.get("start_time")
            if not start_time:
                return 0
            
            elapsed = (_utcnow() - _parse_dt(start_time)).total_seconds()
            return int(elapsed)
            
        except Exception as e:
            print(f"Elapsed time calculation error: {e}")
            return 0
    
    def calculate_question_time(self, session: Dict[str, Any], question_id: str) -> int:
        """Seconds elapsed since the student arrived at the current question.

        Reads ``current_question_start_time`` (set on start/navigate/save). Returns
        0 if it isn't set yet (e.g. legacy sessions created before this field).
        """
        start = session.get("current_question_start_time")
        if not start:
            return 0
        try:
            return max(0, int((_utcnow() - _parse_dt(start)).total_seconds()))
        except Exception as e:
            print(f"Question time calculation error: {e}")
            return 0
    
    def track_exam_completion(self, session: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Track exam completion for analytics"""
        try:
            # Update learning analytics
            from learning_analytics import LearningAnalyticsEngine
            analytics = LearningAnalyticsEngine()
            
            user_id = session["user_id"]
            course_id = session["course_id"]
            
            # Track overall exam performance
            analytics.track_interaction(
                user_id=user_id,
                course_id=course_id,
                question=f"Exam: {session['exam_name']}",
                answer=f"Score: {results.get('percentage', 0)}%",
                confidence=results.get('percentage', 0) / 100,
                response_time=results.get('time_metrics', {}).get('time_used_seconds', 0),
                question_type="exam"
            )
            
            # Track topic-specific performance
            for topic, performance in results.get('topic_performance', {}).items():
                if performance['total'] > 0:
                    topic_score = performance['correct'] / performance['total']
                    analytics.update_learning_progress(user_id, course_id, topic, topic_score)
            
        except Exception as e:
            print(f"Analytics tracking failed: {e}")
    
    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get current session state"""
        try:
            result = self.supabase.table("exam_sessions").select("*").eq("id", session_id).execute()
            
            if result.data:
                session = result.data[0]
                
                # Calculate current time remaining if active
                if session["status"] == "active" and not session["is_paused"]:
                    elapsed = self.calculate_elapsed_time(session)
                    time_limit = session["exam_data"]["time_limit"] * 60
                    session["time_remaining"] = max(0, time_limit - elapsed)
                
                return {"status": "success", "session": session}
            else:
                return {"status": "error", "message": "Session not found"}
                
        except Exception as e:
            print(f"❌ Get session failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_user_exam_history(self, user_id: str, course_id: str = None) -> List[Dict[str, Any]]:
        """Get user's exam history"""
        try:
            query = self.supabase.table("exam_sessions").select("*").eq("user_id", user_id)
            
            if course_id:
                query = query.eq("course_id", course_id)
            
            result = query.order("created_at", desc=True).execute()
            
            return result.data or []
            
        except Exception as e:
            print(f"❌ Get exam history failed: {e}")
            return []
    
    def delete_session(self, session_id: str) -> bool:
        """Delete an exam session"""
        try:
            result = self.supabase.table("exam_sessions").delete().eq("id", session_id).execute()
            return len(result.data) > 0 if result.data else True
            
        except Exception as e:
            print(f"❌ Delete session failed: {e}")
            return False
    
    def auto_submit_expired_exams(self) -> int:
        """Auto-submit exams that have exceeded their time limit"""
        try:
            # Find active sessions that should have expired
            cutoff_time = (_utcnow() - timedelta(hours=6)).isoformat()  # 6 hour buffer
            
            result = self.supabase.table("exam_sessions").select("*").eq("status", "active").lt("start_time", cutoff_time).execute()
            
            expired_count = 0
            for session in result.data or []:
                try:
                    elapsed = self.calculate_elapsed_time(session)
                    time_limit = session["exam_data"]["time_limit"] * 60
                    
                    if elapsed > time_limit:
                        self.submit_exam(session["id"])
                        expired_count += 1
                        print(f"🕐 Auto-submitted expired exam: {session['id']}")
                        
                except Exception as e:
                    print(f"Failed to auto-submit session {session['id']}: {e}")
            
            return expired_count
            
        except Exception as e:
            print(f"❌ Auto-submit failed: {e}")
            return 0