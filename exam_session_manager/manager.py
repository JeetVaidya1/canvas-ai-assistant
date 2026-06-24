# manager.py - The ExamSessionManager: active session lifecycle and persistence
import os
import uuid
from typing import Dict, List, Any, Optional
from datetime import timedelta
from supabase import create_client

from .timing import _utcnow, _utcnow_iso, _parse_dt
from .schemas import ANSWER_JUDGE_SCHEMA, VERDICT_CREDIT
from .grading import GradingMixin
from .timing_metrics import TimingMetricsMixin


class ExamSessionManager(GradingMixin, TimingMetricsMixin):
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
