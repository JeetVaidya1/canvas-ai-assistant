# grading.py - Answer grading, the AI judge, and scoring for exam sessions
import os
from typing import Dict, Any

from .timing import _utcnow_iso
from .schemas import ANSWER_JUDGE_SCHEMA, VERDICT_CREDIT


class GradingMixin:
    """Grading, AI judge, and final-score calculation for :class:`ExamSessionManager`."""

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
