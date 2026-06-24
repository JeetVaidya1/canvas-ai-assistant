# learning_analytics.py - FIXED VERSION with proper imports
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

class LearningAnalyticsEngine:
    """Track student progress and identify learning patterns"""
    
    def track_interaction(self, user_id: str, course_id: str, question: str, 
                         answer: str, confidence: float, response_time: int,
                         question_type: str = "general") -> bool:
        """Track every Q&A interaction for analytics"""
        try:
            interaction_data = {
                "user_id": user_id,
                "course_id": course_id,
                "question": question,
                "answer": answer,
                "confidence_score": confidence,
                "response_time": response_time,
                "question_type": question_type,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            result = supabase.table("user_interactions").insert(interaction_data).execute()
            
            # Update learning progress
            self.update_learning_progress(user_id, course_id, question, confidence)
            
            return True
        except Exception as e:
            print(f"Failed to track interaction: {e}")
            return False
    
    def track_practice_session(self, user_id: str, course_id: str, topic: str,
                               problems_attempted: int, problems_correct: int,
                               duration_minutes: int, difficulty_level: str) -> bool:
        """Persist a completed practice session and update topic mastery.

        Stores a row in ``practice_sessions`` (real per-session duration powers the
        study-time trend) and updates ``learning_progress`` using the explicit topic.
        """
        confidence = (problems_correct / problems_attempted) if problems_attempted > 0 else 0.5
        try:
            supabase.table("practice_sessions").insert({
                "user_id": user_id,
                "course_id": course_id,
                "topic": topic,
                "problems_attempted": problems_attempted,
                "problems_correct": problems_correct,
                "duration_minutes": duration_minutes,
                "difficulty_level": difficulty_level,
                "created_at": datetime.utcnow().isoformat(),
            }).execute()
            self.update_learning_progress(user_id, course_id, topic, confidence, topic=topic)
            return True
        except Exception as e:
            print(f"Failed to track practice session: {e}")
            return False

    def get_study_time_trend(self, user_id: str, course_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """Per-day study activity over the last ``days`` days.

        Returns [{date, questions, duration_minutes, avg_confidence}] sorted by date.
        Questions + confidence come from user_interactions; duration comes from
        practice_sessions (real timer) plus a small estimate per interaction so
        chat/quiz-only days still register time.
        """
        try:
            since = (datetime.utcnow() - timedelta(days=days)).isoformat()

            interactions = supabase.table("user_interactions") \
                .select("confidence_score, timestamp") \
                .eq("user_id", user_id).eq("course_id", course_id) \
                .gte("timestamp", since).execute().data or []

            sessions = supabase.table("practice_sessions") \
                .select("duration_minutes, created_at") \
                .eq("user_id", user_id).eq("course_id", course_id) \
                .gte("created_at", since).execute().data or []

            by_date: Dict[str, Dict[str, float]] = {}

            def bucket(d: str) -> Dict[str, float]:
                return by_date.setdefault(d, {"questions": 0, "duration_minutes": 0.0, "conf_sum": 0.0})

            for it in interactions:
                ts = it.get("timestamp")
                if not ts:
                    continue
                day = str(ts)[:10]
                b = bucket(day)
                b["questions"] += 1
                b["conf_sum"] += float(it.get("confidence_score") or 0.0)
                # ~1 min per question as a floor so non-practice study still counts.
                b["duration_minutes"] += 1.0

            for s in sessions:
                ts = s.get("created_at")
                if not ts:
                    continue
                day = str(ts)[:10]
                b = bucket(day)
                b["duration_minutes"] += float(s.get("duration_minutes") or 0)

            trend = []
            for day in sorted(by_date.keys()):
                b = by_date[day]
                q = int(b["questions"])
                trend.append({
                    "date": day,
                    "questions": q,
                    "duration_minutes": round(b["duration_minutes"], 1),
                    "avg_confidence": round(b["conf_sum"] / q, 2) if q else 0.0,
                })
            return trend
        except Exception as e:
            print(f"Failed to compute study time trend: {e}")
            return []

    def track_quiz_answer(self, user_id: str, course_id: str, concept: str,
                          question: str, is_correct: bool, time_taken: float = 0.0) -> bool:
        """Track a single quiz answer and update mastery for the explicit concept.

        Unlike :meth:`track_interaction`, the concept is supplied by the quiz
        generator (it knows what each question tests) rather than guessed from
        keywords. Confidence is 1.0 for a correct answer, 0.0 otherwise.
        """
        confidence = 1.0 if is_correct else 0.0
        try:
            supabase.table("user_interactions").insert({
                "user_id": user_id,
                "course_id": course_id,
                "question": question,
                "answer": "correct" if is_correct else "incorrect",
                "confidence_score": confidence,
                "response_time": time_taken,
                "question_type": "quiz",
                "timestamp": datetime.utcnow().isoformat(),
            }).execute()
            self.update_learning_progress(user_id, course_id, question, confidence, topic=concept)
            return True
        except Exception as e:
            print(f"Failed to track quiz answer: {e}")
            return False

    def update_learning_progress(self, user_id: str, course_id: str,
                               question: str, confidence: float, topic: str = None):
        """Update student's mastery level for topics"""
        try:
            # Use the explicit topic when provided (e.g. from a quiz question);
            # otherwise fall back to keyword extraction from the question text.
            if not topic:
                topic = self.extract_topic(question)

            # Get current progress
            current = supabase.table("learning_progress") \
                .select("*") \
                .eq("user_id", user_id) \
                .eq("course_id", course_id) \
                .eq("topic", topic) \
                .execute()
            
            if current.data:
                # Update existing progress
                old_mastery = current.data[0]["mastery_level"]
                new_mastery = (old_mastery + confidence) / 2  # Simple average
                
                supabase.table("learning_progress") \
                    .update({
                        "mastery_level": new_mastery,
                        "last_reviewed": datetime.utcnow().isoformat(),
                        "review_count": current.data[0]["review_count"] + 1
                    }) \
                    .eq("id", current.data[0]["id"]) \
                    .execute()
            else:
                # Create new progress entry
                supabase.table("learning_progress").insert({
                    "user_id": user_id,
                    "course_id": course_id,
                    "topic": topic,
                    "mastery_level": confidence,
                    "last_reviewed": datetime.utcnow().isoformat(),
                    "review_count": 1
                }).execute()
                
        except Exception as e:
            print(f"Failed to update progress: {e}")
    
    def extract_topic(self, question: str) -> str:
        """Extract main topic from question - can be enhanced with NLP"""
        # Simple keyword matching - enhance this later
        question_lower = question.lower()
        
        topics = {
            "binary search tree": ["bst", "binary search tree", "tree traversal"],
            "sorting": ["bubble sort", "merge sort", "quick sort", "insertion sort"],
            "algorithms": ["algorithm", "complexity", "big o"],
            "data structures": ["array", "linked list", "stack", "queue"],
        }
        
        for topic, keywords in topics.items():
            if any(keyword in question_lower for keyword in keywords):
                return topic
                
        return "general"
    
    def get_learning_analytics(self, user_id: str, course_id: str) -> Dict[str, Any]:
        """Get comprehensive learning analytics for a student"""
        try:
            # Get progress by topic
            progress = supabase.table("learning_progress") \
                .select("*") \
                .eq("user_id", user_id) \
                .eq("course_id", course_id) \
                .execute()
            
            # Get recent interactions
            interactions = supabase.table("user_interactions") \
                .select("*") \
                .eq("user_id", user_id) \
                .eq("course_id", course_id) \
                .order("timestamp", desc=True) \
                .limit(50) \
                .execute()
            
            # Calculate analytics
            analytics = {
                "topics_progress": self.calculate_topic_progress(progress.data),
                "study_streak": self.calculate_study_streak(interactions.data),
                "weak_areas": self.identify_weak_areas(progress.data),
                "study_recommendations": self.generate_recommendations(progress.data),
                "total_questions": len(interactions.data),
                "avg_confidence": self.calculate_avg_confidence(interactions.data),
                "study_time_trend": self.get_study_time_trend(user_id, course_id)
            }
            
            return analytics
            
        except Exception as e:
            print(f"Failed to get analytics: {e}")
            return {}
    
    def calculate_topic_progress(self, progress_data: List[Dict]) -> List[Dict]:
        """Calculate progress by topic"""
        return [
            {
                "topic": item["topic"],
                "mastery_level": item["mastery_level"],
                "review_count": item["review_count"],
                "last_reviewed": item["last_reviewed"]
            }
            for item in progress_data
        ]
    
    def identify_weak_areas(self, progress_data: List[Dict]) -> List[str]:
        """Identify topics that need more practice"""
        weak_areas = []
        for item in progress_data:
            if item["mastery_level"] < 0.7:  # Below 70% mastery
                weak_areas.append(item["topic"])
        return weak_areas
    
    def generate_recommendations(self, progress_data: List[Dict]) -> List[str]:
        """Generate study recommendations"""
        recommendations = []
        
        for item in progress_data:
            if item["mastery_level"] < 0.6:
                recommendations.append(f"Focus more on {item['topic']} - try practice problems")
            elif item["mastery_level"] > 0.8:
                recommendations.append(f"Great job on {item['topic']}! Try advanced problems")
        
        if not recommendations:
            recommendations.append("Keep up the great work! Try exploring new topics.")
            
        return recommendations[:5]  # Limit to 5 recommendations
    
    def calculate_study_streak(self, interactions: List[Dict]) -> int:
        """Calculate consecutive days of study"""
        if not interactions:
            return 0
            
        # Sort by date
        dates = set()
        for interaction in interactions:
            date = datetime.fromisoformat(interaction["timestamp"]).date()
            dates.add(date)
        
        sorted_dates = sorted(dates, reverse=True)

        # Count consecutive days ending today. A streak requires studying today;
        # step the expected day back by exactly one each time (the previous
        # version subtracted the running streak count, under-counting 3+ day runs).
        streak = 0
        expected = datetime.now().date()
        for day in sorted_dates:
            if day == expected:
                streak += 1
                expected = expected - timedelta(days=1)
            else:
                # First gap (or a future-dated day) ends the streak.
                break

        return streak
    
    def calculate_avg_confidence(self, interactions: List[Dict]) -> float:
        """Calculate average confidence score"""
        if not interactions:
            return 0.0
            
        total_confidence = sum(item["confidence_score"] for item in interactions)
        return total_confidence / len(interactions)
    
    def calculate_study_trend(self, interactions: List[Dict]) -> List[Dict]:
        """Calculate study trend over time"""
        # Simple implementation - can be enhanced
        return []