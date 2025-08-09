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
    
    def update_learning_progress(self, user_id: str, course_id: str, 
                               question: str, confidence: float):
        """Update student's mastery level for topics"""
        try:
            # Extract topic from question (you can make this more sophisticated)
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
                "study_time_trend": self.calculate_study_trend(interactions.data)
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
        
        # Count consecutive days
        streak = 0
        current_date = datetime.now().date()
        
        for date in sorted_dates:
            if date == current_date or date == current_date - timedelta(days=streak):
                streak += 1
                current_date = date
            else:
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