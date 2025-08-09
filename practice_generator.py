# practice_generator.py - FIXED VERSION with dynamic topic extraction
import os
import json
import random
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class PracticeGenerator:
    """Generate practice problems from course materials"""
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def extract_topics_from_course(self, course_id: str) -> List[str]:
        """Extract actual topics from course materials"""
        try:
            from vector_store import VectorStore
            vector_store = VectorStore()
            
            # Get some sample content from the course to analyze
            sample_embedding = [0.0] * 1536  # Dummy embedding to get all content
            results = vector_store.query(course_id, sample_embedding, top_k=20) or []
            
            if not results:
                print(f"No content found for course {course_id}")
                return self.get_fallback_topics()
            
            # Combine content for topic analysis
            combined_content = ""
            for result in results[:10]:  # Use first 10 chunks
                content = result.get('content', '')
                combined_content += content + " "
            
            if not combined_content.strip():
                return self.get_fallback_topics()
            
            # Use AI to extract topics from actual course content
            topics = self.analyze_content_for_topics(combined_content)
            
            if topics:
                print(f"Extracted topics for course {course_id}: {topics}")
                return topics
            else:
                return self.get_fallback_topics()
                
        except Exception as e:
            print(f"Failed to extract topics from course {course_id}: {e}")
            return self.get_fallback_topics()
    
    def analyze_content_for_topics(self, content: str) -> List[str]:
        """Use AI to analyze content and extract main topics"""
        try:
            prompt = f"""
            Analyze this educational content and extract the main topics/concepts covered.
            Return 5-8 specific topics that would be good for practice questions.
            
            Content: {content[:2000]}...
            
            Return ONLY a JSON array of topic strings, like:
            ["Binary Search Trees", "Sorting Algorithms", "Graph Theory", "Dynamic Programming"]
            
            Make the topics specific and suitable for generating practice questions.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Cheaper model for topic extraction
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )
            
            content_response = response.choices[0].message.content.strip()
            
            # Clean up JSON response
            if content_response.startswith("```json"):
                content_response = content_response.replace("```json", "").replace("```", "").strip()
            
            topics = json.loads(content_response)
            
            # Validate topics
            if isinstance(topics, list) and len(topics) > 0:
                return topics[:8]  # Limit to 8 topics
            else:
                return self.get_fallback_topics()
                
        except Exception as e:
            print(f"AI topic extraction failed: {e}")
            return self.get_fallback_topics()
    
    def get_fallback_topics(self) -> List[str]:
        """Fallback topics when extraction fails"""
        return [
            "General Concepts",
            "Key Definitions", 
            "Important Principles",
            "Core Topics",
            "Main Ideas"
        ]
    
    def generate_practice_problems(self, course_id: str, topic: str, 
                                 difficulty: str = "medium", count: int = 5) -> List[Dict[str, Any]]:
        """Generate practice problems for a specific topic"""
        try:
            # Get relevant course materials for this specific topic
            from vector_store import VectorStore
            vector_store = VectorStore()
            
            # Search for content related to the specific topic
            emb_resp = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=[topic]
            )
            
            results = vector_store.query(course_id, emb_resp.data[0].embedding, top_k=8)
            
            if not results:
                print(f"No content found for topic '{topic}' in course {course_id}")
                return self.fallback_problems(topic, count, difficulty)
            
            # Create context from results
            context = ""
            for result in results:
                content = result.get("content", "")
                if content:
                    context += content + "\n\n"
            
            if not context.strip():
                return self.fallback_problems(topic, count, difficulty)
            
            # Generate problems using AI with course-specific content
            problems = self.create_problems_with_ai(topic, context, difficulty, count, course_id)
            
            return problems
            
        except Exception as e:
            print(f"Failed to generate practice problems: {e}")
            return self.fallback_problems(topic, count, difficulty)
    
    def create_problems_with_ai(self, topic: str, context: str, 
                              difficulty: str, count: int, course_id: str) -> List[Dict[str, Any]]:
        """Use AI to create practice problems from actual course content"""
        prompt = f"""
        Create {count} practice problems about "{topic}" based SPECIFICALLY on this course material:
        
        COURSE CONTENT:
        {context}
        
        Requirements:
        - Difficulty level: {difficulty}
        - Base questions ONLY on the provided course content
        - Don't use general knowledge - use the specific concepts, examples, and terminology from the course material
        - If the course material mentions specific algorithms, examples, or concepts, use those
        
        For each problem, provide:
        1. Question text (based on course content)
        2. Multiple choice options (A, B, C, D)
        3. Correct answer (A, B, C, or D)
        4. Explanation (referencing the course material)
        5. Estimated time to solve
        
        Return as JSON array with this structure:
        [{{
            "question": "problem text based on course content",
            "options": ["A) option1", "B) option2", "C) option3", "D) option4"],
            "correct_answer": "A",
            "explanation": "explanation referencing the course material",
            "estimated_time": "2-3 minutes",
            "difficulty": "{difficulty}",
            "topic": "{topic}"
        }}]
        
        IMPORTANT: Use specific details, examples, and terminology from the course content provided above.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2500
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean up JSON if needed
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
                
            problems = json.loads(content)
            
            # Validate problems
            if isinstance(problems, list) and len(problems) > 0:
                return problems
            else:
                return self.fallback_problems(topic, count, difficulty)
                
        except Exception as e:
            print(f"AI problem generation failed: {e}")
            return self.fallback_problems(topic, count, difficulty)
    
    def fallback_problems(self, topic: str, count: int, difficulty: str) -> List[Dict[str, Any]]:
        """Fallback problems if AI generation fails"""
        return [{
            "question": f"Based on your course materials, which statement about {topic} is most accurate?",
            "options": [
                "A) This concept is not covered in the course materials", 
                "B) This topic requires more specific course content to generate questions", 
                "C) The course materials contain relevant information about this topic",
                "D) This question needs more context from your uploaded files"
            ],
            "correct_answer": "C",
            "explanation": f"This is a fallback question. To get better questions about {topic}, make sure your course materials contain relevant content about this topic.",
            "estimated_time": "1-2 minutes",
            "difficulty": difficulty,
            "topic": topic
        } for _ in range(count)]