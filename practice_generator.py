# practice_generator.py - COMPLETE ENHANCED VERSION for any course subject
import os
import json
import random
import re
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class PracticeGenerator:
    """Generate practice problems from course materials - works for any subject"""
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def extract_topics_from_course(self, course_id: str) -> List[str]:
        """Extract actual topics from course materials using multiple strategies - IMPROVED"""
        try:
            from vector_store import VectorStore
            vector_store = VectorStore()
            
            print(f"🔍 Extracting topics for course: {course_id}")
            
            # Strategy 1: Extract from filenames (most reliable for structure)
            filename_topics = self.extract_topics_from_filenames(course_id)
            print(f"📁 Topics from filenames ({len(filename_topics)}): {filename_topics}")
            
            # Strategy 2: Analyze actual content with AI 
            content_topics = self.extract_topics_from_content(course_id, vector_store)
            print(f"📖 Topics from content analysis ({len(content_topics)}): {content_topics}")
            
            # Strategy 3: Get course info from database (if available)
            course_context_topics = self.get_course_context_topics(course_id)
            print(f"🎓 Topics from course context ({len(course_context_topics)}): {course_context_topics}")
            
            # Combine more intelligently with preference for filename topics
            all_topics = self.combine_and_rank_topics(filename_topics, content_topics, course_context_topics)
            
            # Ensure we don't lose important filename topics
            if len(all_topics) < len(filename_topics):
                print(f"⚠️ We lost some filename topics! Adding them back...")
                for ft in filename_topics:
                    if ft not in all_topics:
                        all_topics.append(ft)
                        print(f"  🔄 Restored filename topic: {ft}")
            
            # Clean and limit (but be less aggressive)
            final_topics = self.clean_and_limit_topics(all_topics, max_topics=15)
            
            # Ensure we have reasonable topics
            if len(final_topics) < 3:
                print("⚠️ Very few topics found, adding fallback topics...")
                fallback = self.get_generic_fallback_topics(course_id)
                final_topics.extend(fallback)
                final_topics = final_topics[:15]  # Still limit, but higher
            
            print(f"✅ FINAL topics for course {course_id} ({len(final_topics)}): {final_topics}")
            return final_topics
                
        except Exception as e:
            print(f"❌ Failed to extract topics from course {course_id}: {e}")
            import traceback
            traceback.print_exc()
            return self.get_generic_fallback_topics(course_id)
    
    def extract_topics_from_filenames(self, course_id: str) -> List[str]:
        """Extract topics from uploaded file names - works for any subject"""
        try:
            from supabase import create_client
            SUPABASE_URL = os.getenv("SUPABASE_URL")
            SUPABASE_KEY = os.getenv("SUPABASE_KEY")
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            
            # Get filenames
            result = supabase.table("files").select("filename").eq("course_id", course_id).execute()
            
            if not result.data:
                return []
            
            topics = []
            for row in result.data:
                filename = row.get("filename", "")
                extracted_topic = self.extract_topic_from_filename(filename)
                if extracted_topic and extracted_topic not in topics:
                    topics.append(extracted_topic)
            
            return topics[:12]  # Reasonable limit
            
        except Exception as e:
            print(f"Failed to extract topics from filenames: {e}")
            return []
    
    def extract_topic_from_filename(self, filename: str) -> str:
        """Extract meaningful topic from a single filename"""
        print(f"  🔍 Processing filename: {filename}")
        
        # Remove file extension
        clean_name = re.sub(r'\.(pdf|docx|pptx|txt|md)$', '', filename, flags=re.IGNORECASE)
        print(f"    After extension removal: {clean_name}")
        
        # Remove common academic prefixes
        clean_name = re.sub(r'^(lecture|chapter|week|unit|lesson|section|module|assignment|homework|hw|lab|tutorial)\s*\d*\s*[-_:]?\s*', '', clean_name, flags=re.IGNORECASE)
        print(f"    After prefix removal: {clean_name}")
        
        # Remove common suffixes  
        clean_name = re.sub(r'\s*(part|section|chapter)\s*\d+$', '', clean_name, flags=re.IGNORECASE)
        clean_name = re.sub(r'\s*(in_class_activity|activity|exercise|solutions?|notes?)$', '', clean_name, flags=re.IGNORECASE)
        print(f"    After suffix removal: {clean_name}")
        
        # Clean up separators and formatting
        clean_name = re.sub(r'[-_]+', ' ', clean_name)
        clean_name = re.sub(r'\s+', ' ', clean_name)
        clean_name = clean_name.strip()
        print(f"    After separator cleanup: {clean_name}")
        
        # Capitalize properly
        if len(clean_name) > 2:
            # Handle special cases like "BSTs" or acronyms
            words = clean_name.split()
            formatted_words = []
            for word in words:
                if len(word) <= 4 and word.isupper():
                    formatted_words.append(word)  # Keep acronyms as-is
                else:
                    formatted_words.append(word.capitalize())
            
            result = ' '.join(formatted_words)
            print(f"    Final result: {result}")
            return result
        
        print(f"    Final result: (empty - too short)")
        return ""
    
    def extract_topics_from_content(self, course_id: str, vector_store) -> List[str]:
        """Analyze actual course content for topics using AI - subject agnostic"""
        try:
            # Get diverse content samples
            sample_embedding = [0.0] * 1536  # Dummy for metadata query
            results = vector_store.query(course_id, sample_embedding, top_k=25) or []
            
            if not results:
                return []
            
            # Get content from different documents if possible
            content_by_doc = {}
            for result in results:
                doc_name = result.get('doc_name', 'unknown')
                content = result.get('content', '').strip()
                if content and len(content) > 50:  # Meaningful content
                    if doc_name not in content_by_doc:
                        content_by_doc[doc_name] = []
                    content_by_doc[doc_name].append(content)
            
            # Sample from different docs to get variety
            combined_content = ""
            for doc_name, contents in list(content_by_doc.items())[:8]:  # Max 8 docs
                doc_sample = " ".join(contents[:3])  # Max 3 chunks per doc
                combined_content += f"\n[From {doc_name}]: {doc_sample[:800]}"  # Limit per doc
            
            if not combined_content.strip():
                return []
            
            # Use AI to extract topics (subject-agnostic)
            topics = self.analyze_content_for_topics_ai(combined_content)
            return topics
                
        except Exception as e:
            print(f"Failed to extract topics from content: {e}")
            return []
    
    def analyze_content_for_topics_ai(self, content: str) -> List[str]:
        """Use AI to analyze content and extract main topics - works for any subject"""
        try:
            prompt = f"""
            Analyze this educational course content and extract the main topics/concepts that would be suitable for practice questions.
            
            Course Content Sample:
            {content[:2500]}...
            
            INSTRUCTIONS:
            - Extract 6-10 specific, practice-worthy topics from this content
            - Topics should be concrete concepts that can have questions generated about them
            - Avoid overly broad topics like "Introduction" or "Overview"
            - Focus on substantive concepts, theories, methods, or subject matter
            - Make topics specific enough for meaningful practice questions
            - Use the actual terminology and concepts from the content
            
            Examples of good topics:
            - "Photosynthesis Process" (not just "Biology")
            - "Market Equilibrium" (not just "Economics") 
            - "Binary Search Trees" (not just "Computer Science")
            - "Renaissance Art Techniques" (not just "Art History")
            
            Return ONLY a JSON array of topic strings:
            ["Topic 1", "Topic 2", "Topic 3", ...]
            
            Extract topics that are specific, substantive, and suitable for quiz questions.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=300
            )
            
            content_response = response.choices[0].message.content.strip()
            
            # Clean up JSON response
            if content_response.startswith("```json"):
                content_response = content_response.replace("```json", "").replace("```", "").strip()
            
            topics = json.loads(content_response)
            
            # Validate and clean topics
            if isinstance(topics, list) and len(topics) > 0:
                clean_topics = []
                for topic in topics:
                    if isinstance(topic, str) and len(topic.strip()) > 3:
                        # Clean up the topic
                        cleaned = topic.strip()
                        # Remove quotes if wrapped
                        cleaned = re.sub(r'^["\']|["\']$', '', cleaned)
                        if len(cleaned) > 3 and cleaned not in clean_topics:
                            clean_topics.append(cleaned)
                return clean_topics[:10]
            
            return []
                
        except Exception as e:
            print(f"AI topic extraction failed: {e}")
            return []
    
    def get_course_context_topics(self, course_id: str) -> List[str]:
        """Try to get course context from course title or other metadata"""
        try:
            from supabase import create_client
            SUPABASE_URL = os.getenv("SUPABASE_URL")
            SUPABASE_KEY = os.getenv("SUPABASE_KEY")
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            
            # Get course info
            course_result = supabase.table("courses").select("title").eq("course_id", course_id).execute()
            
            if course_result.data:
                course_title = course_result.data[0].get("title", "")
                # Extract subject hints from course title
                return self.extract_topics_from_course_title(course_title)
            
            return []
            
        except Exception as e:
            print(f"Failed to get course context: {e}")
            return []
    
    def extract_topics_from_course_title(self, title: str) -> List[str]:
        """Extract subject area from course title"""
        # This could be enhanced with a subject classification system
        title_lower = title.lower()
        
        # Basic subject detection (can be expanded)
        subject_hints = {
            "data structures": ["Data Structures", "Algorithms", "Programming Concepts"],
            "calculus": ["Derivatives", "Integrals", "Limits", "Functions"],
            "physics": ["Mechanics", "Thermodynamics", "Electromagnetism"],
            "biology": ["Cell Biology", "Genetics", "Evolution", "Physiology"],
            "chemistry": ["Chemical Bonds", "Reactions", "Organic Chemistry"],
            "history": ["Historical Events", "Historical Analysis", "Timeline Studies"],
            "psychology": ["Cognitive Psychology", "Behavioral Psychology", "Research Methods"],
            "economics": ["Market Economics", "Microeconomics", "Macroeconomics"],
            "literature": ["Literary Analysis", "Literary Themes", "Writing Techniques"]
        }
        
        for subject, topics in subject_hints.items():
            if subject in title_lower:
                return topics
        
        return []
    
    def combine_and_rank_topics(self, filename_topics: List[str], 
                               content_topics: List[str], 
                               context_topics: List[str]) -> List[str]:
        """Intelligently combine topics from different sources - FIXED VERSION"""
        all_topics = []
        
        print(f"🔄 Combining topics:")
        print(f"  📁 Filename topics: {filename_topics}")
        print(f"  📖 Content topics: {content_topics}")
        print(f"  🎓 Context topics: {context_topics}")
        
        # Add ALL filename topics first (they're most reliable for structure)
        for topic in filename_topics:
            if topic and topic.strip():
                cleaned_topic = topic.strip()
                if cleaned_topic not in all_topics:
                    all_topics.append(cleaned_topic)
                    print(f"  ✅ Added filename topic: {cleaned_topic}")
        
        # Add content topics that don't significantly overlap with filename topics
        for topic in content_topics:
            if topic and topic.strip():
                cleaned_topic = topic.strip()
                # Use more lenient similarity check
                if not any(self.topics_are_very_similar(cleaned_topic, existing) for existing in all_topics):
                    all_topics.append(cleaned_topic)
                    print(f"  ✅ Added content topic: {cleaned_topic}")
                else:
                    print(f"  ⏭️ Skipped similar content topic: {cleaned_topic}")
        
        # Add context topics only if we don't have enough topics
        if len(all_topics) < 8:
            for topic in context_topics:
                if topic and topic.strip():
                    cleaned_topic = topic.strip()
                    if not any(self.topics_are_very_similar(cleaned_topic, existing) for existing in all_topics):
                        all_topics.append(cleaned_topic)
                        print(f"  ✅ Added context topic: {cleaned_topic}")
        
        print(f"🎯 Final combined topics: {all_topics}")
        return all_topics
    
    def topics_are_similar(self, topic1: str, topic2: str) -> bool:
        """Check if two topics are similar enough to be considered duplicates - LESS AGGRESSIVE"""
        # Make this much less aggressive
        t1_words = set(topic1.lower().split())
        t2_words = set(topic2.lower().split())
        
        # Only consider them similar if they share most words AND are reasonably similar in length
        shared_words = len(t1_words & t2_words)
        min_words = min(len(t1_words), len(t2_words))
        
        # Only mark as similar if 80%+ words match AND both topics are short
        if shared_words >= min_words * 0.8 and min_words <= 3:
            return True
        
        # Exact match check
        if topic1.lower().strip() == topic2.lower().strip():
            return True
            
        return False

    def topics_are_very_similar(self, topic1: str, topic2: str) -> bool:
        """Even more strict similarity check for content vs filename topics"""
        t1_clean = topic1.lower().strip()
        t2_clean = topic2.lower().strip()
        
        # Exact match
        if t1_clean == t2_clean:
            return True
        
        # One is contained in the other (for short topics)
        if len(t1_clean) <= 15 and len(t2_clean) <= 15:
            if t1_clean in t2_clean or t2_clean in t1_clean:
                return True
        
        # High word overlap for very short topics only
        t1_words = set(t1_clean.split())
        t2_words = set(t2_clean.split())
        
        if len(t1_words) <= 2 and len(t2_words) <= 2:
            shared = len(t1_words & t2_words)
            if shared >= max(len(t1_words), len(t2_words)) * 0.8:
                return True
        
        return False
    
    def clean_and_limit_topics(self, topics: List[str], max_topics: int = 15) -> List[str]:
        """Clean up and limit the final topic list - LESS RESTRICTIVE"""
        cleaned = []
        for topic in topics:
            # Basic cleaning
            cleaned_topic = topic.strip()
            cleaned_topic = re.sub(r'\s+', ' ', cleaned_topic)
            
            # Less restrictive filtering
            if len(cleaned_topic) > 2 and cleaned_topic not in cleaned:
                # Only avoid very generic single words
                very_generic = ['intro', 'overview', 'basic', 'general', 'notes', 'activity']
                topic_lower = cleaned_topic.lower()
                
                # Only skip if it's a single generic word
                if not (len(cleaned_topic.split()) == 1 and topic_lower in very_generic):
                    cleaned.append(cleaned_topic)
                    print(f"  ✅ Cleaned topic kept: {cleaned_topic}")
                else:
                    print(f"  ❌ Filtered out generic topic: {cleaned_topic}")
        
        # Increase max topics to 15
        final_topics = cleaned[:max_topics]
        print(f"🏁 Final cleaned topics ({len(final_topics)}): {final_topics}")
        return final_topics
    
    def get_generic_fallback_topics(self, course_id: str) -> List[str]:
        """Generic fallback topics when all extraction fails"""
        return [
            "Course Fundamentals",
            "Key Concepts", 
            "Core Topics",
            "Main Principles",
            "Important Methods",
            "Essential Knowledge"
        ]
    
    def generate_practice_problems(self, course_id: str, topic: str, 
                                 difficulty: str = "medium", count: int = 5) -> List[Dict[str, Any]]:
        """Generate practice problems - FIXED to use EXACT same search as study chat"""
        try:
            print(f"🎯 Generating {count} {difficulty.upper()} difficulty problems for: '{topic}'")
            
            # FIXED: Use the EXACT same search engine as study chat
            from query_engine import advanced_rag_engine
            
            print(f"🔍 Using study chat's search engine for: '{topic}'")
            
            # Let the study chat engine do the heavy lifting
            # It will handle embedding, query expansion, reranking, etc.
            results = advanced_rag_engine.hybrid_search([topic], course_id, top_k=12)
            
            print(f"📚 Study chat engine found: {len(results) if results else 0} content chunks")
            
            # Debug: Show what we found
            if results:
                print("📄 Content sources:")
                for i, result in enumerate(results[:3]):
                    doc = result.get("doc_name", "unknown")
                    content_preview = result.get("content", "")[:100] + "..."
                    similarity = result.get("similarity", result.get("relevance_score", 0))
                    print(f"  {i+1}. {doc} (sim: {similarity:.3f}): {content_preview}")
            else:
                print("❌ No content found even with study chat's search!")
                print("   This means the topic might not be covered in your course materials")
            
            # Generate problems based on what we found
            if results and len(results) > 0:
                print(f"✅ Using course content to generate problems")
                context = self.create_universal_context(results, topic)
                problems = self.create_problems_universal_ai(topic, context, difficulty, count)
            else:
                print(f"⚠️ No course content found, using general knowledge")
                problems = self.internet_fallback_problems(topic, count, difficulty)
            
            # Validate all problems have correct difficulty
            for problem in problems:
                problem["difficulty"] = difficulty
                problem["topic"] = topic
            
            print(f"✅ Generated {len(problems)} {difficulty} difficulty problems for '{topic}'")
            return problems
            
        except Exception as e:
            print(f"❌ Failed to generate practice problems: {e}")
            import traceback
            traceback.print_exc()
            return self.ultimate_fallback_problems(topic, count, difficulty)

    def create_smart_search_queries(self, topic: str) -> List[str]:
        """Create smarter search queries based on the topic"""
        queries = []
        
        # Original topic
        queries.append(topic)
        
        # Clean up the topic for better matching
        topic_clean = topic.lower().strip()
        
        # Handle specific cases
        if "bst" in topic_clean:
            queries.extend([
                "binary search tree",
                "BST operations",
                "tree insertion deletion",
                "binary tree search"
            ])
        elif "stack" in topic_clean:
            queries.extend([
                "stack data structure",
                "stack operations",
                "push pop operations",
                "LIFO structure"
            ])
        elif "tree" in topic_clean and "part" in topic_clean:
            queries.extend([
                "binary tree",
                "tree traversal",
                "tree terminology",
                "tree structure"
            ])
        elif "graph" in topic_clean:
            queries.extend([
                "graph data structure",
                "graph algorithms",
                "adjacency matrix",
                "graph traversal"
            ])
        elif "heap" in topic_clean or "priority queue" in topic_clean:
            queries.extend([
                "heap data structure",
                "priority queue",
                "binary heap",
                "heap operations"
            ])
        elif "hash" in topic_clean:
            queries.extend([
                "hash table",
                "hashing function",
                "collision resolution",
                "hash map"
            ])
        elif "sort" in topic_clean:
            queries.extend([
                "sorting algorithms",
                "merge sort",
                "quick sort",
                "sorting comparison"
            ])
        elif "algorithm" in topic_clean and "analysis" in topic_clean:
            queries.extend([
                "time complexity",
                "space complexity",
                "big O notation",
                "algorithm efficiency"
            ])
        elif "array" in topic_clean or "linked" in topic_clean:
            queries.extend([
                "array operations",
                "linked list",
                "list implementation",
                "dynamic array"
            ])
        else:
            # Generic fallbacks
            queries.extend([
                f"{topic} implementation",
                f"{topic} operations",
                f"{topic} algorithm"
            ])
        
        # Remove duplicates while preserving order
        unique_queries = []
        for q in queries:
            if q not in unique_queries:
                unique_queries.append(q)
        
        return unique_queries[:6]  # Limit to 6 queries

    def fallback_content_search(self, course_id: str, topic: str, vector_store) -> List[Dict]:
        """Last resort: try to find ANY content that might be relevant"""
        try:
            print(f"🔄 Trying fallback content search for: '{topic}'")
            
            # Extract key words from the topic
            topic_words = topic.lower().split()
            fallback_queries = []
            
            # Try individual words
            for word in topic_words:
                if len(word) > 3:  # Skip short words
                    fallback_queries.append(word)
            
            # Try common variations
            if "bst" in topic.lower():
                fallback_queries.extend(["tree", "binary", "search"])
            elif "stack" in topic.lower():
                fallback_queries.extend(["stack", "push", "pop"])
            
            all_results = []
            for query in fallback_queries[:4]:  # Limit fallback searches
                try:
                    print(f"  🔍 Fallback search: '{query}'")
                    emb_resp = self.openai_client.embeddings.create(
                        model="text-embedding-ada-002",
                        input=[query]
                    )
                    results = vector_store.query(course_id, emb_resp.data[0].embedding, top_k=5)
                    if results:
                        print(f"    ✅ Fallback found {len(results)} results")
                        all_results.extend(results)
                except Exception as e:
                    print(f"    ❌ Fallback search failed: {e}")
            
            return self.deduplicate_results(all_results)
            
        except Exception as e:
            print(f"❌ Fallback search failed: {e}")
            return []
    
    def deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate content from search results"""
        seen_content = set()
        unique_results = []
        
        for result in results:
            content = result.get("content", "").strip()
            content_hash = hash(content[:200])  # Use first 200 chars as signature
            
            if content and content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)
                if len(unique_results) >= 8:  # Reasonable limit
                    break
        
        return unique_results
    
    def create_universal_context(self, results: List[Dict], topic: str) -> str:
        """Create context that works for any subject - IMPROVED"""
        if not results:
            return f"No specific course materials found for {topic}."
        
        context_parts = [f"COURSE MATERIALS RELATED TO {topic.upper()}:"]
        
        for i, result in enumerate(results[:8], 1):  # Use more results
            content = result.get("content", "").strip()
            doc = result.get("doc_name", "unknown")
            page = result.get("page")
            
            source_info = f"[Source {i}: {doc}"
            if page:
                source_info += f", page {page}"
            source_info += "]"
            
            context_parts.append(f"\n{source_info}")
            context_parts.append(content)
            context_parts.append("---")
        
        full_context = "\n".join(context_parts)
        
        # If context is very short, mention that
        if len(full_context) < 500:
            context_parts.insert(1, f"\nNote: Limited course material found for {topic}. Using available related content:")
        
        return "\n".join(context_parts)
    
    def create_problems_universal_ai(self, topic: str, context: str, 
                                   difficulty: str, count: int) -> List[Dict[str, Any]]:
        """Enhanced AI problem generation with better difficulty handling"""
        
        # Enhanced difficulty guidelines
        difficulty_specs = {
            "easy": {
                "description": "Basic recall, definitions, and simple conceptual understanding",
                "cognitive_level": "Remember and Understand",
                "question_types": "Multiple choice with clear distinctions, true/false, basic definitions",
                "complexity": "Single concept, direct application, no multi-step reasoning",
                "bloom_level": "Knowledge and Comprehension"
            },
            "medium": {
                "description": "Application, analysis, and problem-solving with moderate complexity",
                "cognitive_level": "Apply and Analyze", 
                "question_types": "Scenario-based questions, comparisons, step-by-step problems",
                "complexity": "Multi-step reasoning, connecting concepts, real-world applications",
                "bloom_level": "Application and Analysis"
            },
            "hard": {
                "description": "Synthesis, evaluation, and complex critical thinking",
                "cognitive_level": "Evaluate and Create",
                "question_types": "Design problems, complex scenarios, optimization, trade-offs",
                "complexity": "Advanced reasoning, multiple concepts, edge cases, creative solutions",
                "bloom_level": "Synthesis and Evaluation"
            }
        }
        
        spec = difficulty_specs.get(difficulty, difficulty_specs["medium"])
        
        prompt = f"""
        Create {count} high-quality educational practice problems about "{topic}" at {difficulty.upper()} difficulty level.
        
        COURSE CONTENT:
        {context}
        
        DIFFICULTY REQUIREMENTS FOR {difficulty.upper()}:
        - Focus: {spec['description']}
        - Cognitive Level: {spec['cognitive_level']}
        - Question Types: {spec['question_types']}
        - Complexity: {spec['complexity']}
        - Bloom's Taxonomy: {spec['bloom_level']}
        
        SPECIFIC GUIDELINES:
        - Base questions STRICTLY on the provided course material
        - Use specific concepts, examples, and terminology from the course content
        - Ensure questions match the {difficulty} difficulty level appropriately
        - Make distractors (wrong answers) plausible but clearly incorrect
        - Provide thorough explanations that teach the concept
        
        For each problem:
        1. Write a clear question appropriate for {difficulty} level
        2. Provide four plausible multiple choice options (A, B, C, D)
        3. Indicate the correct answer letter
        4. Give a thorough explanation that references course material and explains why other options are wrong
        5. Estimate time needed based on difficulty
        
        Return as JSON array:
        [{{
            "question": "question text appropriate for {difficulty} difficulty about {topic}",
            "options": ["A) option1", "B) option2", "C) option3", "D) option4"],
            "correct_answer": "A",
            "explanation": "detailed explanation referencing course material and explaining why other options are incorrect",
            "estimated_time": "{self.get_time_estimate_by_difficulty(difficulty)}",
            "difficulty": "{difficulty}",
            "topic": "{topic}"
        }}]
        
        CRITICAL: Questions must be appropriate for {difficulty} difficulty and based on actual course content.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4 if difficulty == "hard" else 0.3,  # Slightly more creative for hard questions
                max_tokens=4000  # More tokens for detailed explanations
            )
            
            content = response.choices[0].message.content.strip()
            
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
                
            problems = json.loads(content)
            
            # Validate problems
            if isinstance(problems, list) and len(problems) > 0:
                validated = []
                for problem in problems:
                    if (isinstance(problem, dict) and 
                        all(key in problem for key in ["question", "options", "correct_answer", "explanation"])):
                        # Ensure difficulty is set
                        problem["difficulty"] = difficulty
                        problem["topic"] = topic
                        validated.append(problem)
                
                if validated:
                    print(f"✅ Generated {len(validated)} {difficulty} difficulty problems")
                    return validated
            
            print(f"⚠️ AI generation failed, trying internet fallback...")
            return self.internet_fallback_problems(topic, count, difficulty)
                
        except Exception as e:
            print(f"❌ AI problem generation failed: {e}")
            return self.internet_fallback_problems(topic, count, difficulty)

    def get_time_estimate_by_difficulty(self, difficulty: str) -> str:
        """Get appropriate time estimates based on difficulty"""
        estimates = {
            "easy": "1-2 minutes",
            "medium": "3-5 minutes", 
            "hard": "7-12 minutes"
        }
        return estimates.get(difficulty, "3-5 minutes")

    def internet_fallback_problems(self, topic: str, count: int, difficulty: str) -> List[Dict[str, Any]]:
        """Generate problems using general knowledge when course content is insufficient"""
        
        print(f"🌐 Using internet fallback for {difficulty} {topic} questions...")
        
        # Enhanced difficulty-specific prompts
        difficulty_prompts = {
            "easy": f"""
            Create {count} EASY practice questions about {topic} for computer science students.
            
            EASY LEVEL REQUIREMENTS:
            - Focus on basic definitions and fundamental concepts
            - Simple recall and recognition questions
            - Clear, unambiguous answer choices
            - Basic terminology and concepts
            
            Topics to cover for {topic}:
            - Basic definitions and properties
            - Fundamental operations
            - Simple examples and use cases
            - Key terminology
            """,
            
            "medium": f"""
            Create {count} MEDIUM difficulty practice questions about {topic} for computer science students.
            
            MEDIUM LEVEL REQUIREMENTS:
            - Application of concepts to scenarios
            - Analyze and compare different approaches
            - Multi-step problem solving
            - Understanding of trade-offs and implications
            
            Topics to cover for {topic}:
            - Implementation details
            - Performance characteristics
            - Real-world applications
            - Algorithmic analysis
            """,
            
            "hard": f"""
            Create {count} HARD practice questions about {topic} for computer science students.
            
            HARD LEVEL REQUIREMENTS:
            - Complex scenarios requiring deep understanding
            - Design and optimization problems
            - Edge cases and advanced considerations
            - Integration of multiple concepts
            
            Topics to cover for {topic}:
            - Advanced implementations
            - Optimization strategies
            - Complex problem scenarios
            - Advanced algorithmic considerations
            """
        }
        
        prompt = difficulty_prompts.get(difficulty, difficulty_prompts["medium"])
        
        prompt += f"""
        
        For each question:
        1. Create a {difficulty}-appropriate question about {topic}
        2. Provide four plausible multiple choice options (A, B, C, D)
        3. Indicate the correct answer letter
        4. Provide a detailed explanation suitable for learning
        5. Include appropriate time estimate
        
        Return as JSON array:
        [{{
            "question": "{difficulty} difficulty question about {topic}",
            "options": ["A) option1", "B) option2", "C) option3", "D) option4"],
            "correct_answer": "A",
            "explanation": "detailed educational explanation",
            "estimated_time": "{self.get_time_estimate_by_difficulty(difficulty)}",
            "difficulty": "{difficulty}",
            "topic": "{topic}"
        }}]
        
        Make questions educationally valuable and appropriate for {difficulty} difficulty level.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": f"You are an expert computer science educator creating {difficulty} difficulty practice questions. Use your knowledge to create educational, appropriate questions even without specific course materials."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=3500
            )
            
            content = response.choices[0].message.content.strip()
            
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
                
            problems = json.loads(content)
            
            # Validate and mark as internet fallback
            if isinstance(problems, list) and len(problems) > 0:
                validated = []
                for problem in problems:
                    if (isinstance(problem, dict) and 
                        all(key in problem for key in ["question", "options", "correct_answer", "explanation"])):
                        # Mark as internet fallback
                        problem["difficulty"] = difficulty
                        problem["topic"] = topic
                        problem["source"] = "general_knowledge"
                        # Add note to explanation
                        problem["explanation"] += f"\n\nNote: This question uses general computer science knowledge about {topic}. For course-specific questions, ensure your course materials cover this topic in detail."
                        validated.append(problem)
                
                if validated:
                    print(f"✅ Generated {len(validated)} internet fallback problems ({difficulty} difficulty)")
                    return validated
            
            # Ultimate fallback
            return self.ultimate_fallback_problems(topic, count, difficulty)
            
        except Exception as e:
            print(f"❌ Internet fallback failed: {e}")
            return self.ultimate_fallback_problems(topic, count, difficulty)

    def ultimate_fallback_problems(self, topic: str, count: int, difficulty: str) -> List[Dict[str, Any]]:
        """Last resort fallback with difficulty-appropriate questions"""
        
        difficulty_templates = {
            "easy": {
                "question": f"Which of the following best describes {topic}?",
                "options": [
                    f"A) {topic} is a fundamental concept in computer science",
                    f"B) {topic} is only used in advanced programming",
                    f"C) {topic} is not relevant to data structures",
                    f"D) {topic} is only theoretical with no practical use"
                ],
                "correct": "A",
                "explanation": f"{topic} is indeed a fundamental concept. This is a basic recall question appropriate for easy difficulty."
            },
            "medium": {
                "question": f"When implementing {topic}, which factor is most important to consider?",
                "options": [
                    "A) Memory usage and time complexity",
                    "B) Only the programming language used", 
                    "C) The color of the code editor",
                    "D) The day of the week when coding"
                ],
                "correct": "A",
                "explanation": "Memory usage and time complexity are crucial factors when implementing any data structure or algorithm. This requires understanding of performance implications."
            },
            "hard": {
                "question": f"In a complex system, how would you optimize {topic} for both space and time efficiency while maintaining correctness?",
                "options": [
                    "A) Analyze trade-offs, consider use patterns, and implement adaptive solutions",
                    "B) Always choose the fastest algorithm regardless of memory",
                    "C) Use the simplest implementation without optimization",
                    "D) Optimization is not necessary for complex systems"
                ],
                "correct": "A", 
                "explanation": "Complex optimization requires analyzing trade-offs, understanding usage patterns, and potentially implementing adaptive solutions. This demonstrates high-level thinking required for hard difficulty."
            }
        }
        
        template = difficulty_templates.get(difficulty, difficulty_templates["medium"])
        
        base_problem = {
            "question": template["question"],
            "options": template["options"],
            "correct_answer": template["correct"],
            "explanation": f"{template['explanation']} Upload specific course materials about {topic} to get detailed, course-specific practice questions.",
            "estimated_time": self.get_time_estimate_by_difficulty(difficulty),
            "difficulty": difficulty,
            "topic": topic,
            "source": "fallback"
        }
        
        return [base_problem.copy() for _ in range(count)]