"""Topic extraction mixin for practice generation.

Extracts practice-worthy topics from course materials using multiple
strategies (filenames, AI content analysis, course-title context) and
combines/cleans them into a final ranked list. Subject-agnostic.
"""
import os
import json
import re
from typing import List


class TopicsMixin:
    """Multi-strategy topic extraction and combination logic."""

    def extract_topics_from_course(self, course_id: str) -> List[str]:
        """Course topics for a course — Course Brain first, legacy strategies last.

        The Course Brain (course_brain.py / course_topics table) is the single
        source of truth: content-grounded, persisted, rebuilt on ingest. The
        legacy filename/content/context strategies below survive only as a
        last-resort fallback for courses where synthesis is impossible.
        """
        try:
            import course_brain
            names = course_brain.topic_names(course_id, auto_generate=True)
            if names:
                return names
        except Exception as e:  # noqa: BLE001
            print(f"Course Brain topics unavailable, using legacy extraction: {e}")
        return self._extract_topics_legacy(course_id)

    def _extract_topics_legacy(self, course_id: str) -> List[str]:
        """LEGACY multi-strategy extraction (filenames/content/title). Fallback only."""
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
            sample_embedding = [0.0] * 1024  # Dummy for metadata query (BGE-large dim)
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
