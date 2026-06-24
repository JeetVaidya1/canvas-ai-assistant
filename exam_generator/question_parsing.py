# exam_generator/question_parsing.py - Regex-based question extraction and classification
import re
from typing import Any, Dict, List, Optional


class QuestionParsingMixin:
    """Heuristic extraction and classification of questions from raw exam text."""

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
        return "medium"  # neutral default; respect the requested difficulty elsewhere

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
