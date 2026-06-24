"""Problem-generation mixin for practice generation.

Orchestrates adaptive-difficulty problem generation grounded in retrieved
course content, with structured-output AI generation plus internet-knowledge
and ultimate static fallbacks.
"""
import os
import json
from typing import List, Dict, Any

from .schemas import PROBLEM_SCHEMA
from .difficulty import route_difficulty_by_mastery
from providers import structured_call


class GenerationMixin:
    """Practice-problem generation, structured AI calls, and fallbacks."""

    def generate_practice_problems(self, course_id: str, topic: str,
                                 difficulty: str = "adaptive", count: int = 5,
                                 user_id: str = "anonymous") -> List[Dict[str, Any]]:
        """Generate practice problems grounded in course material.

        When ``difficulty == 'adaptive'`` the level is chosen from the user's
        mastery of the topic (weaker -> easier, mastered -> harder).
        """
        try:
            if difficulty == "adaptive":
                mastery = self.lookup_topic_mastery(course_id, topic, user_id)
                difficulty = route_difficulty_by_mastery(mastery)
                print(f"🎚️ Adaptive difficulty for '{topic}': mastery={mastery:.2f} -> {difficulty}")
            if difficulty not in ("easy", "medium", "hard"):
                difficulty = "medium"

            print(f"🎯 Generating {count} {difficulty.upper()} difficulty problems for: '{topic}'")

            # Canonical hybrid + reranked retrieval (same as the rest of Phase 3).
            from rag.retrieval import retrieve
            results = retrieve(topic, course_id, top_k=12)
            print(f"📚 Retrieval found: {len(results) if results else 0} content chunks")

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
            # Guaranteed-schema tool use — no regex/JSON parsing of model text.
            out = structured_call(
                [{"role": "user", "content": prompt}],
                schema=PROBLEM_SCHEMA,
                tool_name="practice_problems",
                model=os.getenv("MODEL_COMPLEX"),
                max_tokens=4000,
            )
            problems = out.get("problems", []) if isinstance(out, dict) else []

            validated = []
            for problem in problems:
                if (isinstance(problem, dict) and
                        all(key in problem for key in ["question", "options", "correct_answer", "explanation"])):
                    problem.setdefault("estimated_time", self.get_time_estimate_by_difficulty(difficulty))
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
