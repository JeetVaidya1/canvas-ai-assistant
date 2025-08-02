# quiz_assistant_engine.py - ENHANCED WITH WEB FALLBACK
import os
import re
import json
from typing import Dict, List, Any
from dotenv import load_dotenv
from openai import OpenAI
from vector_store import VectorStore

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
vector_store = VectorStore()

def parse_quiz_question(raw_text: str) -> Dict:
    """Parse quiz question using AI"""
    parsing_prompt = f"""
    Parse this quiz question and return ONLY valid JSON:
    {{
        "question_text": "the main question",
        "question_type": "multiple_choice|true_false|short_answer|essay|fill_blank",
        "options": ["option1", "option2", ...] or null,
        "topic": "likely academic topic",
        "difficulty": "easy|medium|hard"
    }}
    
    Question: {raw_text}
    
    Return ONLY the JSON, no other text:
    """
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": parsing_prompt}],
            temperature=0.1,
            max_tokens=500
        )
        content = response.choices[0].message.content.strip()
        
        # Try to extract JSON if it's wrapped in text
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()
        
        return json.loads(content)
    except Exception as e:
        print(f"❌ Parsing failed: {e}")
        # Fallback: simple heuristic parsing
        if any(marker in raw_text.lower() for marker in ['a)', 'b)', 'c)', 'd)', '(a)', '(b)']):
            return {
                "question_text": raw_text,
                "question_type": "multiple_choice",
                "options": extract_options_heuristic(raw_text),
                "topic": "unknown",
                "difficulty": "medium"
            }
        else:
            return {
                "question_text": raw_text,
                "question_type": "short_answer",
                "options": None,
                "topic": "unknown",
                "difficulty": "medium"
            }

def extract_options_heuristic(text: str) -> List[str]:
    """Extract multiple choice options using regex"""
    try:
        # Look for patterns like "A) text", "B) text", etc.
        options = re.findall(r'[A-D]\)\s*([^A-D\n]+)', text, re.IGNORECASE)
        if options:
            return [opt.strip() for opt in options]
        
        # Look for patterns like "(A) text", "(B) text", etc.
        options = re.findall(r'\([A-D]\)\s*([^(]+?)(?=\([A-D]\)|$)', text, re.IGNORECASE)
        if options:
            return [opt.strip() for opt in options]
            
        return []
    except:
        return []

def find_relevant_materials(question: str, course_id: str) -> List[Dict]:
    """Find relevant course materials"""
    try:
        emb_resp = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=[question]
        )
        results = vector_store.query(course_id, emb_resp.data[0].embedding, top_k=5) or []
        return results
    except Exception as e:
        print(f"❌ Materials search failed: {e}")
        return []

def web_search_fallback(question: str, topic: str = None) -> str:
    """Use AI to provide general knowledge when course materials are insufficient"""
    
    search_prompt = f"""You are an expert tutor answering a quiz question. The student doesn't have specific course materials for this topic, so provide your best answer based on general academic knowledge.

Question: {question}
{f"Topic: {topic}" if topic else ""}

Provide a comprehensive answer that includes:
1. The correct answer with reasoning
2. Brief explanation of key concepts
3. Why other options (if multiple choice) are incorrect

Be educational and cite that this is based on general academic knowledge since specific course materials weren't available.
"""
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": search_prompt}],
            temperature=0.2,
            max_tokens=800
        )
        
        web_answer = response.choices[0].message.content
        return f"[Based on general academic knowledge]\n\n{web_answer}"
        
    except Exception as e:
        print(f"❌ Web fallback failed: {e}")
        return "I don't have enough specific information in your course materials to answer this question accurately. Consider uploading more relevant materials or consulting your textbook."

def generate_quiz_response(parsed_question: Dict, materials: List[Dict], course_id: str) -> Dict:
    """Generate intelligent quiz response with web fallback"""
    
    # Build context from materials
    context = ""
    source_materials = []
    if materials:
        context_parts = []
        for i, material in enumerate(materials[:3], 1):
            content = material.get('content', '')[:500]
            doc_name = material.get('doc_name', 'Unknown')
            context_parts.append(f"Source {i} ({doc_name}): {content}")
            source_materials.append(doc_name)
        context = "\n\n".join(context_parts)
    
    question_type = parsed_question.get("question_type", "short_answer")
    question_text = parsed_question.get("question_text", "")
    options = parsed_question.get("options", [])
    topic = parsed_question.get("topic", "")
    
    # Check if we have sufficient course materials
    has_sufficient_materials = len(materials) > 0 and any(
        len(m.get('content', '')) > 100 for m in materials
    )
    
    if not has_sufficient_materials:
        print("🌐 Using web fallback - insufficient course materials")
        web_answer = web_search_fallback(question_text, topic)
        
        return {
            "status": "success",
            "answer": "Based on general academic knowledge (course materials not found)",
            "explanation": web_answer,
            "confidence": 0.7,  # Lower confidence for web fallback
            "question_type": question_type,
            "study_tips": [
                "📚 Upload relevant course materials for more accurate answers",
                "🔍 Verify this answer with your textbook or course notes",
                "💡 This answer is based on general knowledge, not your specific course"
            ],
            "similar_concepts": [],
            "estimated_time": get_time_estimate(question_type),
            "relevant_sources": ["General Academic Knowledge", "AI Knowledge Base"]
        }
    
    # Create specialized prompt based on question type
    if question_type == "multiple_choice" and options:
        options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
        instruction = f"""This is a multiple choice question. Analyze each option and choose the correct answer.
        
Question: {question_text}
Options:
{options_text}

Provide the correct letter (A, B, C, or D) and explain why it's correct."""
        
    elif question_type == "true_false":
        instruction = f"""This is a true/false question. Determine if the statement is true or false.
        
Statement: {question_text}

Answer with "True" or "False" and explain your reasoning."""
        
    else:
        instruction = f"""Answer this question clearly and concisely.
        
Question: {question_text}

Provide a direct answer with explanation."""
    
    prompt = f"""You are an expert tutor helping a student with a quiz question.

{instruction}

Relevant course materials:
{context}

You MUST respond with ONLY valid JSON in this exact format:
{{
    "answer": "Your direct answer here",
    "explanation": "Clear explanation of why this is correct",
    "confidence": 0.85,
    "study_tips": ["tip1", "tip2"]
}}

Do not include any other text, markdown, or formatting. Only return the JSON object."""
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1000
        )
        
        content = response.choices[0].message.content.strip()
        
        # Clean up the response
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()
        
        # Try to parse JSON
        try:
            result = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"❌ JSON parse error: {e}")
            # Return a safe fallback response
            return {
                "status": "success",
                "answer": "Found relevant information in your course materials",
                "explanation": f"Based on your course materials: {context[:300]}...",
                "confidence": 0.6,
                "question_type": question_type,
                "study_tips": ["Review the course materials for this topic"],
                "similar_concepts": [],
                "estimated_time": get_time_estimate(question_type),
                "relevant_sources": source_materials
            }
        
        return {
            "status": "success",
            "answer": result.get("answer", "Unable to determine"),
            "explanation": result.get("explanation", "No explanation available"),
            "confidence": result.get("confidence", 0.8),  # Higher confidence with course materials
            "question_type": question_type,
            "study_tips": result.get("study_tips", []),
            "similar_concepts": [],
            "estimated_time": get_time_estimate(question_type),
            "relevant_sources": source_materials
        }
        
    except Exception as e:
        print(f"❌ Response generation error: {e}")
        # Try web fallback if course materials failed
        print("🌐 Trying web fallback due to processing error")
        web_answer = web_search_fallback(question_text, topic)
        
        return {
            "status": "success",
            "answer": "Based on general knowledge (processing error with course materials)",
            "explanation": web_answer,
            "confidence": 0.6,
            "question_type": question_type,
            "study_tips": [
                "🔄 Try rephrasing the question",
                "📚 Upload more relevant course materials",
                "✅ This answer is based on general academic knowledge"
            ],
            "similar_concepts": [],
            "estimated_time": get_time_estimate(question_type),
            "relevant_sources": ["General Academic Knowledge"]
        }

def get_time_estimate(question_type: str) -> str:
    """Estimate time needed"""
    time_map = {
        "multiple_choice": "2-3 minutes",
        "true_false": "1-2 minutes", 
        "short_answer": "3-5 minutes",
        "essay": "15-30 minutes",
        "fill_blank": "2-4 minutes"
    }
    return time_map.get(question_type, "5-10 minutes")

def assist_with_quiz_question(question_text: str, course_id: str, session_id: str = None) -> Dict:
    """Main entry point for quiz assistance"""
    try:
        print(f"🎯 Processing quiz question: {question_text[:100]}...")
        
        # Parse the question
        parsed = parse_quiz_question(question_text)
        print(f"📝 Parsed as {parsed.get('question_type')}")
        
        # Find relevant materials
        materials = find_relevant_materials(question_text, course_id)
        print(f"📚 Found {len(materials)} relevant materials")
        
        # Generate response (with automatic web fallback if needed)
        response = generate_quiz_response(parsed, materials, course_id)
        print(f"✅ Generated response with {response.get('confidence', 0):.0%} confidence")
        
        return response
        
    except Exception as e:
        print(f"❌ Quiz assistance error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "answer": "Processing error occurred",
            "explanation": "Please try again or rephrase your question.",
            "confidence": 0.0,
            "question_type": "unknown",
            "study_tips": [],
            "similar_concepts": [],
            "estimated_time": "",
            "relevant_sources": []
        }