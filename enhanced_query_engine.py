# enhanced_query_engine.py - Clean, natural conversational responses

import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from vector_store import VectorStore
from response_formatter import format_ai_response  # Updated import

# Load keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)
vector_store = VectorStore()

def classify_question_simple(question: str) -> str:
    """Simple question classification"""
    question_lower = question.lower()
    
    if any(word in question_lower for word in ['what is', 'define', 'definition', 'meaning']):
        return 'definition'
    elif any(word in question_lower for word in ['how', 'explain', 'why', 'process']):
        return 'explanation'  
    elif any(word in question_lower for word in ['example', 'show me', 'demonstrate']):
        return 'example'
    elif any(word in question_lower for word in ['compare', 'difference', 'versus', 'vs']):
        return 'comparison'
    elif any(word in question_lower for word in ['apply', 'use', 'implement']):
        return 'application'
    else:
        return 'explanation'

def generate_natural_prompt(question: str, question_type: str, context: str) -> str:
    """Create a natural, conversational prompt that produces clean responses"""
    
    # Simple, natural instructions based on question type
    if question_type == 'definition':
        instruction = """Provide a clear definition and explanation. Keep it conversational and natural."""
    elif question_type == 'explanation':
        instruction = """Explain this concept clearly and naturally, like you're talking to a student. Use examples when helpful."""
    elif question_type == 'example':
        instruction = """Provide clear examples and explain why they're relevant. Keep it conversational."""
    elif question_type == 'comparison':
        instruction = """Compare these concepts naturally, highlighting key differences and similarities."""
    else:
        instruction = """Answer the question clearly and naturally, as if you're having a conversation with a student."""

    # Clean, natural prompt without excessive formatting instructions
    enhanced_prompt = f"""You're a knowledgeable tutor having a conversation with a student. Answer their question naturally and clearly.

Student's question: {question}

Course materials available:
{context}

Instructions: {instruction}

Guidelines:
- Write naturally and conversationally
- Use the course materials as your primary source
- Reference specific diagrams or images when relevant (e.g., "The diagram shows...")
- Keep explanations clear but not overly structured
- Use examples to illustrate points
- Be helpful and educational

Answer the student's question:"""
    
    return enhanced_prompt

def enhanced_ask_question(question: str, course_id: str) -> str:
    """
    Enhanced question answering with clean, natural formatting
    """
    try:
        print(f"🤖 Enhanced RAG processing: {question}")
        
        # Step 1: Classify question type
        question_type = classify_question_simple(question)
        print(f"📝 Question type: {question_type}")
        
        # Step 2: Get embeddings and search
        emb_resp = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=[question]
        )
        query_embedding = emb_resp.data[0].embedding
        
        # Step 3: Search with higher k to get both text and images
        try:
            results = vector_store.query(course_id, query_embedding, top_k=8) or []
            print(f"🔍 Found {len(results)} results")
        except Exception:
            results = []
        
        if not results:
            return "I don't have enough information in your course materials to answer this question. Please make sure you've uploaded relevant files for this topic."
        
        # Step 4: Separate text and visual content
        text_chunks = []
        image_chunks = []
        
        for result in results:
            content = result.get("content", "")
            if "[IMAGE CONTENT" in content:
                image_chunks.append(result)
            else:
                text_chunks.append(result)
        
        print(f"📄 Text chunks: {len(text_chunks)}, 🖼️ Image chunks: {len(image_chunks)}")
        
        # Step 5: Build enhanced context
        context_parts = []
        
        if text_chunks:
            context_parts.append("=== TEXT FROM COURSE MATERIALS ===")
            for i, chunk in enumerate(text_chunks[:5]):
                context_parts.append(f"Text {i+1}: {chunk.get('content', '')}")
        
        if image_chunks:
            context_parts.append("\n=== VISUAL CONTENT FROM COURSE MATERIALS ===")
            for i, chunk in enumerate(image_chunks[:3]):
                context_parts.append(f"Visual {i+1}: {chunk.get('content', '')}")
        
        context = "\n".join(context_parts)
        
        # Step 6: Create natural, conversational prompt
        enhanced_prompt = generate_natural_prompt(question, question_type, context)
        
        # Step 7: Generate response
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": enhanced_prompt}],
            temperature=0.1,
            max_tokens=2000
        )
        
        answer = response.choices[0].message.content
        
        # Step 8: Format the response naturally and cleanly
        formatted_answer = format_ai_response(answer, question_type)
        
        print("✅ Enhanced response generated!")
        return formatted_answer
        
    except Exception as e:
        print(f"❌ Enhanced processing failed: {e}")
        import traceback
        traceback.print_exc()
        return "I encountered an error while processing your question. Please try rephrasing your question or check if your course materials are properly uploaded."