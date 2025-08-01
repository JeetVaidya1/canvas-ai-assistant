# enhanced_query_engine.py - Phase 1: Multimodal awareness

import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from vector_store import VectorStore
from response_formatter import format_response_basic

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

def enhanced_ask_question(question: str, course_id: str) -> str:
    """
    Enhanced question answering with multimodal awareness and better formatting
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
        
        # Step 6: Create enhanced prompt based on question type
        if question_type == 'definition':
            instruction = """Provide a clear, comprehensive definition. Include:
            1. Simple explanation in plain language
            2. Technical definition if applicable  
            3. Key characteristics or components
            4. Why this concept matters"""
        elif question_type == 'explanation':
            instruction = """Explain the concept thoroughly with:
            1. Step-by-step breakdown
            2. The underlying principles
            3. Real-world analogies when helpful
            4. Reference any diagrams or visuals if available"""
        elif question_type == 'example':
            instruction = """Provide concrete examples:
            1. 2-3 clear, relevant examples
            2. Explain why each example works
            3. Show different contexts or applications
            4. Use any visual content to illustrate examples"""
        elif question_type == 'comparison':
            instruction = """Compare the concepts by:
            1. Identifying key similarities and differences
            2. Using clear comparison structure
            3. Explaining when to use each approach
            4. Reference any comparative diagrams if available"""
        else:
            instruction = """Provide a comprehensive response that:
            1. Addresses the question directly
            2. Uses both text and visual information when available
            3. Provides clear explanations and examples
            4. Connects concepts for better understanding"""

        enhanced_prompt = f"""You are an expert tutor helping a student learn. Your goal is to provide the most helpful, accurate, and educational response possible.

QUESTION TYPE: {question_type}
STUDENT QUESTION: {question}

INSTRUCTIONS: {instruction}

COURSE MATERIALS:
{context}

RESPONSE GUIDELINES:
- Write clearly and conversationally, as if explaining to a student
- Use the course materials as your primary source of truth
- **When you have visual content, reference it explicitly** (e.g., "As shown in the diagram...")
- Include specific examples and analogies when helpful
- Structure your response with clear sections
- If materials don't fully answer the question, acknowledge this
- End with a brief summary
- Suggest follow-up questions when relevant

Provide your response:"""
        
        # Step 7: Generate response
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": enhanced_prompt}],
            temperature=0.1,
            max_tokens=2000
        )
        
        answer = response.choices[0].message.content
        
        # Step 8: Format the response for better readability
        formatted_answer = format_response_basic(answer, question_type)
        
        print("✅ Enhanced response generated!")
        return formatted_answer
        
    except Exception as e:
        print(f"❌ Enhanced processing failed: {e}")
        import traceback
        traceback.print_exc()
        return "I encountered an error while processing your question. Please try rephrasing your question or check if your course materials are properly uploaded."