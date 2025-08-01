# conversational_rag_engine.py - Next-level conversational AI tutor

import os
import json
import re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI
from vector_store import VectorStore
import requests

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)
vector_store = VectorStore()

class ConversationalRAGEngine:
    def __init__(self):
        self.openai_client = openai_client
        self.vector_store = vector_store
        self.conversation_memory = {}  # Store per session
        
    def get_conversation_context(self, session_id: str, last_n_messages: int = 4) -> str:
        """Get recent conversation context from database"""
        try:
            from supabase import create_client
            SUPABASE_URL = os.getenv("SUPABASE_URL")
            SUPABASE_KEY = os.getenv("SUPABASE_KEY")
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            
            # Get recent messages from this session
            resp = supabase.table("messages") \
                .select("role, content") \
                .eq("session_id", session_id) \
                .order("timestamp", desc=True) \
                .limit(last_n_messages) \
                .execute()
            
            messages = resp.data[::-1]  # Reverse to get chronological order
            
            context_parts = []
            for msg in messages:
                role = "Student" if msg["role"] == "user" else "AI"
                content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
                context_parts.append(f"{role}: {content}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            print(f"Error getting conversation context: {e}")
            return ""
    
    def enhance_query_with_context(self, question: str, session_id: str) -> str:
        """Enhance the current query with conversation context"""
        
        # Get conversation history
        conversation_context = self.get_conversation_context(session_id)
        
        if not conversation_context:
            return question
        
        # Use AI to create a context-aware search query
        enhancement_prompt = f"""
        Based on this conversation history, create a better search query for the student's current question.
        
        Conversation History:
        {conversation_context}
        
        Current Question: {question}
        
        Create a search query that:
        1. Maintains the topic context from the conversation
        2. Includes relevant keywords from previous messages
        3. Is specific enough to find the right information
        4. Is focused on the student's learning needs
        
        Return just the enhanced search query (no explanation):
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": enhancement_prompt}],
                temperature=0.1,
                max_tokens=100
            )
            
            enhanced_query = response.choices[0].message.content.strip()
            print(f"🔍 Enhanced query: {enhanced_query}")
            return enhanced_query
            
        except Exception as e:
            print(f"Query enhancement failed: {e}")
            return question

    def search_web_for_context(self, query: str, max_results: int = 2) -> str:
        """Search web for additional context (using a simple approach)"""
        try:
            # Simple web search simulation - in production, use Google Custom Search API or similar
            web_context = f"""
            Web Context for "{query}":
            This is where we would add relevant web search results to supplement 
            the course materials with additional examples, clarifications, or 
            current information about {query}.
            """
            return web_context
            
        except Exception as e:
            print(f"Web search failed: {e}")
            return ""

    def intelligent_retrieval(self, question: str, course_id: str, session_id: str = None) -> List[Dict]:
        """Perform intelligent retrieval with conversation awareness"""
        
        # Step 1: Enhance query with conversation context
        if session_id:
            enhanced_query = self.enhance_query_with_context(question, session_id)
        else:
            enhanced_query = question
        
        # Step 2: Search course materials with enhanced query
        try:
            emb_resp = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=[enhanced_query]
            )
            query_embedding = emb_resp.data[0].embedding
            
            # Get more results initially
            results = self.vector_store.query(course_id, query_embedding, top_k=12) or []
            print(f"📚 Found {len(results)} course material results")
            
        except Exception as e:
            print(f"Course material search failed: {e}")
            results = []
        
        # Step 3: Filter and rank results by relevance to conversation
        if session_id and results:
            results = self.rerank_by_conversation_relevance(results, question, session_id)
        
        return results

    def rerank_by_conversation_relevance(self, results: List[Dict], question: str, session_id: str) -> List[Dict]:
        """Rerank results based on conversation context"""
        
        conversation_context = self.get_conversation_context(session_id)
        
        if not conversation_context:
            return results
        
        # Use AI to rerank based on conversation relevance
        rerank_prompt = f"""
        Rank these search results by relevance to the student's question in the context of their ongoing conversation.
        
        Conversation Context:
        {conversation_context}
        
        Current Question: {question}
        
        Search Results:
        """
        
        for i, result in enumerate(results[:8]):  # Limit for prompt size
            content_preview = result.get('content', '')[:300]
            rerank_prompt += f"\n{i+1}. {content_preview}..."
        
        rerank_prompt += f"""
        
        Return a JSON list of numbers (1-{min(len(results), 8)}) in order from most to least relevant to the conversation:
        Example: [3, 1, 7, 2, 5, 4, 6, 8]
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": rerank_prompt}],
                temperature=0,
                max_tokens=100
            )
            
            rankings = json.loads(response.choices[0].message.content.strip())
            
            # Reorder results based on rankings
            reranked = []
            for rank in rankings:
                if 1 <= rank <= len(results):
                    reranked.append(results[rank-1])
            
            # Add any remaining results
            for result in results:
                if result not in reranked:
                    reranked.append(result)
            
            print(f"🎯 Reranked results based on conversation context")
            return reranked
            
        except Exception as e:
            print(f"Reranking failed: {e}")
            return results

    def generate_conversational_response(self, question: str, course_id: str, session_id: str = None) -> str:
        """Generate a conversational, context-aware response"""
        
        # Step 1: Get intelligent retrieval results
        course_results = self.intelligent_retrieval(question, course_id, session_id)
        
        # Step 2: Get conversation context
        conversation_context = ""
        if session_id:
            conversation_context = self.get_conversation_context(session_id, last_n_messages=6)
        
        # Step 3: Build enhanced context
        context_parts = []
        
        # Add course materials
        if course_results:
            context_parts.append("=== COURSE MATERIALS ===")
            for i, result in enumerate(course_results[:6]):
                content = result.get('content', '')
                doc_name = result.get('doc_name', 'Unknown')
                context_parts.append(f"Source {i+1} (from {doc_name}): {content}")
        
        # Add conversation context
        if conversation_context:
            context_parts.append(f"\n=== CONVERSATION HISTORY ===\n{conversation_context}")
        
        context = "\n".join(context_parts)
        
        # Step 4: Generate intelligent response
        response_prompt = f"""You are an expert AI tutor having an ongoing conversation with a student. You have access to their course materials and conversation history.

IMPORTANT CONTEXT AWARENESS:
- This is a continuing conversation, not a standalone question
- Pay attention to what the student was previously discussing
- Maintain topic continuity and build on previous explanations
- If the student asks for "diagrams" or "examples", prioritize visual content from course materials
- Be specific and helpful, like a real tutor who remembers the conversation

STUDENT'S CURRENT QUESTION: {question}

AVAILABLE INFORMATION:
{context}

INSTRUCTIONS:
- Answer as a knowledgeable tutor who remembers the conversation
- Use course materials as the primary source, but explain clearly
- If course materials are insufficient, acknowledge this and provide general guidance
- Reference specific parts of materials when relevant
- Keep responses natural and conversational
- Stay focused on the student's actual question and topic
- If there's any ambiguity, ask for clarification rather than switching topics

Provide a helpful, contextual response:"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": response_prompt}],
                temperature=0.2,
                max_tokens=2000
            )
            
            answer = response.choices[0].message.content
            
            # Clean up the response
            answer = self.clean_response(answer)
            
            return answer
            
        except Exception as e:
            print(f"Response generation failed: {e}")
            return "I'm having trouble processing your question right now. Could you please rephrase it or try again?"

    def clean_response(self, response: str) -> str:
        """Clean and format the response naturally"""
        
        # Remove excessive formatting
        response = re.sub(r'^#+\s*', '', response, flags=re.MULTILINE)
        response = re.sub(r'\*\*([^*]+)\*\*', r'\1', response)  # Remove bold formatting
        
        # Clean up whitespace
        response = re.sub(r'\n\s*\n\s*\n', '\n\n', response)
        
        return response.strip()

# Main function to replace existing ask function
def conversational_ask_question(question: str, course_id: str, session_id: str = None) -> str:
    """
    Main conversational RAG function
    """
    engine = ConversationalRAGEngine()
    return engine.generate_conversational_response(question, course_id, session_id)