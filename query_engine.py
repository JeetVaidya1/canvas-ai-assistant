# advanced_query_engine.py

import os
import json
import re
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from vector_store import VectorStore
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Load keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Instantiate clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
vector_store = VectorStore()

class AdvancedRAGEngine:
    def __init__(self):
        self.openai_client = openai_client
        self.vector_store = vector_store
        
    def classify_query_type(self, question: str) -> str:
        """Classify the type of student question for better handling"""
        classification_prompt = f"""
        Classify this student question into one of these categories:
        1. DEFINITION - asking what something means
        2. EXPLANATION - asking how something works
        3. EXAMPLE - asking for examples or demonstrations
        4. COMPARISON - comparing concepts or ideas
        5. APPLICATION - asking how to apply knowledge
        6. ANALYSIS - asking to analyze or break down concepts
        7. SYNTHESIS - asking to connect multiple concepts
        8. EVALUATION - asking for judgment or assessment

        Question: "{question}"
        
        Return just the category name.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Faster for classification
                messages=[{"role": "user", "content": classification_prompt}],
                temperature=0
            )
            return response.choices[0].message.content.strip()
        except:
            return "EXPLANATION"  # Default fallback

    def expand_query(self, question: str, query_type: str) -> List[str]:
        """Generate multiple query variations for better retrieval"""
        expansion_prompt = f"""
        You're helping a student research. Generate 3 different ways to search for information about this question.
        Make the searches more specific and academic.
        
        Original question: "{question}"
        Question type: {query_type}
        
        Generate 3 search queries that would help find relevant information:
        1. A direct factual query
        2. A conceptual/theoretical query  
        3. A practical/application query
        
        Return as a JSON list of strings.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": expansion_prompt}],
                temperature=0.3
            )
            
            queries = json.loads(response.choices[0].message.content.strip())
            return [question] + queries  # Include original + expansions
        except:
            return [question]  # Fallback to original

    def hybrid_search(self, queries: List[str], course_id: str, top_k: int = 15) -> List[Dict]:
        """Perform hybrid search using multiple query variations"""
        all_results = []
        seen_content = set()
        
        for query in queries:
            # Embed the query
            emb_resp = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=[query]
            )
            query_embedding = emb_resp.data[0].embedding
            
            # Get results from vector store
            try:
                results = self.vector_store.query(course_id, query_embedding, top_k=5) or []
                
                for result in results:
                    content = result.get("content", "")
                    # Avoid duplicates
                    if content and content not in seen_content:
                        result["query_used"] = query
                        result["relevance_score"] = result.get("similarity", 0.0)
                        all_results.append(result)
                        seen_content.add(content)
                        
            except Exception as e:
                print(f"Search error for query '{query}': {e}")
                continue
        
        return all_results[:top_k]

    def rerank_results(self, question: str, results: List[Dict]) -> List[Dict]:
        """Rerank results based on relevance to the specific question"""
        if len(results) <= 1:
            return results
            
        # Create reranking prompt
        docs_text = ""
        for i, result in enumerate(results):
            docs_text += f"Document {i+1}: {result.get('content', '')[:500]}...\n\n"
        
        rerank_prompt = f"""
        Question: "{question}"
        
        Rank these documents by relevance to answering the student's question.
        Consider:
        - Direct relevance to the question
        - Educational value for a student
        - Clarity and completeness of information
        
        Documents:
        {docs_text}
        
        Return a JSON list of document numbers (1-{len(results)}) in order from most to least relevant.
        Example: [3, 1, 5, 2, 4]
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": rerank_prompt}],
                temperature=0
            )
            
            rankings = json.loads(response.choices[0].message.content.strip())
            
            # Reorder results based on rankings
            reranked = []
            for rank in rankings:
                if 1 <= rank <= len(results):
                    reranked.append(results[rank-1])
            
            # Add any missing results at the end
            for i, result in enumerate(results):
                if result not in reranked:
                    reranked.append(result)
                    
            return reranked
            
        except Exception as e:
            print(f"Reranking error: {e}")
            return results  # Return original order if reranking fails

    def create_enhanced_context(self, question: str, query_type: str, results: List[Dict]) -> str:
        """Create rich context with proper formatting for the LLM"""
        if not results:
            return "No relevant course materials found."
        
        context_parts = []
        
        # Add context header
        context_parts.append(f"=== COURSE MATERIALS FOR: {question} ===\n")
        
        # Group similar content and add structure
        for i, result in enumerate(results[:8], 1):  # Limit to top 8 for context window
            content = result.get("content", "").strip()
            doc_name = result.get("doc_name", "Unknown Document")
            
            context_parts.append(f"📄 Source {i}: {doc_name}")
            context_parts.append(f"Content: {content}")
            context_parts.append("---")
        
        return "\n".join(context_parts)

    def generate_student_optimized_prompt(self, question: str, query_type: str, context: str) -> str:
        """Create a specialized prompt optimized for student learning"""
        
        # Different prompt strategies based on question type
        if query_type == "DEFINITION":
            instruction = """Provide a clear, comprehensive definition. Include:
            1. Simple explanation in plain language
            2. Technical definition if applicable
            3. Key characteristics or components
            4. Why this concept matters in the field"""
            
        elif query_type == "EXPLANATION":
            instruction = """Explain the concept thoroughly with:
            1. Step-by-step breakdown of how it works
            2. The underlying principles or mechanisms
            3. Real-world analogies to aid understanding
            4. Common misconceptions to avoid"""
            
        elif query_type == "EXAMPLE":
            instruction = """Provide concrete examples:
            1. 2-3 clear, relevant examples
            2. Explain why each example demonstrates the concept
            3. Show different contexts or applications
            4. Connect examples back to the theory"""
            
        elif query_type == "COMPARISON":
            instruction = """Compare the concepts by:
            1. Identifying key similarities and differences
            2. Using a clear comparison structure (pros/cons, table format)
            3. Explaining when to use each approach
            4. Providing examples of each"""
            
        elif query_type == "APPLICATION":
            instruction = """Show practical application:
            1. Step-by-step process for applying the concept
            2. Real-world scenarios where this applies
            3. Tips for successful implementation
            4. Common pitfalls and how to avoid them"""
            
        else:  # ANALYSIS, SYNTHESIS, EVALUATION, or default
            instruction = """Provide a comprehensive analysis:
            1. Break down the key components or aspects
            2. Explain relationships between different parts
            3. Discuss implications and significance
            4. Connect to broader concepts or themes"""

        return f"""You are an expert tutor helping a student learn. Your goal is to provide the most helpful, accurate, and educational response possible.

QUESTION TYPE: {query_type}
STUDENT QUESTION: {question}

INSTRUCTIONS: {instruction}

COURSE MATERIALS:
{context}

RESPONSE GUIDELINES:
- Write clearly and conversationally, as if explaining to a student
- Use the course materials as your primary source of truth
- If the materials don't fully answer the question, acknowledge this
- Include specific examples and analogies when helpful
- Structure your response with clear sections or bullet points
- End with a brief summary or key takeaway
- If relevant, suggest follow-up questions for deeper learning

Provide your response:"""

    async def ask_question(self, question: str, course_id: str) -> str:
        """Main entry point for advanced RAG question answering"""
        try:
            # Step 1: Classify the question type
            query_type = self.classify_query_type(question)
            
            # Step 2: Expand the query for better retrieval
            expanded_queries = self.expand_query(question, query_type)
            
            # Step 3: Perform hybrid search
            search_results = self.hybrid_search(expanded_queries, course_id, top_k=12)
            
            if not search_results:
                return "I don't have enough information in your course materials to answer this question. Please make sure you've uploaded relevant files for this topic."
            
            # Step 4: Rerank results for relevance
            reranked_results = self.rerank_results(question, search_results)
            
            # Step 5: Create enhanced context
            context = self.create_enhanced_context(question, query_type, reranked_results)
            
            # Step 6: Generate optimized prompt
            prompt = self.generate_student_optimized_prompt(question, query_type, context)
            
            # Step 7: Generate final response with GPT-4
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  # Use best model for generation
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for factual accuracy
                max_tokens=2000   # Allow for comprehensive responses
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error in advanced RAG: {e}")
            return "I encountered an error while processing your question. Please try rephrasing your question or check if your course materials are properly uploaded."

# Create global instance
advanced_rag_engine = AdvancedRAGEngine()

# Updated function for backward compatibility
def ask_question(question: str, course_id: str) -> str:
    """Enhanced ask_question function using advanced RAG"""
    return asyncio.run(advanced_rag_engine.ask_question(question, course_id))