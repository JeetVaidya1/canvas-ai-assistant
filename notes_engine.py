# notes_engine.py - Comprehensive notes generation from lecture materials
import os
import uuid
from typing import Dict, List, Any
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from vector_store import VectorStore

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
vector_store = VectorStore()

def extract_content_from_files(course_id: str, file_names: List[str]) -> Dict[str, str]:
    """Extract content from specific files in the course"""
    try:
        file_contents = {}
        
        for file_name in file_names:
            # Get all chunks for this specific file
            emb_resp = openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=[file_name]  # Search by filename
            )
            
            # Query vector store for this file's content
            results = vector_store.query(course_id, emb_resp.data[0].embedding, top_k=50) or []
            
            # Filter results to only include content from this specific file
            file_chunks = [r for r in results if r.get('doc_name') == file_name]
            
            if file_chunks:
                # Sort chunks by chunk_id to maintain order
                sorted_chunks = sorted(file_chunks, key=lambda x: x.get('chunk_id', 0))
                content = "\n\n".join([chunk.get('content', '') for chunk in sorted_chunks])
                file_contents[file_name] = content
            else:
                file_contents[file_name] = f"Content not found for {file_name}"
        
        return file_contents
    except Exception as e:
        print(f"❌ Content extraction failed: {e}")
        return {}

def generate_detailed_notes(content: Dict[str, str], topic: str = "", style: str = "detailed") -> Dict[str, Any]:
    """Generate comprehensive, student-quality notes"""
    
    # Combine all content
    combined_content = ""
    source_info = ""
    
    for file_name, file_content in content.items():
        combined_content += f"\n\n=== Content from {file_name} ===\n{file_content}"
        source_info += f"- {file_name}\n"
    
    # Create style-specific prompts
    style_instructions = {
        "detailed": """
        Create EXTREMELY DETAILED lecture notes that a diligent student would take. Include:
        - Complete explanations of every concept
        - Step-by-step breakdowns of processes
        - Detailed examples with full workings
        - Important formulas, definitions, and theorems
        - Key insights and connections between concepts
        - Potential exam questions and their answers
        - Warning notes about common mistakes
        - Visual descriptions of any diagrams or charts mentioned
        
        Format as comprehensive study notes with clear headings, subheadings, bullet points, and numbered lists.
        Make these notes so thorough that a student could study from them alone.
        """,
        
        "summary": """
        Create concise but comprehensive summary notes that capture:
        - Main concepts and key points
        - Essential formulas and definitions
        - Important examples
        - Key takeaways and conclusions
        
        Format as clear, organized summary notes with bullet points and headings.
        """,
        
        "outline": """
        Create a detailed outline format with:
        - Hierarchical structure (I, II, III, A, B, C, 1, 2, 3)
        - Main topics and subtopics
        - Key points under each section
        - Important details and examples
        
        Format as a traditional outline structure.
        """
    }
    
    topic_focus = f"\n\nSPECIAL FOCUS: Pay particular attention to content related to '{topic}' and provide extra detail on this topic." if topic else ""
    
    prompt = f"""You are an expert academic note-taker creating comprehensive lecture notes for a student.

SOURCE MATERIALS:
{combined_content}

INSTRUCTIONS:
{style_instructions.get(style, style_instructions['detailed'])}
{topic_focus}

REQUIREMENTS:
- Write in a clear, academic style that students can easily understand
- Include ALL important information, don't summarize too heavily
- Use proper formatting with headers, bullet points, and numbering
- Add explanatory notes in brackets [like this] for complex concepts
- Include memory aids and mnemonics where helpful
- Make connections between different concepts explicit
- Add study tips and exam preparation notes throughout

Generate comprehensive notes that would help a student succeed in their course:"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=4000  # Allow for very detailed notes
        )
        
        notes_content = response.choices[0].message.content
        
        # Generate a title if topic is provided
        if topic:
            title = f"Detailed Notes: {topic}"
        else:
            title_prompt = f"Based on this content, generate a concise, descriptive title for these lecture notes:\n\n{notes_content[:500]}..."
            
            title_response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": title_prompt}],
                temperature=0.2,
                max_tokens=50
            )
            title = title_response.choices[0].message.content.strip()
        
        # Extract topics for tagging
        topics = extract_topics_from_content(notes_content)
        
        # Calculate metrics
        word_count = len(notes_content.split())
        reading_time = f"{max(1, word_count // 200)} min"
        
        return {
            "notes": notes_content,
            "suggested_title": title,
            "topics": topics,
            "word_count": word_count,
            "reading_time": reading_time,
            "source_files": list(content.keys())
        }
        
    except Exception as e:
        print(f"❌ Notes generation failed: {e}")
        return {
            "notes": "Failed to generate notes. Please try again with different source materials.",
            "suggested_title": "Error - Notes Generation Failed",
            "topics": [],
            "word_count": 0,
            "reading_time": "0 min",
            "source_files": list(content.keys())
        }

def extract_topics_from_content(content: str) -> List[str]:
    """Extract key topics from notes content for tagging"""
    try:
        topic_prompt = f"""Analyze this content and extract 5-8 key topics/concepts that are covered. Return only a JSON list of topic strings.

Content: {content[:1000]}...

Return format: ["topic1", "topic2", "topic3", ...]
"""
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": topic_prompt}],
            temperature=0.1,
            max_tokens=200
        )
        
        import json
        topics = json.loads(response.choices[0].message.content.strip())
        return topics[:8]  # Limit to 8 topics
        
    except Exception as e:
        print(f"❌ Topic extraction failed: {e}")
        # Fallback: extract topics using simple heuristics
        words = content.lower().split()
        # Look for common academic terms and capitalize them
        potential_topics = []
        academic_terms = ['algorithm', 'data structure', 'function', 'method', 'process', 'theory', 'concept', 'principle']
        
        for term in academic_terms:
            if term in words:
                potential_topics.append(term.title())
        
        return potential_topics[:5] if potential_topics else ["General Topics"]

def generate_notes_from_files(course_id: str, file_names: List[str], topic: str = "", style: str = "detailed") -> Dict[str, Any]:
    """Main function to generate notes from selected files"""
    try:
        print(f"📝 Generating {style} notes for {len(file_names)} files")
        print(f"📚 Files: {', '.join(file_names)}")
        if topic:
            print(f"🎯 Topic focus: {topic}")
        
        # Extract content from selected files
        content = extract_content_from_files(course_id, file_names)
        
        if not content or all(not v.strip() for v in content.values()):
            return {
                "status": "error",
                "message": "No content found in selected files",
                "notes": "Unable to generate notes: No content could be extracted from the selected files. Please ensure the files contain readable text content.",
                "suggested_title": "Error - No Content Found",
                "topics": [],
                "word_count": 0,
                "reading_time": "0 min",
                "source_files": file_names
            }
        
        # Generate detailed notes
        result = generate_detailed_notes(content, topic, style)
        result["status"] = "success"
        
        print(f"✅ Generated {result['word_count']} word notes")
        return result
        
    except Exception as e:
        print(f"❌ Notes generation error: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "status": "error",
            "message": str(e),
            "notes": "An error occurred while generating notes. Please try again.",
            "suggested_title": "Error - Generation Failed",
            "topics": [],
            "word_count": 0,
            "reading_time": "0 min",
            "source_files": file_names
        }

# Database functions for saving/loading notes
def save_notes_to_db(course_id: str, title: str, content: str, source_files: List[str], 
                    topic: str = "", note_id: str = None) -> Dict[str, Any]:
    """Save notes to database"""
    try:
        from supabase import create_client
        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_KEY")
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Extract topics and calculate metrics
        topics = extract_topics_from_content(content)
        word_count = len(content.split())
        reading_time = f"{max(1, word_count // 200)} min"
        
        note_data = {
            "course_id": course_id,
            "title": title,
            "content": content,
            "source_files": source_files,
            "topic_focus": topic,
            "topics": topics,
            "word_count": word_count,
            "reading_time": reading_time,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        if note_id:
            # Update existing note
            result = supabase.table("notes").update(note_data).eq("id", note_id).execute()
            saved_note = result.data[0] if result.data else None
        else:
            # Create new note
            note_data["id"] = str(uuid.uuid4())
            note_data["created_at"] = datetime.utcnow().isoformat()
            
            result = supabase.table("notes").insert(note_data).execute()
            saved_note = result.data[0] if result.data else None
        
        if saved_note:
            return {
                "status": "success",
                "note": saved_note
            }
        else:
            return {
                "status": "error",
                "message": "Failed to save note to database"
            }
            
    except Exception as e:
        print(f"❌ Note saving error: {e}")
        return {
            "status": "error",
            "message": f"Database error: {str(e)}"
        }

def get_notes_from_db(course_id: str) -> List[Dict[str, Any]]:
    """Get all notes for a course"""
    try:
        from supabase import create_client
        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_KEY")
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        result = supabase.table("notes").select("*").eq("course_id", course_id).order("updated_at", desc=True).execute()
        
        return result.data if result.data else []
        
    except Exception as e:
        print(f"❌ Notes retrieval error: {e}")
        return []

def delete_note_from_db(note_id: str) -> bool:
    """Delete a note from database"""
    try:
        from supabase import create_client
        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_KEY")
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        result = supabase.table("notes").delete().eq("id", note_id).execute()
        
        return len(result.data) > 0 if result.data else True
        
    except Exception as e:
        print(f"❌ Note deletion error: {e}")
        return False