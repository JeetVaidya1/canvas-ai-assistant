import logging
import os

from fastapi import APIRouter, HTTPException, Depends
from deps import (
    CONVERSATIONAL_MODE,
    ENHANCED_MODE,
    analyze_course_content_diversity,
    extract_topic_from_filename_debug,
    practice_generator,
    supabase,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def require_debug_enabled() -> None:
    """Gate for debug/introspection routes.

    They dump course content and internals with no auth, so they 404 unless
    ENABLE_DEBUG_ENDPOINTS=1 is set (default: off). 404 (not 403) so their
    existence isn't advertised in production.
    """
    if os.getenv("ENABLE_DEBUG_ENDPOINTS", "").strip().lower() not in ("1", "true", "yes"):
        raise HTTPException(status_code=404, detail="Not Found")


@router.get("/system-status")
def get_system_status():
    """Get current system capabilities and status"""
    return {
        "enhanced_mode": ENHANCED_MODE,
        "conversational_mode": CONVERSATIONAL_MODE,
        "capabilities": {
            "multimodal_processing": ENHANCED_MODE,
            "image_extraction": ENHANCED_MODE,
            "enhanced_formatting": ENHANCED_MODE,
            "question_classification": ENHANCED_MODE,
            "quiz_assistance": True,
            "intelligent_parsing": True,
            "confidence_scoring": True,
            "study_recommendations": True,
            "notes_generation": True,
            "comprehensive_notes": True,
            "notes_management": True,
            "learning_analytics": True,  # NEW!
            "practice_mode": True,       # NEW!
            "progress_tracking": True,   # NEW!
            "adaptive_difficulty": True, # NEW!
            "spaced_repetition": True    # NEW!
        },
        "version": "3.0.0" if ENHANCED_MODE else "2.0.0"
    }


@router.get("/admin/usage", dependencies=[Depends(require_debug_enabled)])
def get_usage_stats():
    """AI token usage + estimated USD cost since process start (cost observability).

    Gated behind ENABLE_DEBUG_ENDPOINTS (404 when off), same as the debug routes.
    """
    import usage_tracker

    return usage_tracker.snapshot()


@router.get("/health/rag")
def rag_health():
    try:
        row = supabase.table("embeddings").select("course_id, embedding").limit(1).execute()
        if not row.data:
            return {"ok": False, "reason": "no embeddings yet"}
        course_id = row.data[0]["course_id"]
        e_txt = str(row.data[0]["embedding"])  # vector -> text (PostgREST serializes it)
        res = supabase.rpc("match_embeddings", {
            "query_embedding": e_txt,
            "course_id_param": course_id,
            "match_count": 1
        }).execute()
        return {"ok": True, "rows": len(res.data or []), "course_id": course_id}
    except Exception:
        logger.exception("RAG health check failed")
        return {"ok": False, "error": "rag health check failed"}


@router.get("/debug-course-content/{course_id}", dependencies=[Depends(require_debug_enabled)])
async def debug_course_content(course_id: str):
    """Debug endpoint to see what content is available for any course"""
    try:
        # Get course info
        course_info = supabase.table("courses").select("*").eq("course_id", course_id).execute()
        course_data = course_info.data[0] if course_info.data else None
        
        # Get files info
        files_result = supabase.table("files").select("*").eq("course_id", course_id).execute()
        files_info = files_result.data or []
        
        # Get embeddings info
        embeddings_result = supabase.table("embeddings").select("doc_name, content, page, slide").eq("course_id", course_id).limit(15).execute()
        embeddings_info = embeddings_result.data or []
        
        # Analyze content diversity
        content_analysis = analyze_course_content_diversity(embeddings_info)
        
        # Sample content from different sources
        sample_content = []
        seen_docs = set()
        for emb in embeddings_info:
            doc_name = emb.get("doc_name", "unknown")
            if doc_name not in seen_docs and len(sample_content) < 5:
                content = emb.get("content", "")
                sample_content.append({
                    "doc": doc_name,
                    "page": emb.get("page"),
                    "slide": emb.get("slide"),
                    "content_preview": content[:300] + "..." if len(content) > 300 else content,
                    "content_length": len(content)
                })
                seen_docs.add(doc_name)
        
        return {
            "course_id": course_id,
            "course_info": {
                "title": course_data.get("title") if course_data else "Unknown",
                "created_at": course_data.get("created_at") if course_data else None
            },
            "files_summary": {
                "count": len(files_info),
                "files": [{"name": f["filename"], "type": f.get("file_type"), "uploaded": f.get("uploaded_at")} for f in files_info]
            },
            "content_analysis": content_analysis,
            "sample_content": sample_content,
            "vector_store_status": {
                "populated": len(embeddings_info) > 0,
                "total_chunks": len(embeddings_info),
                "unique_documents": len(seen_docs)
            }
        }
        
    except Exception as e:
        return {"error": str(e), "course_id": course_id}


@router.get("/course-subject-detection/{course_id}", dependencies=[Depends(require_debug_enabled)])
async def detect_course_subject(course_id: str):
    """Detect what subject area a course covers - useful for UI and analytics"""
    try:
        # Get course info
        course_info = supabase.table("courses").select("title").eq("course_id", course_id).execute()
        course_title = course_info.data[0].get("title", "") if course_info.data else ""
        
        # Get filenames
        files_info = supabase.table("files").select("filename").eq("course_id", course_id).execute()
        filenames = [f["filename"] for f in files_info.data] if files_info.data else []
        
        # Get sample content
        embeddings_sample = supabase.table("embeddings").select("content").eq("course_id", course_id).limit(5).execute()
        sample_content = " ".join([e["content"][:200] for e in embeddings_sample.data]) if embeddings_sample.data else ""
        
        # Analyze subject
        combined_text = f"{course_title} {' '.join(filenames)} {sample_content}".lower()
        
        # Subject detection logic
        subject_scores = {}
        subject_patterns = {
            "Computer Science": ["programming", "algorithm", "data", "structure", "software", "code", "java", "python", "cs"],
            "Mathematics": ["calculus", "algebra", "geometry", "statistics", "math", "equation", "theorem", "proof"],
            "Biology": ["biology", "cell", "organism", "genetics", "evolution", "ecology", "physiology", "bio"],
            "Chemistry": ["chemistry", "chemical", "reaction", "molecule", "atom", "organic", "inorganic", "chem"],
            "Physics": ["physics", "mechanics", "thermodynamics", "electromagnetic", "quantum", "force", "energy"],
            "History": ["history", "historical", "century", "war", "civilization", "culture", "society", "period"],
            "Literature": ["literature", "poetry", "novel", "author", "writing", "literary", "text", "analysis"],
            "Psychology": ["psychology", "behavior", "cognitive", "mental", "brain", "learning", "memory", "psych"],
            "Economics": ["economics", "market", "economy", "finance", "business", "trade", "money", "econ"],
            "Engineering": ["engineering", "design", "system", "technical", "mechanical", "electrical", "civil"]
        }
        
        for subject, keywords in subject_patterns.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            if score > 0:
                subject_scores[subject] = score
        
        # Find best match
        if subject_scores:
            detected_subject = max(subject_scores, key=subject_scores.get)
            confidence = subject_scores[detected_subject]
        else:
            detected_subject = "General Studies"
            confidence = 0
        
        return {
            "course_id": course_id,
            "detected_subject": detected_subject,
            "confidence_score": confidence,
            "all_scores": subject_scores,
            "course_title": course_title,
            "files_analyzed": len(filenames)
        }
        
    except Exception as e:
        return {
            "course_id": course_id,
            "detected_subject": "Unknown",
            "error": str(e)
        }


@router.get("/debug-topic-extraction/{course_id}", dependencies=[Depends(require_debug_enabled)])
async def debug_topic_extraction(course_id: str):
    """Debug endpoint to see exactly what's happening in topic extraction"""
    try:
        print(f"🔍 DEBUGGING topic extraction for course: {course_id}")
        
        # Step 1: Check files in database
        files_result = supabase.table("files").select("filename").eq("course_id", course_id).execute()
        filenames = [f["filename"] for f in files_result.data] if files_result.data else []
        print(f"📁 Files in database: {filenames}")
        
        # Step 2: Test filename extraction manually
        filename_topics = []
        for filename in filenames:
            extracted_topic = extract_topic_from_filename_debug(filename)
            filename_topics.append({
                "original_filename": filename,
                "extracted_topic": extracted_topic
            })
        
        print(f"📝 Filename topic extraction results: {filename_topics}")
        
        # Step 3: Check vector store content
        embeddings_result = supabase.table("embeddings").select("doc_name, content").eq("course_id", course_id).limit(10).execute()
        embeddings_info = embeddings_result.data or []
        
        content_sample = []
        for emb in embeddings_info[:3]:
            content_sample.append({
                "doc_name": emb.get("doc_name"),
                "content_preview": emb.get("content", "")[:200] + "..."
            })
        
        # Step 4: Try the practice generator methods individually
        debug_results = {
            "course_id": course_id,
            "files_found": len(filenames),
            "filenames": filenames,
            "filename_extraction_results": filename_topics,
            "embeddings_found": len(embeddings_info),
            "content_sample": content_sample,
        }
        
        # Step 5: Test each extraction method
        try:
            # Test filename extraction
            clean_filename_topics = [item["extracted_topic"] for item in filename_topics if item["extracted_topic"]]
            debug_results["clean_filename_topics"] = clean_filename_topics
            
            # Test practice generator filename method
            pg_filename_topics = practice_generator.extract_topics_from_filenames(course_id)
            debug_results["practice_generator_filename_topics"] = pg_filename_topics
            
            # Test content extraction if we have embeddings
            if embeddings_info:
                from vector_store import VectorStore
                vector_store = VectorStore()
                pg_content_topics = practice_generator.extract_topics_from_content(course_id, vector_store)
                debug_results["practice_generator_content_topics"] = pg_content_topics
            
            # Test full extraction
            full_topics = practice_generator.extract_topics_from_course(course_id)
            debug_results["full_extraction_result"] = full_topics
            
        except Exception as e:
            debug_results["extraction_error"] = str(e)
            import traceback
            debug_results["extraction_traceback"] = traceback.format_exc()
        
        return debug_results
        
    except Exception as e:
        return {
            "error": str(e),
            "course_id": course_id
        }


@router.get("/debug-practice-content/{course_id}/{topic}", dependencies=[Depends(require_debug_enabled)])
async def debug_practice_content(course_id: str, topic: str):
    """Debug what content is found when generating practice questions for a topic"""
    try:
        from vector_store import VectorStore
        from providers import make_client
        
        vector_store = VectorStore()
        openai_client = make_client()
        
        print(f"🔍 Debugging content retrieval for topic: '{topic}' in course: {course_id}")
        
        # Test the search queries that practice generator uses
        search_queries = [
            topic,
            f"{topic} examples",
            f"{topic} concepts", 
            f"{topic} definition"
        ]
        
        debug_results = {
            "course_id": course_id,
            "topic": topic,
            "search_results": {},
            "combined_results": [],
            "context_preview": "",
            "total_chunks_found": 0
        }
        
        all_results = []
        
        for query in search_queries:
            try:
                print(f"  🔍 Searching for: '{query}'")
                
                # Create embedding
                emb_resp = openai_client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=[query]
                )
                
                # Search vector store
                results = vector_store.query(course_id, emb_resp.data[0].embedding, top_k=5)
                
                search_info = {
                    "query": query,
                    "results_count": len(results) if results else 0,
                    "results": []
                }
                
                if results:
                    for i, result in enumerate(results):
                        search_info["results"].append({
                            "doc_name": result.get("doc_name", "unknown"),
                            "page": result.get("page"),
                            "similarity": result.get("similarity", 0),
                            "content_preview": result.get("content", "")[:200] + "..." if result.get("content") else "",
                            "content_length": len(result.get("content", ""))
                        })
                        all_results.append(result)
                    
                    print(f"    ✅ Found {len(results)} results")
                else:
                    print(f"    ❌ No results found")
                
                debug_results["search_results"][query] = search_info
                
            except Exception as e:
                print(f"    ❌ Search failed: {e}")
                debug_results["search_results"][query] = {
                    "query": query,
                    "error": str(e),
                    "results_count": 0
                }
        
        # Deduplicate results (same logic as practice generator)
        seen_content = set()
        unique_results = []
        
        for result in all_results:
            content = result.get("content", "").strip()
            content_hash = hash(content[:200])
            
            if content and content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)
                if len(unique_results) >= 8:
                    break
        
        debug_results["combined_results"] = [
            {
                "doc_name": r.get("doc_name"),
                "page": r.get("page"),
                "content_preview": r.get("content", "")[:300] + "..." if r.get("content") else "",
                "content_length": len(r.get("content", ""))
            }
            for r in unique_results
        ]
        
        debug_results["total_chunks_found"] = len(unique_results)
        
        # Create context preview (same as practice generator)
        if unique_results:
            context_parts = [f"COURSE MATERIALS ABOUT {topic.upper()}:"]
            
            for i, result in enumerate(unique_results[:6], 1):
                content = result.get("content", "").strip()
                doc = result.get("doc_name", "unknown")
                page = result.get("page")
                
                source_info = f"[Source {i}: {doc}"
                if page:
                    source_info += f", page {page}"
                source_info += "]"
                
                context_parts.append(f"\n{source_info}")
                context_parts.append(content[:500] + "..." if len(content) > 500 else content)
                context_parts.append("---")
            
            debug_results["context_preview"] = "\n".join(context_parts)[:2000] + "..." if len("\n".join(context_parts)) > 2000 else "\n".join(context_parts)
        else:
            debug_results["context_preview"] = "No relevant content found"
        
        # Check if we have any files that should contain this topic
        files_result = supabase.table("files").select("filename").eq("course_id", course_id).execute()
        relevant_files = []
        
        if files_result.data:
            topic_lower = topic.lower()
            for file_info in files_result.data:
                filename = file_info["filename"].lower()
                if any(word in filename for word in topic_lower.split()):
                    relevant_files.append(file_info["filename"])
        
        debug_results["relevant_files"] = relevant_files
        debug_results["should_have_content"] = len(relevant_files) > 0
        
        return debug_results
        
    except Exception as e:
        return {
            "error": str(e),
            "course_id": course_id,
            "topic": topic
        }


@router.get("/debug-vector-content/{course_id}", dependencies=[Depends(require_debug_enabled)])
async def debug_vector_content(course_id: str, limit: int = 20):
    """See what content is actually in the vector store for a course"""
    try:
        # Get sample of embeddings
        embeddings_result = supabase.table("embeddings").select("doc_name, content, page, slide").eq("course_id", course_id).limit(limit).execute()
        
        if not embeddings_result.data:
            return {
                "error": "No embeddings found for this course",
                "course_id": course_id
            }
        
        # Group by document
        by_document = {}
        for emb in embeddings_result.data:
            doc_name = emb.get("doc_name", "unknown")
            if doc_name not in by_document:
                by_document[doc_name] = []
            
            by_document[doc_name].append({
                "page": emb.get("page"),
                "slide": emb.get("slide"),
                "content_preview": emb.get("content", "")[:200] + "..." if emb.get("content") else "",
                "content_length": len(emb.get("content", ""))
            })
        
        return {
            "course_id": course_id,
            "total_chunks_sampled": len(embeddings_result.data),
            "documents_found": list(by_document.keys()),
            "content_by_document": by_document
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "course_id": course_id
        }

