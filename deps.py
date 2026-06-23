# deps.py — shared app state, engines, and helper functions.
# Auto-extracted from the original main.py.

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from query_engine import ask_question
import os
import json
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from storage import upload_file
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Query
from supabase import create_client
from datetime import datetime
from dotenv import load_dotenv
from ingest import process_file
from quiz_assistant_engine import assist_with_quiz_question
from notes_engine import generate_notes_from_files, save_notes_to_db, get_notes_from_db, delete_note_from_db
from learning_analytics import LearningAnalyticsEngine
from practice_generator import PracticeGenerator
from typing import Dict, List, Any, Optional
import asyncio
from fastapi import Form, UploadFile, File
from fastapi import HTTPException
from exam_generator import ExamGenerator
from exam_session_manager import ExamSessionManager
from typing import Optional
from ingest import delete_file_from_course, delete_course
from fastapi.responses import Response
from fastapi import Depends
import exports
from auth import get_current_user, current_user_id, require_course_access


# ---- shared state ----
analytics_engine = LearningAnalyticsEngine()
practice_generator = PracticeGenerator()
try:
    from enhanced_ingest import process_file_enhanced, delete_file_from_course as enhanced_delete_file
    from enhanced_query_engine import enhanced_ask_question
    ENHANCED_MODE = True
    print("✅ Enhanced multimodal system loaded!")
except ImportError as e:
    print(f"⚠️ Enhanced system not available: {e}")
    ENHANCED_MODE = False
try:
    from conversational_rag_engine import conversational_ask_question
    CONVERSATIONAL_MODE = True
    print("✅ Conversational RAG system loaded!")
except ImportError as e:
    print(f"⚠️ Conversational RAG not available: {e}")
    CONVERSATIONAL_MODE = False
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
COURSE_DB_PATH = "courses.json"
if not os.path.exists(COURSE_DB_PATH):
    with open(COURSE_DB_PATH, "w") as f:
        json.dump({}, f)
exam_generator = ExamGenerator()
exam_session_manager = ExamSessionManager()


# ---- helpers ----
def load_courses():
    with open(COURSE_DB_PATH) as f:
        return json.load(f)

def save_courses(courses):
    with open(COURSE_DB_PATH, "w") as f:
        json.dump(courses, f, indent=2)

async def validate_course_for_practice(course_id: str) -> dict:
    """Validate that a course exists and has content for practice generation"""
    try:
        # Check if course exists
        course_check = supabase.table("courses").select("*").eq("course_id", course_id).execute()
        if not course_check.data:
            print(f"❌ Course {course_id} not found")
            return {
                "error": "Course not found",
                "topics": ["Course Not Found"],
                "status": "error"
            }
        
        # Check if course has uploaded files
        files_check = supabase.table("files").select("filename").eq("course_id", course_id).execute()
        if not files_check.data:
            print(f"❌ No files found for course {course_id}")
            return {
                "error": "No files uploaded. Please upload course materials first.",
                "topics": ["No Files Uploaded"],
                "status": "error"
            }
        
        # Check if files have been processed (have embeddings)
        embeddings_check = supabase.table("embeddings").select("id").eq("course_id", course_id).limit(1).execute()
        if not embeddings_check.data:
            print(f"⚠️ Course {course_id} files not yet processed for AI analysis")
            return {
                "error": "Course files are still being processed. Please try again in a moment.",
                "topics": ["Processing Files"],
                "status": "processing"
            }
        
        print(f"✅ Course {course_id} validation passed - {len(files_check.data)} files found")
        return {
            "error": None,
            "files_count": len(files_check.data),
            "status": "valid"
        }
        
    except Exception as e:
        print(f"❌ Course validation error: {e}")
        return {
            "error": f"Validation error: {str(e)}",
            "topics": ["Validation Error"],
            "status": "error"
        }

async def get_intelligent_fallback_topics(course_id: str) -> list:
    """Generate intelligent fallback topics based on available course info"""
    try:
        # Try to get course title for subject hints
        course_info = supabase.table("courses").select("title").eq("course_id", course_id).execute()
        course_title = ""
        if course_info.data:
            course_title = course_info.data[0].get("title", "").lower()
        
        # Try to get some filenames for topic hints
        files_info = supabase.table("files").select("filename").eq("course_id", course_id).execute()
        filenames = []
        if files_info.data:
            filenames = [f["filename"].lower() for f in files_info.data]
        
        # Generate subject-appropriate fallback topics
        fallback_topics = generate_subject_fallback_topics(course_title, filenames)
        
        print(f"📋 Generated intelligent fallback topics: {fallback_topics}")
        return fallback_topics
        
    except Exception as e:
        print(f"❌ Fallback topic generation failed: {e}")
        return [
            "Course Fundamentals",
            "Key Concepts",
            "Core Material",
            "Main Topics",
            "Essential Knowledge"
        ]

def generate_subject_fallback_topics(course_title: str, filenames: list) -> list:
    """Generate fallback topics based on course title and filenames - subject aware"""
    
    # Combine course title and filenames for analysis
    text_to_analyze = f"{course_title} {' '.join(filenames)}"
    
    # Subject detection patterns (can be expanded)
    subject_patterns = {
        "computer_science": {
            "keywords": ["programming", "algorithm", "data", "structure", "software", "code", "java", "python", "cs", "computer"],
            "topics": ["Programming Fundamentals", "Algorithm Analysis", "Data Structures", "Software Development", "Problem Solving", "Computational Thinking"]
        },
        "mathematics": {
            "keywords": ["calculus", "algebra", "geometry", "statistics", "math", "equation", "theorem", "proof", "derivative", "integral"],
            "topics": ["Mathematical Concepts", "Problem Solving", "Theoretical Foundations", "Applied Mathematics", "Mathematical Analysis", "Computational Methods"]
        },
        "biology": {
            "keywords": ["biology", "cell", "organism", "genetics", "evolution", "ecology", "physiology", "anatomy", "molecular", "bio"],
            "topics": ["Biological Systems", "Cell Biology", "Genetics and Evolution", "Physiology", "Ecological Concepts", "Molecular Biology"]
        },
        "chemistry": {
            "keywords": ["chemistry", "chemical", "reaction", "molecule", "atom", "organic", "inorganic", "lab", "compound", "chem"],
            "topics": ["Chemical Principles", "Molecular Structure", "Chemical Reactions", "Organic Chemistry", "Inorganic Chemistry", "Laboratory Techniques"]
        },
        "physics": {
            "keywords": ["physics", "mechanics", "thermodynamics", "electromagnetic", "quantum", "force", "energy", "motion", "wave"],
            "topics": ["Mechanics", "Thermodynamics", "Electromagnetism", "Wave Physics", "Modern Physics", "Applied Physics"]
        },
        "history": {
            "keywords": ["history", "historical", "century", "war", "civilization", "culture", "society", "period", "ancient", "modern"],
            "topics": ["Historical Events", "Cultural Analysis", "Historical Periods", "Social Movements", "Historical Methods", "Comparative History"]
        },
        "literature": {
            "keywords": ["literature", "poetry", "novel", "author", "writing", "literary", "text", "analysis", "criticism", "english"],
            "topics": ["Literary Analysis", "Literary Themes", "Writing Techniques", "Literary History", "Critical Reading", "Textual Interpretation"]
        },
        "psychology": {
            "keywords": ["psychology", "behavior", "cognitive", "mental", "brain", "learning", "memory", "perception", "psych"],
            "topics": ["Cognitive Psychology", "Behavioral Psychology", "Research Methods", "Psychological Theories", "Human Development", "Mental Processes"]
        },
        "economics": {
            "keywords": ["economics", "market", "economy", "finance", "business", "trade", "money", "supply", "demand", "econ"],
            "topics": ["Economic Principles", "Market Analysis", "Microeconomics", "Macroeconomics", "Economic Policy", "Financial Systems"]
        },
        "engineering": {
            "keywords": ["engineering", "design", "system", "technical", "mechanical", "electrical", "civil", "project", "analysis"],
            "topics": ["Engineering Design", "System Analysis", "Technical Problem Solving", "Engineering Principles", "Project Management", "Applied Engineering"]
        }
    }
    
    # Detect subject based on keywords
    detected_subject = None
    max_matches = 0
    
    for subject, info in subject_patterns.items():
        matches = sum(1 for keyword in info["keywords"] if keyword in text_to_analyze)
        if matches > max_matches:
            max_matches = matches
            detected_subject = subject
    
    # Return subject-specific topics or generic ones
    if detected_subject and max_matches > 0:
        return subject_patterns[detected_subject]["topics"]
    else:
        # Generic academic topics
        return [
            "Course Fundamentals",
            "Key Concepts and Definitions", 
            "Core Principles",
            "Practical Applications",
            "Theoretical Foundations",
            "Problem-Solving Methods"
        ]

def analyze_course_content_diversity(embeddings_info: list) -> dict:
    """Analyze the diversity and richness of course content"""
    if not embeddings_info:
        return {"status": "no_content"}
    
    # Count documents
    doc_counts = {}
    total_content_length = 0
    page_info = {"has_pages": False, "page_range": []}
    slide_info = {"has_slides": False, "slide_range": []}
    
    for emb in embeddings_info:
        doc_name = emb.get("doc_name", "unknown")
        content = emb.get("content", "")
        page = emb.get("page")
        slide = emb.get("slide")
        
        # Count by document
        doc_counts[doc_name] = doc_counts.get(doc_name, 0) + 1
        total_content_length += len(content)
        
        # Track page info
        if page:
            page_info["has_pages"] = True
            page_info["page_range"].append(page)
        
        # Track slide info
        if slide:
            slide_info["has_slides"] = True
            slide_info["slide_range"].append(slide)
    
    # Calculate content richness
    avg_chunk_length = total_content_length / len(embeddings_info) if embeddings_info else 0
    content_richness = "rich" if avg_chunk_length > 800 else "moderate" if avg_chunk_length > 400 else "sparse"
    
    return {
        "total_chunks": len(embeddings_info),
        "unique_documents": len(doc_counts),
        "document_distribution": doc_counts,
        "average_chunk_length": round(avg_chunk_length),
        "content_richness": content_richness,
        "page_info": {
            "has_pages": page_info["has_pages"],
            "page_range": f"{min(page_info['page_range'])}-{max(page_info['page_range'])}" if page_info["page_range"] else None
        },
        "slide_info": {
            "has_slides": slide_info["has_slides"],
            "slide_range": f"{min(slide_info['slide_range'])}-{max(slide_info['slide_range'])}" if slide_info["slide_range"] else None
        }
    }

def extract_topic_from_filename_debug(filename: str) -> str:
    """Debug version of filename topic extraction with detailed logging"""
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

def download_file(bucket_name: str, file_path: str) -> bytes:
    """Download file from Supabase storage"""
    try:
        result = supabase.storage.from_(bucket_name).download(file_path)
        return result
    except Exception as e:
        print(f"Download failed: {e}")
        raise HTTPException(404, detail=f"File not found: {file_path}")

def calculate_exam_analytics(exam_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate comprehensive exam analytics"""
    if not exam_history:
        return {
            "average_score": 0,
            "total_exams": 0,
            "improvement_trend": "no_data",
            "strong_topics": [],
            "weak_topics": [],
            "time_efficiency": 0,
            "grade_distribution": {}
        }
    
    completed_exams = [exam for exam in exam_history if exam["status"] == "completed"]
    
    if not completed_exams:
        return {
            "average_score": 0,
            "total_exams": len(exam_history),
            "improvement_trend": "no_completed_exams",
            "strong_topics": [],
            "weak_topics": [],
            "time_efficiency": 0,
            "grade_distribution": {}
        }
    
    # Calculate average score
    scores = [exam["final_score"]["percentage"] for exam in completed_exams if exam.get("final_score")]
    average_score = sum(scores) / len(scores) if scores else 0
    
    # Calculate improvement trend
    improvement_trend = "stable"
    if len(scores) >= 3:
        recent_avg = sum(scores[-3:]) / 3
        earlier_avg = sum(scores[:-3]) / (len(scores) - 3) if len(scores) > 3 else scores[0]
        if recent_avg > earlier_avg + 5:
            improvement_trend = "improving"
        elif recent_avg < earlier_avg - 5:
            improvement_trend = "declining"
    
    # Topic performance analysis
    topic_stats = {}
    for exam in completed_exams:
        final_score = exam.get("final_score", {})
        topic_performance = final_score.get("topic_performance", {})
        
        for topic, performance in topic_performance.items():
            if topic not in topic_stats:
                topic_stats[topic] = {"correct": 0, "total": 0}
            
            topic_stats[topic]["correct"] += performance["correct"]
            topic_stats[topic]["total"] += performance["total"]
    
    # Identify strong and weak topics
    strong_topics = []
    weak_topics = []
    
    for topic, stats in topic_stats.items():
        if stats["total"] >= 3:  # Only consider topics with sufficient data
            accuracy = stats["correct"] / stats["total"]
            if accuracy >= 0.8:
                strong_topics.append({"topic": topic, "accuracy": round(accuracy * 100, 1)})
            elif accuracy <= 0.6:
                weak_topics.append({"topic": topic, "accuracy": round(accuracy * 100, 1)})
    
    # Time efficiency
    time_efficiencies = []
    for exam in completed_exams:
        final_score = exam.get("final_score", {})
        time_metrics = final_score.get("time_metrics", {})
        if time_metrics.get("time_efficiency"):
            time_efficiencies.append(time_metrics["time_efficiency"])
    
    avg_time_efficiency = sum(time_efficiencies) / len(time_efficiencies) if time_efficiencies else 0
    
    # Grade distribution
    grade_distribution = {}
    for exam in completed_exams:
        final_score = exam.get("final_score", {})
        grade = final_score.get("letter_grade", "F")
        grade_distribution[grade] = grade_distribution.get(grade, 0) + 1
    
    return {
        "average_score": round(average_score, 1),
        "total_exams": len(completed_exams),
        "improvement_trend": improvement_trend,
        "strong_topics": sorted(strong_topics, key=lambda x: x["accuracy"], reverse=True)[:5],
        "weak_topics": sorted(weak_topics, key=lambda x: x["accuracy"])[:5],
        "time_efficiency": round(avg_time_efficiency, 1),
        "grade_distribution": grade_distribution,
        "recent_scores": scores[-5:] if len(scores) >= 5 else scores,
        "score_trend": scores
    }
