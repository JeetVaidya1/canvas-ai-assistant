from fastapi import APIRouter, Form, HTTPException, Depends
import json
from datetime import datetime

from auth import current_user_id
from deps import (
    CONVERSATIONAL_MODE,
    ENHANCED_MODE,
    conversational_ask_question,
    enhanced_ask_question,
    supabase,
)
from query_engine import ask_question
from rate_limit import ai_rate_limit

import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/ask", dependencies=[Depends(ai_rate_limit)])
async def ask_endpoint(
    question: str = Form(...),
    course_id: str = Form(...),
    session_id: str | None = Form(None),
    user_id: str = Depends(current_user_id)
):
    # 1) Create a new chat_session if none was provided
    if not session_id:
        try:
            resp = supabase.table("chat_sessions").insert({
                "user_id": user_id,
                "course_id": course_id,
                "title": question[:50],
                "created_at": datetime.utcnow().isoformat()
            }).execute()
            session_id = resp.data[0]["id"]
        except Exception as e:
            logger.exception("Couldn't create session")
            raise HTTPException(500, detail="Couldn't create session")

    # 2) Record the user's question
    try:
        supabase.table("messages").insert({
            "session_id": session_id,
            "role": "user",
            "content": question,
            "timestamp": datetime.utcnow().isoformat()
        }).execute()
    except Exception as e:
        logger.exception("Couldn't save question")
        raise HTTPException(500, detail="Couldn't save question")

    # 3) **NEW: Generate conversational answer with context awareness**
    try:
        if CONVERSATIONAL_MODE:
            logger.info("Using conversational RAG with context awareness")
            answer = conversational_ask_question(question, course_id, session_id)
            logger.info("Conversational answer generated")
        elif ENHANCED_MODE:
            logger.info("Using enhanced question answering")
            answer = enhanced_ask_question(question, course_id)
        else:
            logger.info("Using basic question answering")
            answer = ask_question(question, course_id)
    except Exception as e:
        logger.exception("All QA methods failed; using fallback answer")
        answer = "I'm having trouble processing your question. Could you please rephrase it or try asking in a different way?"

    # 4) Record the assistant's answer
    try:
        supabase.table("messages").insert({
            "session_id": session_id,
            "role": "assistant",
            "content": answer,
            "timestamp": datetime.utcnow().isoformat()
        }).execute()
    except Exception as e:
        logger.exception("Couldn't save answer")
        raise HTTPException(500, detail="Couldn't save answer")

    return {
        "session_id": session_id,
        "question": question,
        "answer": answer
    }


@router.post("/ask/stream", dependencies=[Depends(ai_rate_limit)])
async def ask_stream_endpoint(
    question: str = Form(...),
    course_id: str = Form(...),
    session_id: str | None = Form(None),
    user_id: str = Depends(current_user_id),
):
    """Streaming chat: emits Server-Sent Events with answer text deltas."""
    from fastapi.responses import StreamingResponse
    from conversational_rag_engine import conversational_ask_stream

    # 1) Ensure a chat session exists, then record the question (same as /ask).
    if not session_id:
        try:
            resp = supabase.table("chat_sessions").insert({
                "user_id": user_id,
                "course_id": course_id,
                "title": question[:50],
                "created_at": datetime.utcnow().isoformat(),
            }).execute()
            session_id = resp.data[0]["id"]
        except Exception as e:
            logger.exception("Couldn't create session")
            raise HTTPException(500, detail="Couldn't create session")
    try:
        supabase.table("messages").insert({
            "session_id": session_id,
            "role": "user",
            "content": question,
            "timestamp": datetime.utcnow().isoformat(),
        }).execute()
    except Exception as e:
        logger.exception("Couldn't save question")
        raise HTTPException(500, detail="Couldn't save question")

    def event_stream():
        # First event carries the session id so the client can track it.
        yield f"data: {json.dumps({'session_id': session_id})}\n\n"
        collected = []
        stream_sources = []
        try:
            for evt in conversational_ask_stream(question, course_id, session_id):
                if "sources" in evt:
                    stream_sources = evt["sources"]
                if "delta" in evt:
                    collected.append(evt["delta"])
                yield f"data: {json.dumps(evt)}\n\n"
        except Exception as e:
            logger.exception("Chat stream failed mid-flight")
            yield f"data: {json.dumps({'delta': ' (stream interrupted)'})}\n\n"
        answer = "".join(collected).strip()
        # Persist the assistant message (with its sources) once the stream completes.
        try:
            supabase.table("messages").insert({
                "session_id": session_id,
                "role": "assistant",
                "content": answer,
                "sources": stream_sources,
                "timestamp": datetime.utcnow().isoformat(),
            }).execute()
        except Exception as e:
            logger.exception("Couldn't save streamed answer for session %s", session_id)
        yield f"data: {json.dumps({'done': True, 'session_id': session_id})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/sessions")
def list_sessions(user_id: str = Depends(current_user_id)):
    """
    List all sessions for a user, newest first.
    """
    try:
        resp = supabase.table("chat_sessions") \
            .select("*") \
            .eq("user_id", user_id) \
            .order("created_at", desc=True) \
            .execute()
        sessions = resp.data
    except Exception as e:
        logger.exception("Couldn't fetch sessions")
        raise HTTPException(500, detail="Couldn't fetch sessions")
    return {"sessions": sessions}


@router.get("/sessions/{session_id}/messages")
def get_messages(session_id: str):
    """
    Fetch all messages in a session, oldest first.
    """
    try:
        resp = supabase.table("messages") \
            .select("*") \
            .eq("session_id", session_id) \
            .order("timestamp", desc=False) \
            .execute()
        messages = resp.data
    except Exception as e:
        logger.exception("Couldn't fetch messages")
        raise HTTPException(500, detail="Couldn't fetch messages")
    return {"messages": messages}


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """
    Delete a chat session and all its messages.
    """
    try:
        # First delete all messages in the session
        supabase.table("messages").delete().eq("session_id", session_id).execute()
        
        # Then delete the session itself
        supabase.table("chat_sessions").delete().eq("id", session_id).execute()
        
        return {"status": "ok", "message": "Session deleted successfully"}
    except Exception as e:
        logger.exception("Couldn't delete session")
        raise HTTPException(500, detail="Couldn't delete session")

