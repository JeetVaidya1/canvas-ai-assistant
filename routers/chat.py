from fastapi import APIRouter, BackgroundTasks, Form, HTTPException, Depends
import json
from datetime import datetime

from auth import current_user_id, get_current_user, require_course_access
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


def _require_owned_session(session_id: str, user_id: str) -> dict:
    """Fetch a chat session and enforce that the token user owns it.

    Non-enumeration choice: a session owned by someone else returns the SAME
    404 as a session that doesn't exist, so an attacker probing session ids
    can't even learn that a foreign session exists.
    """
    try:
        rows = (supabase.table("chat_sessions")
                .select("*")
                .eq("id", session_id)
                .limit(1)
                .execute()
                .data)
    except Exception:
        logger.exception("Couldn't fetch session %s", session_id)
        raise HTTPException(500, detail="Couldn't fetch session")
    if not rows or rows[0].get("user_id") != user_id:
        raise HTTPException(404, detail="Session not found")
    return rows[0]


def _track_chat_interaction(user_id: str, course_id: str, question: str, answer: str) -> None:
    """Fire-and-forget analytics write for a successful chat answer.

    Runs on BackgroundTasks AFTER the response is sent; feeds user_interactions
    + learning_progress so chat activity counts toward streak/questions/topics.
    Every failure is logged and swallowed — tracking can never affect a
    response the user already received.
    """
    try:
        from deps import analytics_engine
        analytics_engine.track_interaction(
            user_id=user_id,
            course_id=course_id,
            question=question,
            answer=(answer or "")[:500],
            confidence=0.5,  # neutral: chat has no right/wrong signal
            response_time=0,
            question_type="chat",
        )
    except Exception:  # noqa: BLE001
        logger.warning("Chat interaction tracking failed (user=%s course=%s)",
                       user_id, course_id, exc_info=True)


@router.post("/ask", dependencies=[Depends(ai_rate_limit)])
async def ask_endpoint(
    background_tasks: BackgroundTasks,
    question: str = Form(...),
    course_id: str = Form(...),
    session_id: str | None = Form(None),
    user=Depends(get_current_user)
):
    user_id = user["id"]
    # 0) The token user must own or be a member of the course being queried.
    require_course_access(course_id, user)

    # 1) Create a new chat_session if none was provided; if one WAS provided
    # it must belong to the token user (404 otherwise, non-enumeration).
    if session_id:
        _require_owned_session(session_id, user_id)
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

    # 5) Count this exchange toward mastery/streak analytics AFTER the response
    # is sent (fire-and-forget; failures are logged inside the helper).
    background_tasks.add_task(_track_chat_interaction, user_id, course_id, question, answer)

    return {
        "session_id": session_id,
        "question": question,
        "answer": answer
    }


@router.post("/ask/stream", dependencies=[Depends(ai_rate_limit)])
async def ask_stream_endpoint(
    background_tasks: BackgroundTasks,
    question: str = Form(...),
    course_id: str = Form(...),
    session_id: str | None = Form(None),
    user=Depends(get_current_user),
):
    """Streaming chat: emits Server-Sent Events with answer text deltas."""
    from fastapi.responses import StreamingResponse
    from conversational_rag_engine import conversational_ask_stream

    user_id = user["id"]
    # Same guards as /ask: course access + session ownership come first.
    require_course_access(course_id, user)
    if session_id:
        _require_owned_session(session_id, user_id)

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
        # Track the exchange only when we actually produced an answer. Tasks
        # added mid-stream still run AFTER the response completes (FastAPI
        # attaches the injected BackgroundTasks to the StreamingResponse).
        if answer:
            background_tasks.add_task(
                _track_chat_interaction, user_id, course_id, question, answer)
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
def get_messages(session_id: str, user_id: str = Depends(current_user_id)):
    """
    Fetch all messages in a session, oldest first.

    Only the session's owner may read it; foreign/unknown sessions 404
    (see _require_owned_session for the non-enumeration rationale).
    """
    _require_owned_session(session_id, user_id)
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
def delete_session(session_id: str, user_id: str = Depends(current_user_id)):
    """
    Delete a chat session and all its messages.

    Only the session's owner may delete it; foreign/unknown sessions 404.
    """
    _require_owned_session(session_id, user_id)
    try:
        # First delete all messages in the session
        supabase.table("messages").delete().eq("session_id", session_id).execute()
        
        # Then delete the session itself
        supabase.table("chat_sessions").delete().eq("id", session_id).execute()
        
        return {"status": "ok", "message": "Session deleted successfully"}
    except Exception as e:
        logger.exception("Couldn't delete session")
        raise HTTPException(500, detail="Couldn't delete session")

