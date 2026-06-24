from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Query
from fastapi.responses import Response, StreamingResponse
from deps import *  # noqa: F401,F403  shared state, engines, helpers, stdlib re-exports

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
            raise HTTPException(500, detail=f"Couldn't create session: {e}")

    # 2) Record the user's question
    try:
        supabase.table("messages").insert({
            "session_id": session_id,
            "role": "user",
            "content": question,
            "timestamp": datetime.utcnow().isoformat()
        }).execute()
    except Exception as e:
        raise HTTPException(500, detail=f"Couldn't save question: {e}")

    # 3) **NEW: Generate conversational answer with context awareness**
    try:
        if CONVERSATIONAL_MODE:
            print("🧠 Using conversational RAG with context awareness...")
            answer = conversational_ask_question(question, course_id, session_id)
            print("✅ Conversational answer generated!")
        elif ENHANCED_MODE:
            print("🤖 Using enhanced question answering...")
            answer = enhanced_ask_question(question, course_id)
        else:
            print("📝 Using basic question answering...")
            answer = ask_question(question, course_id)
    except Exception as e:
        print(f"❌ All QA methods failed, using fallback: {e}")
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
        raise HTTPException(500, detail=f"Couldn't save answer: {e}")

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
            raise HTTPException(500, detail=f"Couldn't create session: {e}")
    try:
        supabase.table("messages").insert({
            "session_id": session_id,
            "role": "user",
            "content": question,
            "timestamp": datetime.utcnow().isoformat(),
        }).execute()
    except Exception as e:
        raise HTTPException(500, detail=f"Couldn't save question: {e}")

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
            print(f"❌ Stream failed: {e}")
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
            print(f"Couldn't save streamed answer: {e}")
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
        raise HTTPException(500, detail=f"Couldn't fetch sessions: {e}")
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
        raise HTTPException(500, detail=f"Couldn't fetch messages: {e}")
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
        raise HTTPException(500, detail=f"Couldn't delete session: {e}")

