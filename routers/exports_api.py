from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Query
from fastapi.responses import Response, StreamingResponse
from deps import *  # noqa: F401,F403  shared state, engines, helpers, stdlib re-exports

router = APIRouter()


@router.get("/api/export-notes-pdf/{course_id}")
def export_notes_pdf(course_id: str):
    """Download the course's study notes as a PDF."""
    try:
        pdf_bytes = exports.build_notes_pdf(course_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not export notes: {e}")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{course_id}_notes.pdf"'},
    )


@router.get("/api/export-flashcards-anki/{course_id}")
def export_flashcards_anki(course_id: str, user_id: str | None = None):
    """Download course flashcards as an Anki .apkg deck.

    With user_id and a saved deck, exports that deck with each card's SM-2 state
    (next-due, interval) baked in as scheduling info + tags, and stable GUIDs so
    re-imports update rather than duplicate."""
    try:
        apkg_bytes = exports.build_flashcards_apkg(course_id, user_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not export flashcards: {e}")
    return Response(
        content=apkg_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{course_id}_flashcards.apkg"'},
    )


@router.get("/api/export-planner-ical/{course_id}")
def export_planner_ical(course_id: str):
    """Download a generated study plan as an iCalendar (.ics) file."""
    try:
        ics_bytes = exports.build_planner_ics(course_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not export planner: {e}")
    return Response(
        content=ics_bytes,
        media_type="text/calendar",
        headers={"Content-Disposition": f'attachment; filename="{course_id}_study_plan.ics"'},
    )

