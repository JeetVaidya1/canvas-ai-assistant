from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import Response

import logging

logger = logging.getLogger(__name__)

import exports
import github_engine

router = APIRouter()


@router.get("/api/export-markdown/{course_id}")
def export_markdown_endpoint(course_id: str):
    """Download a course's notes, flashcards, and study plan as a Markdown .zip."""
    try:
        zip_bytes = exports.build_course_markdown_zip(course_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not export markdown: {e}")
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{course_id}_markdown.zip"'},
    )


@router.post("/api/github/push")
async def github_push_endpoint(
    course_id: str = Form(...),
    repo: str = Form(...),          # "owner/name"
    token: str = Form(...),
    base_path: str = Form("vindexa"),
):
    """Commit the course's Markdown export to a GitHub repo (study-as-code)."""
    if not (course_id and repo and token):
        raise HTTPException(400, detail="course_id, repo, and token are required")
    try:
        return github_engine.push_markdown(course_id, token, repo, base_path)
    except Exception as e:
        print(f"GitHub push failed: {e}")
        logger.exception("GitHub push failed")
        raise HTTPException(502, detail="GitHub push failed")


@router.post("/api/github/import")
async def github_import_endpoint(
    course_id: str = Form(...),
    repo: str = Form(...),          # "owner/name"
    token: str | None = Form(None),
    subdir: str = Form(""),
):
    """Import text/markdown files from a GitHub repo into a course as materials."""
    if not (course_id and repo):
        raise HTTPException(400, detail="course_id and repo are required")
    try:
        return github_engine.import_repo_materials(course_id, repo, token, subdir)
    except Exception as e:
        print(f"GitHub import failed: {e}")
        logger.exception("GitHub import failed")
        raise HTTPException(502, detail="GitHub import failed")
