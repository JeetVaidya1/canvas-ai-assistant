from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from auth import current_user_id

import logging

logger = logging.getLogger(__name__)

import context_pack

router = APIRouter()


@router.get("/api/context-pack/{course_id}/{user_id}")
async def context_pack_endpoint(course_id: str, user_id: str, _auth: str = Depends(current_user_id)):
    user_id = _auth
    """A paste-ready Markdown study brief (weak areas + grounded excerpts) for any AI."""
    try:
        return {"markdown": context_pack.build_context_pack(course_id, user_id)}
    except Exception as e:
        print(f"Context pack build failed: {e}")
        logger.exception("Context pack build failed")
        raise HTTPException(500, detail="Context pack build failed")


@router.get("/api/context-pack/{course_id}/{user_id}/download", response_class=PlainTextResponse)
async def context_pack_download(course_id: str, user_id: str, _auth: str = Depends(current_user_id)):
    user_id = _auth
    """Same context pack as a downloadable .md file."""
    try:
        md = context_pack.build_context_pack(course_id, user_id)
        return PlainTextResponse(
            content=md,
            headers={"Content-Disposition": f'attachment; filename="{course_id}_context.md"'},
        )
    except Exception as e:
        logger.exception("Context pack build failed")
        raise HTTPException(500, detail="Context pack build failed")
