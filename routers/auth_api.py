import logging

from fastapi import APIRouter, Depends, HTTPException
from deps import supabase

from auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/me")
async def me_endpoint(user=Depends(get_current_user)):
    """Return the authenticated user's identity (proves the token is valid)."""
    return {"id": user["id"], "email": user.get("email")}


@router.post("/api/claim-legacy-data")
async def claim_legacy_data(user=Depends(get_current_user)):
    """First sign-in claims any unowned (legacy) courses for this account.

    Idempotent: only courses with owner_id IS NULL are claimed, so subsequent
    users get nothing. Course-scoped data follows course ownership automatically.

    This is the ONLY path to unclaimed legacy courses: auth.user_owns_or_member
    denies access to courses with owner_id NULL, so a user must claim them here
    before any course-scoped endpoint will serve them.
    """
    try:
        unowned = supabase.table("courses").select("course_id").is_("owner_id", "null").execute().data or []
        for c in unowned:
            supabase.table("courses").update({"owner_id": user["id"]}).eq("course_id", c["course_id"]).execute()
        return {"claimed": len(unowned)}
    except Exception:
        logger.exception("claim_legacy_data failed for user %s", user["id"])
        raise HTTPException(500, detail="Could not claim legacy courses. Please try again.")
