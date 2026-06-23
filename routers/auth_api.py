from fastapi import APIRouter, Depends
from deps import *  # noqa: F401,F403  shared state, engines, helpers, stdlib re-exports

from auth import get_current_user

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
    """
    try:
        unowned = supabase.table("courses").select("course_id").is_("owner_id", "null").execute().data or []
        for c in unowned:
            supabase.table("courses").update({"owner_id": user["id"]}).eq("course_id", c["course_id"]).execute()
        return {"claimed": len(unowned)}
    except Exception as e:
        print(f"claim_legacy_data failed: {e}")
        return {"claimed": 0, "error": str(e)}
