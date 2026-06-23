from fastapi import APIRouter, Depends
from deps import *  # noqa: F401,F403  shared state, engines, helpers, stdlib re-exports

from auth import get_current_user

router = APIRouter()


@router.get("/api/me")
async def me_endpoint(user=Depends(get_current_user)):
    """Return the authenticated user's identity (proves the token is valid)."""
    return {"id": user["id"], "email": user.get("email")}
