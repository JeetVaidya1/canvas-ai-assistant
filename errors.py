"""Consistent API error handling.

Unhandled exceptions must never leak a stack trace or internal detail to the
client (security) and should return a predictable JSON shape the frontend can
rely on (robustness). Full detail is logged server-side for debugging.

HTTPException (the deliberate 4xx/429 we raise, e.g. rate limits, auth) keeps
FastAPI's default handling — only *unexpected* errors are caught here.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("vindexa")


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):  # noqa: ANN202
        logger.exception(
            "Unhandled error on %s %s", request.method, request.url.path
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "Something went wrong on our end. Please try again.",
            },
        )
