"""Vindexa MCP server — exposes a course as a study-memory tool for any MCP
client (Claude Desktop, Cursor, etc.).

This is the anti-wrapper play: rather than wrapping a model, Vindexa becomes the
grounded substrate a model plugs into. Each server instance is scoped to one
course (and optionally one student) via env vars, so a client can be configured
per course:

    {
      "mcpServers": {
        "vindexa-bio201": {
          "command": "python",
          "args": ["/path/to/mcp_server.py"],
          "env": {"VINDEXA_COURSE_ID": "bio201", "VINDEXA_USER_ID": "<id>",
                  "SUPABASE_URL": "...", "SUPABASE_KEY": "..."}
        }
      }
    }

Run directly for a quick smoke test: ``python mcp_server.py --selfcheck``.
"""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()

COURSE_ID = os.getenv("VINDEXA_COURSE_ID", "")
USER_ID = os.getenv("VINDEXA_USER_ID", "anonymous")


def _build_server():
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("vindexa")

    @mcp.tool()
    def search_course(query: str, top_k: int = 5) -> str:
        """Search the course's materials and return grounded passages with citations."""
        from rag.retrieval import retrieve
        rows = retrieve(query, COURSE_ID, top_k=top_k)
        if not rows:
            return "No matching course material."
        out = []
        for r in rows:
            doc = r.get("doc_name", "source")
            page = r.get("page") or r.get("slide")
            cite = f"{doc}" + (f", p.{page}" if page else "")
            out.append(f"[{cite}]\n{(r.get('content') or '').strip()[:700]}")
        return "\n\n---\n\n".join(out)

    @mcp.tool()
    def weak_topics() -> str:
        """List the student's weakest topics in this course (mastery ascending)."""
        from context_pack import _weak_topics
        weak = _weak_topics(COURSE_ID, USER_ID)
        if not weak:
            return "No mastery data yet."
        return "\n".join(f"- {w['topic']}: ~{w['mastery_pct']}% mastery" for w in weak)

    @mcp.tool()
    def exam_readiness() -> str:
        """Predicted exam-readiness score and the biggest gaps for this student."""
        import readiness_engine
        r = readiness_engine.get_readiness(COURSE_ID, USER_ID)
        gaps = ", ".join(r.get("gaps", [])) or "none"
        return f"Predicted readiness: {r.get('score_pct', 0)}% (confidence: {r.get('confidence')}). Biggest gaps: {gaps}."

    @mcp.tool()
    def context_pack() -> str:
        """A full paste-ready study brief: weak areas + grounded excerpts."""
        from context_pack import build_context_pack
        return build_context_pack(COURSE_ID, USER_ID)

    return mcp


def _selfcheck() -> int:
    """Construct the server and confirm tools register (no client needed)."""
    if not COURSE_ID:
        print("selfcheck: set VINDEXA_COURSE_ID to scope the server (continuing with empty).")
    server = _build_server()
    import asyncio
    tools = asyncio.run(server.list_tools())
    print("Vindexa MCP server OK. Tools:", [t.name for t in tools])
    return 0


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        raise SystemExit(_selfcheck())
    _build_server().run()
