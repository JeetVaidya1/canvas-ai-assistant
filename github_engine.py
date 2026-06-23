"""GitHub interop — push a course's Markdown to a repo, and import course
materials from a repo. Vindexa as 'study-as-code': your notes/decks live in git,
versioned, and a lecture-notes repo can seed a course.

All GitHub calls use the REST API. A token is required to push (and to read
private repos); public-repo reads work unauthenticated (subject to rate limits).
Tokens are passed per-request and never stored.
"""
from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional

import requests

API = "https://api.github.com"
TEXT_EXT = (".md", ".markdown", ".txt", ".rst")


def _headers(token: Optional[str]) -> Dict[str, str]:
    h = {"Accept": "application/vnd.github+json", "User-Agent": "Vindexa"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _default_branch(repo: str, token: Optional[str]) -> str:
    r = requests.get(f"{API}/repos/{repo}", headers=_headers(token), timeout=20)
    r.raise_for_status()
    return r.json().get("default_branch", "main")


def push_markdown(course_id: str, token: str, repo: str,
                  base_path: str = "vindexa", branch: Optional[str] = None) -> Dict[str, Any]:
    """Commit a course's Markdown export to ``repo`` under ``base_path``.

    Creates or updates each file via the Contents API (one commit per file).
    Requires a token with repo write access. Returns {pushed, repo, branch}.
    """
    from exports import build_course_markdown

    if not token:
        raise ValueError("A GitHub token with write access is required to push.")
    branch = branch or _default_branch(repo, token)
    files = build_course_markdown(course_id)

    pushed = []
    for rel_path, content in files.items():
        path = f"{base_path.rstrip('/')}/{course_id}/{rel_path}"
        url = f"{API}/repos/{repo}/contents/{path}"
        # Look up an existing file's sha so we update instead of failing.
        sha = None
        existing = requests.get(url, headers=_headers(token), params={"ref": branch}, timeout=20)
        if existing.status_code == 200:
            sha = existing.json().get("sha")
        body = {
            "message": f"Vindexa: update {rel_path} for {course_id}",
            "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
            "branch": branch,
        }
        if sha:
            body["sha"] = sha
        resp = requests.put(url, headers=_headers(token), json=body, timeout=30)
        resp.raise_for_status()
        pushed.append(path)

    return {"pushed": len(pushed), "files": pushed, "repo": repo, "branch": branch}


def list_repo_text_files(repo: str, token: Optional[str] = None,
                         subdir: str = "") -> List[Dict[str, str]]:
    """List text/markdown files in a repo (recursively) via the git tree API."""
    branch = _default_branch(repo, token)
    r = requests.get(f"{API}/repos/{repo}/git/trees/{branch}",
                     headers=_headers(token), params={"recursive": "1"}, timeout=30)
    r.raise_for_status()
    tree = r.json().get("tree", [])
    prefix = subdir.strip("/")
    out = []
    for node in tree:
        if node.get("type") != "blob":
            continue
        path = node.get("path", "")
        if prefix and not path.startswith(prefix):
            continue
        if path.lower().endswith(TEXT_EXT):
            out.append({"path": path, "sha": node.get("sha", "")})
    return out


def fetch_file(repo: str, path: str, token: Optional[str] = None) -> str:
    r = requests.get(f"{API}/repos/{repo}/contents/{path}",
                     headers=_headers(token), timeout=30)
    r.raise_for_status()
    data = r.json()
    if data.get("encoding") == "base64":
        return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    return data.get("content", "")


def import_repo_materials(course_id: str, repo: str, token: Optional[str] = None,
                          subdir: str = "", max_files: int = 50) -> Dict[str, Any]:
    """Import text/markdown files from ``repo`` into a course as study materials.

    Each file is ingested through the normal pipeline so it becomes searchable
    course content. Returns {imported, skipped, files}.
    """
    files = list_repo_text_files(repo, token, subdir)[:max_files]
    if not files:
        return {"imported": 0, "skipped": 0, "files": [], "message": "No text files found in repo."}

    from ingest import process_file

    imported, skipped, names = 0, 0, []
    for f in files:
        path = f["path"]
        try:
            text = fetch_file(repo, path, token)
            if not text.strip():
                skipped += 1
                continue
            filename = path.replace("/", "__")
            process_file(filename, text.encode("utf-8"), course_id)
            imported += 1
            names.append(path)
        except Exception as e:  # noqa: BLE001
            print(f"GitHub import skipped {path}: {e}")
            skipped += 1

    return {"imported": imported, "skipped": skipped, "files": names}
