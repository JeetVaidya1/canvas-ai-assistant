"""Canvas LMS import — pull a course's syllabus, due dates, and materials, and
auto-detect the next exam so the planner can build around it.

Kills the cold-start problem: instead of manually uploading files and typing an
exam date, a student connects their Canvas course and Vindexa ingests the
syllabus + materials and learns when the exam is.

Canvas REST API: https://canvas.<institution>/api/v1. A per-institution base URL
and a user access token are required (both supplied per-request, never stored).
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

EXAM_WORDS = ("exam", "midterm", "final", "test", "quiz")
MATERIAL_EXT = (".pdf", ".docx", ".pptx", ".txt", ".md")


def _headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def _api(base_url: str, path: str, token: str, params: Optional[Dict] = None) -> Any:
    url = f"{base_url.rstrip('/')}/api/v1{path}"
    r = requests.get(url, headers=_headers(token), params=params or {}, timeout=30)
    r.raise_for_status()
    return r.json()


def _is_exam(name: str) -> bool:
    n = (name or "").lower()
    return any(w in n for w in EXAM_WORDS)


def _collect_due_dates(items: List[Dict], name_key: str = "name") -> List[Dict[str, str]]:
    out = []
    for it in items or []:
        due = it.get("due_at")
        name = it.get(name_key) or it.get("title") or ""
        if due:
            out.append({"name": name, "due_at": due, "is_exam": _is_exam(name)})
    return out


def import_course(base_url: str, token: str, canvas_course_id: str, course_id: str,
                  ingest_materials: bool = True, max_files: int = 20) -> Dict[str, Any]:
    """Import a Canvas course into the app course ``course_id``.

    Returns a summary including detected exam due dates (soonest first) so the
    caller can prefill the planner.
    """
    summary: Dict[str, Any] = {"syllabus_imported": False, "assignments": [],
                               "exam_dates": [], "materials_imported": 0, "errors": []}

    # 1) Course + syllabus
    try:
        course = _api(base_url, f"/courses/{canvas_course_id}", token,
                      {"include[]": "syllabus_body"})
        syllabus_html = course.get("syllabus_body") or ""
        if syllabus_html and ingest_materials:
            text = re.sub(r"<[^>]+>", " ", syllabus_html)  # strip HTML tags
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                from ingest import process_file
                process_file("Canvas Syllabus.txt", text.encode("utf-8"), course_id)
                summary["syllabus_imported"] = True
    except Exception as e:  # noqa: BLE001
        summary["errors"].append(f"syllabus: {e}")

    # 2) Assignments + quizzes -> due dates / exam detection
    due_dates: List[Dict[str, str]] = []
    try:
        assignments = _api(base_url, f"/courses/{canvas_course_id}/assignments", token,
                           {"per_page": 100})
        due_dates += _collect_due_dates(assignments, name_key="name")
    except Exception as e:  # noqa: BLE001
        summary["errors"].append(f"assignments: {e}")
    try:
        quizzes = _api(base_url, f"/courses/{canvas_course_id}/quizzes", token,
                       {"per_page": 100})
        due_dates += _collect_due_dates(quizzes, name_key="title")
    except Exception as e:  # noqa: BLE001
        summary["errors"].append(f"quizzes: {e}")

    summary["assignments"] = due_dates
    exams = sorted([d for d in due_dates if d["is_exam"]], key=lambda d: d["due_at"])
    # Only keep exams still in the future when possible.
    now = datetime.utcnow().isoformat()
    upcoming = [e for e in exams if e["due_at"] >= now] or exams
    summary["exam_dates"] = upcoming
    summary["next_exam_date"] = upcoming[0]["due_at"][:10] if upcoming else None

    # 3) Course files -> materials (capped; best-effort)
    if ingest_materials:
        try:
            files = _api(base_url, f"/courses/{canvas_course_id}/files", token, {"per_page": 100})
            from ingest import process_file
            count = 0
            for f in files or []:
                if count >= max_files:
                    break
                name = f.get("display_name") or f.get("filename") or ""
                url = f.get("url")
                if not url or not name.lower().endswith(MATERIAL_EXT):
                    continue
                try:
                    blob = requests.get(url, headers=_headers(token), timeout=60).content
                    process_file(name, blob, course_id)
                    count += 1
                except Exception as e:  # noqa: BLE001
                    summary["errors"].append(f"file {name}: {e}")
            summary["materials_imported"] = count
        except Exception as e:  # noqa: BLE001
            summary["errors"].append(f"files: {e}")

    return summary
