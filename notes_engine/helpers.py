# notes_engine/helpers.py — Small pure utilities
import json
from typing import List


def _is_dict(x) -> bool:
    return isinstance(x, dict)

def _safe_get(d: dict, k: str, default=None):
    try:
        return d.get(k, default)
    except Exception:
        return default

def _safe_json_obj(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                return json.loads(text[start:end+1])
        except Exception:
            return {}
    return {}

def _prettify_notes(notes: str) -> str:
    import re
    notes = re.sub(r'\n{3,}', '\n\n', notes)
    notes = re.sub(r'\*\*([^*]+)\*\*', r'\1', notes)
    return notes.strip()

def _nice_fallback_title(topic: str, files: List[str]) -> str:
    base = (topic or "").strip()
    if not base:
        cleaned = [f.rsplit(".",1)[0].replace("_"," ").replace("-", " ").title() for f in files[:2]]
        base = " • ".join(cleaned) if cleaned else "Course Notes"
    return f"{base} — Concepts, Implementations & Pitfalls"[:120]
