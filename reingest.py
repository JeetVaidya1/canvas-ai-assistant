#!/usr/bin/env python
"""
Repopulate a fresh database from the bundled course files in data/.

Run AFTER applying schema.sql to your Supabase project and filling in .env:

    .venv/bin/python reingest.py            # enhanced (vision) ingest
    .venv/bin/python reingest.py --basic    # text-only, faster/cheaper

For each data/<course_id>/<file> it:
  1. upserts the course row (title from courses.json when available),
  2. uploads the file to the 'course-files' storage bucket,
  3. upserts the files row,
  4. embeds + stores chunks in the 'embeddings' table (1024-dim, local model).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
SUPPORTED = {".pdf", ".pptx", ".docx", ".txt", ".md"}


def _require_env() -> None:
    missing = [k for k in ("SUPABASE_URL", "SUPABASE_KEY") if not os.getenv(k)]
    if missing:
        sys.exit(f"Missing required env vars: {', '.join(missing)} (see .env.example)")
    # Claude auth may be an API key OR the Claude Code/Max keychain token.
    from providers.claude_auth import resolve_auth
    try:
        mode, _ = resolve_auth()
    except Exception as exc:  # noqa: BLE001
        sys.exit(str(exc))
    print(f"Claude auth: {mode}")


def _load_titles() -> dict[str, str]:
    path = Path(__file__).parent / "courses.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {cid: meta.get("title", cid) for cid, meta in data.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basic", action="store_true",
                        help="use text-only ingest instead of the vision path")
    args = parser.parse_args()

    _require_env()

    # Imported here so env is loaded first.
    from supabase import create_client
    from storage import upload_file

    if args.basic:
        from ingest import process_file as ingest_fn
    else:
        try:
            from enhanced_ingest import process_file_enhanced as ingest_fn
        except Exception as exc:  # noqa: BLE001
            print(f"Enhanced ingest unavailable ({exc}); falling back to basic.")
            from ingest import process_file as ingest_fn

    supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    titles = _load_titles()

    if not DATA_DIR.exists():
        sys.exit(f"No data directory at {DATA_DIR}")

    total_files = 0
    total_chunks = 0
    for course_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        course_id = course_dir.name
        title = titles.get(course_id, course_id)
        supabase.table("courses").upsert(
            {"course_id": course_id, "title": title},
            on_conflict="course_id",
        ).execute()
        print(f"\n=== course '{course_id}' ({title}) ===")

        for file_path in sorted(course_dir.iterdir()):
            if file_path.suffix.lower() not in SUPPORTED:
                continue
            content = file_path.read_bytes()
            storage_path = f"{course_id}/{file_path.name}"
            try:
                upload_file("course-files", content, storage_path)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! storage upload failed for {file_path.name}: {exc}")

            supabase.table("files").upsert(
                {
                    "course_id": course_id,
                    "filename": file_path.name,
                    "storage_path": storage_path,
                    "file_type": file_path.suffix.lstrip("."),
                    "ext": file_path.suffix.lstrip("."),
                    "uploaded_at": datetime.now(timezone.utc).isoformat(),
                },
                on_conflict="course_id,filename",
            ).execute()

            chunks = ingest_fn(file_path.name, content, course_id)
            n = len(chunks) if isinstance(chunks, list) else 0
            total_files += 1
            total_chunks += n
            print(f"  + {file_path.name}: {n} chunks")

    print(f"\nDone. {total_files} files, {total_chunks} chunks embedded.")


if __name__ == "__main__":
    main()
