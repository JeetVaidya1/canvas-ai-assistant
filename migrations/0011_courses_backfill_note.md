# 0011 — courses.json retirement: manual backfill note (no SQL to run blindly)

## What changed in the code

The backend no longer reads or writes the local `courses.json` file
(`deps.load_courses` / `deps.save_courses` are gone). Course records now live
exclusively in the Supabase `courses` table (schema.sql: `course_id`, `title`,
`owner_id`, `created_at`), with per-course files in the `files` table. The new
query layer is `core/courses_store.py`.

The `courses.json` file itself is intentionally left on disk as legacy data —
do **not** delete it until the backfill check below has been done in prod.

## Why a manual step may be needed

`courses.json` was per-instance mutable state. If any course was ever created
while Supabase was unreachable (the old code fell back to JSON-only writes and
still returned 200), that course exists in `courses.json` but not in the
`courses` table — and it will be invisible to the API after this change.

## Manual check (read-only)

1. Copy `courses.json` from the deployed instance (or the repo checkout that
   ran in prod).
2. For each `course_id` key in the JSON, check it exists in the DB:

   ```sql
   select course_id from courses where course_id in ('<id1>', '<id2>', ...);
   ```

3. Diff the two lists. If nothing is missing (expected — every recent code
   path wrote to Supabase as well), no action is needed.

## Backfill (only for ids missing from the DB)

Idempotent upsert, one row per missing course; titles come from the JSON:

```sql
insert into courses (course_id, title, owner_id)
values ('<missing_id>', '<title from courses.json>', null)
on conflict (course_id) do nothing;
```

Leave `owner_id` NULL: unowned legacy courses are deliberately inaccessible
until a user claims them via `POST /api/claim-legacy-data` (see auth.py's
`user_owns_or_member` for the security rationale).

Per-course file lists in the JSON (`files: [...]`) do **not** need backfilling:
the `files` table has been dual-written since the Supabase migration, and file
entries without embeddings would need re-ingestion anyway (`reingest.py`).

## Nothing destructive

This note requires no schema change and drops no data. `reingest.py` (legacy
script, excluded from the app) still reads `courses.json` for titles; that is
unaffected.
