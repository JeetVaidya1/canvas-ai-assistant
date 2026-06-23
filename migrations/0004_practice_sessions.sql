-- Migration 0004 — Practice session persistence for analytics (Phase 3)
-- Apply to the live Vindexa1 DB via the Supabase SQL editor. Idempotent.

create table if not exists practice_sessions (
    id                 bigint generated always as identity primary key,
    user_id            text,
    course_id          text,
    topic              text,
    problems_attempted integer,
    problems_correct   integer,
    duration_minutes   integer,
    difficulty_level   text,
    created_at         timestamptz not null default now()
);
create index if not exists practice_sessions_idx on practice_sessions (user_id, course_id, created_at desc);
