-- Migration 0006 — Mistake-driven review queue (Phase 4)
-- Apply to the live Vindexa1 DB via the Supabase SQL editor. Idempotent.

create table if not exists review_items (
    id            uuid primary key,
    user_id       text,
    course_id     text,
    concept       text,
    prompt        text,
    answer        text,
    explanation   text,
    source        text,
    ease          double precision default 2.5,
    interval      integer default 0,
    repetitions   integer default 0,
    due_date      date,
    last_reviewed timestamptz,
    status        text default 'active',
    created_at    timestamptz not null default now()
);
create index if not exists review_items_due_idx on review_items (user_id, course_id, status, due_date);
