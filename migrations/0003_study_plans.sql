-- Migration 0003 — Study planner persistence (Phase 3)
-- Apply to the live Vindexa1 DB via the Supabase SQL editor. Idempotent.

create table if not exists study_plans (
    id         uuid primary key,
    course_id  text,
    user_id    text,
    plan       jsonb,
    created_at timestamptz not null default now()
);
create index if not exists study_plans_course_idx on study_plans (course_id, created_at desc);
