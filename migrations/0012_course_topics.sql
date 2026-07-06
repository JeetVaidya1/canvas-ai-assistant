-- Migration 0012 — Course Brain topics (V3 workstream A)
-- Apply to the live Vindexa1 DB via the Supabase SQL editor. Idempotent.
--
-- One row per synthesized course topic. Topics are content-grounded (LLM
-- synthesis over sampled chunks), replacing the legacy filename-regex topics.
-- Rebuilds are delete+insert per course, so `unique (course_id, slug)` is the
-- only identity that matters across regenerations.

create extension if not exists pgcrypto;  -- gen_random_uuid()

create table if not exists course_topics (
    id           uuid primary key default gen_random_uuid(),
    course_id    text not null references courses(course_id) on delete cascade,
    slug         text not null,             -- kebab-case stable key
    name         text not null,             -- clean human title (1-4 words)
    description  text,                      -- one sentence
    doc_coverage jsonb default '[]',        -- [{"doc": filename, "pages": [min, max]}]
    prereq_slugs text[] default '{}',       -- edges within this course's topic set
    position     int default 0,             -- teaching order (0-based)
    created_at   timestamptz default now(),
    unique (course_id, slug)
);

create index if not exists course_topics_course_idx on course_topics (course_id);
