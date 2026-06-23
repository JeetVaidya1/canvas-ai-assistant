-- Migration 0007 — Concept prerequisite graph (Phase 4)
-- Apply to the live Vindexa1 DB via the Supabase SQL editor. Idempotent.

create table if not exists concept_graphs (
    course_id  text primary key,
    graph      jsonb,
    created_at timestamptz not null default now()
);
