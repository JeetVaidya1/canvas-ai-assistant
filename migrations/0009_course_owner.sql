-- Migration 0009 — Course ownership (Phase 6). Idempotent.
alter table courses add column if not exists owner_id text;
create index if not exists courses_owner_idx on courses (owner_id);
