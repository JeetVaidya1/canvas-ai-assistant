-- Migration 0008 — Shared class courses (Phase 5)
-- Apply to the live Vindexa1 DB via the Supabase SQL editor. Idempotent.

create table if not exists shared_courses (
    course_id    text primary key references courses(course_id) on delete cascade,
    share_code   text unique,
    title        text,
    subject      text,
    school       text,
    term         text,
    description  text,
    published_by text,
    join_count   integer default 0,
    created_at   timestamptz not null default now()
);
create index if not exists shared_courses_code_idx on shared_courses (share_code);

create table if not exists course_memberships (
    id        bigint generated always as identity primary key,
    user_id   text,
    course_id text,
    role      text default 'member',
    joined_at timestamptz not null default now(),
    unique (user_id, course_id)
);
create index if not exists course_memberships_user_idx on course_memberships (user_id);
