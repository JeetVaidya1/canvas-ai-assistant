-- Migration 0001 — Quiz runner tables (Phase 3)
-- Apply to the live Vindexa1 DB (project ref eddozjbdezpdcwuxuzpo) via the
-- Supabase SQL editor. Idempotent (create if not exists).

create table if not exists quiz_sessions (
    id            uuid primary key,
    course_id     text,
    user_id       text,
    topic         text,
    difficulty    text,
    num_questions integer,
    status        text default 'active',
    score         jsonb,
    created_at    timestamptz not null default now()
);
create index if not exists quiz_sessions_course_idx on quiz_sessions (course_id);

create table if not exists quiz_questions (
    id             bigint generated always as identity primary key,
    quiz_id        uuid references quiz_sessions(id) on delete cascade,
    question_id    text,
    question       text,
    options        jsonb,
    correct_answer text,
    explanation    text,
    concept        text,
    difficulty     text,
    source_doc     text,
    source_page    integer
);
create index if not exists quiz_questions_quiz_idx on quiz_questions (quiz_id);

create table if not exists quiz_responses (
    id          bigint generated always as identity primary key,
    quiz_id     uuid references quiz_sessions(id) on delete cascade,
    question_id text,
    user_id     text,
    selected    text,
    is_correct  boolean,
    time_taken  double precision,
    ts          timestamptz not null default now()
);
create index if not exists quiz_responses_quiz_idx on quiz_responses (quiz_id, user_id);
