-- Migration 0005 — Flashcards + spaced repetition (Phase 3)
-- Apply to the live Vindexa1 DB via the Supabase SQL editor. Idempotent.

create table if not exists flashcards (
    id         uuid primary key,
    course_id  text,
    note_id    uuid,
    q          text,
    a          text,
    created_at timestamptz not null default now()
);
create index if not exists flashcards_course_idx on flashcards (course_id);

create table if not exists flashcard_reviews (
    id            bigint generated always as identity primary key,
    card_id       uuid references flashcards(id) on delete cascade,
    user_id       text,
    ease          double precision default 2.5,
    interval      integer default 0,
    repetitions   integer default 0,
    due_date      date,
    last_reviewed timestamptz,
    unique (card_id, user_id)
);
create index if not exists flashcard_reviews_user_idx on flashcard_reviews (user_id, due_date);
