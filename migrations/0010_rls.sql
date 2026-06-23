-- Migration 0010 — Row-level security (Phase 6 M4)
-- Backend uses the service-role key (bypasses RLS); the only direct client read is
-- `courses`, which gets an owner-or-member SELECT policy. All other tables get RLS
-- enabled with no anon policy => denied to the publishable key, backend unaffected.

-- courses: client lists only courses it owns or has joined.
alter table courses enable row level security;
drop policy if exists courses_select_own on courses;
create policy courses_select_own on courses for select
  using (
    owner_id = auth.uid()::text
    or exists (select 1 from course_memberships m
               where m.course_id = courses.course_id and m.user_id = auth.uid()::text)
  );

-- Everything else: lock to the backend (service-role bypasses RLS).
alter table files               enable row level security;
alter table embeddings          enable row level security;
alter table chat_sessions       enable row level security;
alter table messages            enable row level security;
alter table exam_sessions       enable row level security;
alter table learning_progress   enable row level security;
alter table user_interactions   enable row level security;
alter table practice_sessions   enable row level security;
alter table notes               enable row level security;
alter table past_papers         enable row level security;
alter table past_paper_analyses enable row level security;
alter table quiz_sessions       enable row level security;
alter table quiz_questions      enable row level security;
alter table quiz_responses      enable row level security;
alter table study_plans         enable row level security;
alter table flashcards          enable row level security;
alter table flashcard_reviews   enable row level security;
alter table review_items        enable row level security;
alter table concept_graphs      enable row level security;
alter table shared_courses      enable row level security;
alter table course_memberships  enable row level security;

-- Let the courses policy's membership subquery match: users see their own memberships.
drop policy if exists memberships_select_own on course_memberships;
create policy memberships_select_own on course_memberships for select
  using (user_id = auth.uid()::text);
