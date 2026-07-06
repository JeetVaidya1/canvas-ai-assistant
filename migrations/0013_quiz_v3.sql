-- 0013_quiz_v3.sql — fast-start drills + confidence calibration (V3 workstreams B/E).
-- Idempotent: safe to run more than once.
--
-- quiz_sessions gains:
--   num_requested     how many questions the user asked for (num_questions tracks
--                     how many exist so far; they converge when generation finishes)
--   generation_status 'ready' | 'generating' | 'partial'
--                     'generating': background batch still writing questions
--                     'partial':    background batch failed; session stays playable
--                                   with the questions that exist
--
-- user_id backfill note: rows created before this migration have user_id NULL
-- (sessions were anonymous-friendly; the engine now stamps the token user on
-- insert). Old NULL rows are left as-is — per-user reads treat NULL as legacy
-- and ownership checks admit them.
--
-- quiz_responses gains:
--   confidence  'sure' | 'thinkso' | 'guessing' | NULL — the learner's
--               pre-reveal confidence tap, powering the calibration read-out.

alter table quiz_sessions  add column if not exists num_requested     integer;
alter table quiz_sessions  add column if not exists generation_status text default 'ready';
alter table quiz_responses add column if not exists confidence        text;
