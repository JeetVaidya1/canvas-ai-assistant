-- Migration 0002 — Per-question exam timing (Phase 3)
-- Adds a column to record when the student arrived at the current question, so
-- save_answer can compute real time-spent (previously a 0 stub).
-- Apply to the live Vindexa1 DB via the Supabase SQL editor. Idempotent.

alter table exam_sessions
    add column if not exists current_question_start_time timestamptz;
