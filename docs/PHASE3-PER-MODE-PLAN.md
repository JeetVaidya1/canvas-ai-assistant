# Phase 3 — Per-Mode Upgrades (execution plan)

Detailed plan for the next session, derived from three end-to-end audits (quiz, exam,
and the other modes). Phases 1 & 2 are done: markdown/streaming/citations, structured
outputs, router/deps refactor, and the RAG retrieval rebuild (hybrid + reranker,
recall@1 0.71→0.83). This document covers the remaining per-mode quality work.

## Shared foundations (use these everywhere)
- **Structured output**: `providers.structured_call(messages, schema=..., tool_name=...)` →
  schema-guaranteed dict via Claude tool use. Replace all `json.loads(model_text)` + regex.
- **Grounded retrieval**: `rag.retrieval.retrieve(question, course_id, top_k)` →
  hybrid (BGE-prefixed vector + Postgres FTS, RRF) + cross-encoder rerank.
- **Frontend rendering**: `@/components/ui/Markdown` for any AI text (already applied to
  chat/notes/quiz/practice/exam/flashcards).
- **Analytics + adaptive difficulty**: `learning_analytics.py` (track interactions,
  per-topic mastery). Route question difficulty off mastery.
- **DB**: apply schema changes to `schema.sql` AND to the live Vindexa1 DB
  (project ref `eddozjbdezpdcwuxuzpo`) via the SQL editor.
- **The interactive pattern** (generate → answer one-at-a-time → instant feedback →
  score → feed analytics) is shared by **Quiz, Exam, and Practice** — build it once for
  Quiz, reuse for the others.

## Recommended order
1. Quiz rebuild (highest impact; establishes the shared runner pattern)
2. Exam quality pass (flow already fixed in PR #8)
3. Planner (net-new, currently a placeholder calling missing endpoints)
4. Analytics charts
5. Practice (adaptive difficulty + structured outputs)
6. Notes (style-aware + edit-in-place)
7. Flashcards (spaced repetition)

---

## 1. QUIZ — rebuild into a real quiz runner  (L)
**Problem:** `quiz_assistant_engine.py` + `QuizAsisstant.tsx` only *answer a pasted
question* (`/quiz-assist`). Not a quiz. Brittle `parse_quiz_question` heuristics; no
generation, scoring, or analytics.

**Target:** generate N grounded MCQs → answer one at a time → instant feedback +
explanation + source → final score + per-topic weak areas → feed `learning_analytics`.

**Backend** (new `quiz_engine.py` + routes in `routers/quiz.py`, keep `/quiz-assist` for back-compat):
- `POST /quiz/generate` (course_id, topic?, num_questions=10, difficulty=medium) →
  `{quiz_id, questions:[{id, question, options[4], correct_answer, explanation, concept,
  difficulty, source:{doc_name,page}}]}`. Reuse `practice_generator` MCQ generation +
  `rag.retrieval.retrieve` for grounding; build questions with `structured_call`
  (QUIZ_QUESTION_SCHEMA). Persist to new tables.
- `POST /quiz/{quiz_id}/answer` (question_id, selected, time_taken) →
  `{is_correct, correct_answer, explanation, concept, source}`; call
  `learning_analytics.track_quiz_answer` + update concept mastery.
- `POST /quiz/{quiz_id}/submit` → `{score:{correct,total,pct}, by_topic[], weak_areas[]}`.

**DB:** `quiz_sessions(id, course_id, user_id, topic, difficulty, created_at, status)`,
`quiz_questions(quiz_id, question_id, question, options jsonb, correct_answer,
explanation, concept, source_doc, source_page)`,
`quiz_responses(quiz_id, question_id, user_id, selected, is_correct, time_taken, ts)`.

**Frontend** (`QuizMode.tsx` replacing `QuizAsisstant.tsx`; keep the old one as
"Answer helper" if desired): setup (topic/difficulty/count) → runner (one question,
radio options, submit → feedback card with green/red + explanation + `<Markdown>` +
source chip) → results (score, per-topic Recharts bar, weak areas → links to Practice).
`lib/api.ts`: `generateQuiz`, `submitQuizAnswer`, `submitQuiz`.

**Verify:** generate → answer all → submit → score shows; wrong answers appear in
`/analytics`. **Effort L.**

---

## 2. EXAM — quality pass  (M)  (flow fixed in PR #8)
Remaining issues (all in `exam_generator.py` / `exam_session_manager.py`):
- **Difficulty ignored**: forced `"hard"` at `estimate_difficulty` return, the
  `generate_practice_exam` default, and `validate_and_clean_questions` (`q["difficulty"]="hard"`),
  plus the prompt says "Generate HARD questions". → respect the requested difficulty.
- **MCQs stripped**: `generate_exam_questions` filters out `multiple_choice` and
  `validate_and_clean_questions` converts MC→short_answer. → support MCQ when requested.
- **Grading too lax**: `compare_text_answers` accepts 60% word overlap. → grade
  short_answer/essay with `structured_call` (AI judge: correct/partial/incorrect + reason);
  keep exact match for MCQ.
- **Timing is a stub**: `calculate_question_time` returns 0. → record
  `current_question_start_time` on navigate; compute elapsed on save.
- Migrate exam generation to `structured_call` + `rag.retrieval.retrieve`.

**Verify:** generate easy/medium/hard → questions match; MCQ renders + grades; short
answers AI-graded; per-question time non-zero; results persist + show in exam-history.
**Effort M.**

---

## 3. PLANNER — build it  (L)
**Problem:** `PlannerPage.tsx` is a "Coming Soon" placeholder; `lib/api.ts` calls
`/api/generate-study-plan` and `/api/study-plan/{course_id}` that **don't exist**.

**Backend** (new `routers/planner.py` + `planner_engine.py`):
- `POST /api/generate-study-plan` (course_id, days_available, hours_per_day, exam_date?) →
  `StudyPlan{id, course_id, days:[{date, topics[], duration_minutes, type:review|new|practice}],
  created_at}`. Extract topics (reuse `practice_generator` topic extraction), distribute
  across days with spaced-repetition spacing (review at +1/+3/+7 days), build via
  `structured_call`. Persist `study_plans(id, course_id, user_id, plan jsonb, created_at)`.
- `GET /api/study-plan/{course_id}` → latest plan or null.
- Register the router in `main.py`.

**Frontend:** replace the placeholder `PlannerPage.tsx` with a form (days/hours/exam date)
→ generate → calendar/list view → "Export to iCal" (already wired to
`exportPlannerIcal` → `exports.build_planner_ics`).

**Verify:** generate plan → renders days/topics → iCal export downloads. **Effort L.**

---

## 4. ANALYTICS — real charts  (M)
**Problem:** `AnalyticsDashboard.tsx` uses hand-rolled CSS progress bars;
`learning_analytics.study_time_trend` returns `[]` (computed-but-unused field).

- Add **Recharts**. Replace mastery bars with a real bar chart; add a line chart for
  study-time trend; confidence-over-time.
- Implement `study_time_trend` in `learning_analytics.py`: group `user_interactions` by
  date (reuse the streak grouping), return `[{date, questions, duration_minutes}]`.
- Ensure `/track-interaction` + `/track-practice-session` (and the new Quiz tracking)
  actually populate data; persist a `practice_sessions` row on track-practice-session.

**Verify:** after using chat/quiz/practice, `/analytics` shows populated charts +
trend. **Effort M.**

---

## 5. PRACTICE — adaptive difficulty + structured outputs  (M)
`practice_generator.py`:
- **Structured outputs**: replace `json.loads(content)` + ```json strip (~line 715) with
  `structured_call` (PROBLEM_SCHEMA). Removes the fragile parsing.
- **Adaptive difficulty**: look up `learning_analytics` mastery for the topic; route
  easy/medium/hard by mastery (<0.5 easy, 0.5–0.8 medium, >0.8 hard) instead of the
  hardcoded "hard"/"medium" default.
- Use `rag.retrieval.retrieve` for grounding (currently `advanced_rag_engine.hybrid_search`).
- Remove the duplicate `/practice-topics/{course_id}` route (two handlers in
  `routers/practice.py`).

**Verify:** practice on a mastered vs weak topic yields harder vs easier problems;
problems always parse. **Effort M.**

---

## 6. NOTES — style-aware + edit-in-place  (S + M)
Already done: markdown rendering + structured flashcards. Remaining:
- **(S) Make the `style` selector work**: `notes_engine._notes_instruction` ignores
  `style`. Branch the prompt: `detailed` = current 11 sections; `summary` = ≤500 words,
  overview + key points; `outline` = nested bullets only. The frontend dropdown already
  sends it.
- **(M) Edit saved notes in-place**: add `PUT /notes/{note_id}` (reuse
  `save_notes_to_db`); in `NotesCreator.tsx` load a saved note into the editor (it already
  tracks `currentNoteId`) and allow edit + resave.
- (Optional) single structured pass instead of draft+polish; validate sections present.

**Verify:** each style produces visibly different output; editing a saved note updates it
without regenerating. **Effort S+M.**

---

## 7. FLASHCARDS — spaced repetition + persistence  (M)
`FlashCards.tsx` has flip/hide/typing modes but no SR or persistence.
- Persist cards + reviews: `flashcards(id, course_id, note_id?, q, a, created_at)`,
  `flashcard_reviews(card_id, user_id, ease, interval, due_date, last_reviewed)`.
- Implement **SM-2**: `POST /flashcards/review` (card_id, grade 0–5) updates interval/ease/due.
- Frontend: surface "due" cards first; rate recall after each card.
- Anki export already works (`exports.build_flashcards_apkg`).

**Verify:** review a card → its due date advances; due cards surface first next session.
**Effort M.**

---

## Notes for the executor
- Commit each mode as its own PR (branch off `main`, verify live, merge). Keep route
  paths stable so the frontend keeps working.
- For each new DB table/RPC: add to `schema.sql` **and** apply to the live DB.
- The dev loop: backend `uvicorn main:app --port 8000`; frontend `npm run dev` (5173);
  `.env` uses Max OAuth (no API key) + Vindexa1 service key.
- Re-run `python -m rag.eval` if you touch retrieval; keep the regression baseline.
