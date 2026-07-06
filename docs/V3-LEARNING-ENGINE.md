# V3 — The Learning Engine

Mandate (Jeet, 2026-07-06): evaluate everything from the core; make each feature genuinely
world-class, not just reskinned. Redo layouts and UX around the new mechanics — the current
layouts are the baseline to beat, not preserve. If backend code isn't world-class for what a
feature needs, rebuild it. Full freedom.

## Owner's diagnosis (what is NOT world-class today)

1. **Topics are filenames.** `['301 3 Excel', 'Trees Part1 ...']` — extracted from file names,
   not content. Topics key the ENTIRE loop (quiz targeting, mastery, readiness gaps, concept
   graph, planner), so the loop is built on garbage strings.
2. **68s to start a drill.** Single blocking LLM call generates all N questions before the
   session starts. World-class = first question in seconds.
3. **The mastery loop is half-dead.** Streak/questions/topics-studied read 0 despite real
   activity — the write path is broken somewhere; the moat never becomes visible.
4. **MCQ-only drills = recognition, not recall.** No confidence calibration, no free recall,
   no interleaving. Pedagogically thin.
5. **The app is not opinionated.** It never says "here's what to do today." World-class study
   tools lead with a plan, not a menu.

## The five workstreams

### A. Course Brain (foundation — everything keys on this)
- New `course_topics` table (migration 0012): id, course_id, name (clean, human), slug,
  description, source documents + page ranges, prerequisite topic ids, ordering.
- Ingest gains an LLM synthesis pass: after chunks are embedded, sample/summarize content →
  produce 8–15 well-named topics w/ descriptions + prereq edges + doc/page coverage.
  Runs automatically post-upload/import AND via a backfill endpoint for existing courses.
- Replace every filename-topic producer/consumer with Course Brain topics. Mastery keys move
  to topic slug (with legacy-string migration mapping where feasible).
- Concept graph reads prereq edges from course_topics (no more ad-hoc LLM graph builds).

### B. Instant-start drills
- `/quiz/generate` two-phase: phase 1 returns a session with the first 3 questions
  (~one small LLM call); FastAPI BackgroundTasks generates the remainder into
  quiz_questions; new `GET /quiz/{quiz_id}/questions?after=N` lets the client pull
  the rest. Frontend starts the session the moment phase 1 lands; fetches ahead in
  background; graceful "writing more questions…" state only if the user outruns it.

### C. Close the mastery loop (visibly)
- Fix the broken write path (recon identifies it) so quiz/exam submits update
  learning_progress/mastery + streak + interactions.
- Quiz/exam results show per-topic mastery deltas ("Hashing 42% → 55%").
- CourseHome readiness/streak live; Progress numbers real.

### D. Opinionated guidance
- **Today panel** leads CourseHome: assembled from due reviews (review_engine), weakest
  Course Brain topic (readiness), upcoming exam countdown (planner/canvas dates) — a
  checklist with time estimates, one primary "Start" action.
- **Course Brief** on Materials post-ingest: "What Vindexa understood" — topic list with
  descriptions + coverage (docs/pages per topic) + suggested first actions. This is the
  trust-building magic moment after upload.

### E. Quiz pedagogy v3
- Confidence tap (Sure / Think so / Guessing) before reveal on each item; results include a
  calibration read-out (confident-wrong = priority reviews, seeded accordingly).
- Explanations always cite (footnote-ref chips to doc+page).
- Whole-course drills interleave toward weak topics (weighted sampling by mastery).
- Optional free-recall items (short answer, AI-graded) mixed in at Hard difficulty.

## Layout mandates (not restyles — new structures)
- **CourseHome → command center**: Today panel (left, checklist w/ Start CTA) + right rail
  (readiness ring, streak, exam countdown) + topic mastery grid (each Course Brain topic a
  row: mastery bar + inline Drill / Review / Ask actions). Dense, tool-like.
- **Quiz results → debrief**: score + calibration + per-topic deltas + mistake list w/
  citations + one recommended next action.
- **Materials → two-column**: files/upload left, Course Brief right.
- **Progress**: topic taxonomy view keyed on Course Brain (prereq edges), real numbers.
- **Dashboard**: cross-course due-reviews strip + streak; courses grid stays.
- Learn/Chat interaction is the approved gold standard — light touch only.

## Sequencing
1. Recon (done → findings appended below).
2. Backend A: Course Brain (migration + engine + rewiring + tests). Apply migration to live
   Supabase via SQL editor (see memory: reference-supabase-migrations-via-chrome).
3. Backend B: fast-start drills + mastery-loop fix + calibration fields (+ tests).
4. Frontend wave (parallel, disjoint): CourseHome command center; Quiz session+debrief;
   Materials Course Brief; Progress taxonomy + Dashboard strip.
5. Live E2E as a user; gates; PR; merge on green.

## Invariants
- HTTP contract stays backward compatible where the frontend depends on it (add, don't break).
- All suites stay green (backend 309+, FE vitest + Playwright).
- Paper & Ink design system is the visual law (CLAUDE.md); new layouts express it.
- No fake progress states; honest staged copy everywhere.
