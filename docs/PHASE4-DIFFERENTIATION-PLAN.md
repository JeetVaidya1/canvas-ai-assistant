# Phase 4 — Differentiation: the closed mastery loop + interop layer

Goal: stop being a wrapper. Two theses, each its own set of PRs.

**Thesis 1 — Own the closed loop.** The pieces from Phase 3 (mastery analytics,
quiz, practice, flashcard SR, exam past-paper analysis, planner) are silos. Wire
them into one loop that no single LLM call can replicate:

```
wrong answer (quiz/exam)
  -> pinpoint the concept (already tagged per question)
  -> explain the mistake, cited to the exact source page
  -> seed a per-user spaced-repetition review item for that concept
  -> planner schedules it against the exam date
  -> re-test; mastery only rises when it survives spaced reviews
  -> recompute predicted exam readiness
```

**Thesis 2 — Be the study context layer.** Vindexa owns the grounded course +
per-student mastery; other tools (Anki, GitHub, other AIs) are endpoints that
plug in. Inverts the wrapper narrative.

Shared foundations unchanged: `providers.structured_call`, `rag.retrieval.retrieve`,
`learning_analytics`, `@/components/ui/Markdown`. DB changes go to `schema.sql`
**and** a `migrations/000N_*.sql` applied to the live Vindexa1 DB.

---

## Closed loop

### A. Exam-readiness score  (M)
`readiness_engine.py` + `GET /api/readiness/{course_id}/{user_id}` (optional
`exam_id`). Combine per-concept mastery (`learning_progress`) with past-paper
topic weighting (`exam_generator` past-paper analyses) → `{score_pct, by_topic,
gaps[], confidence}`. Surface as a headline number on the course overview +
analytics ("72% predicted; gap = Hashing, DAGs").

### B. Mistake-driven review queue  (L)
New `review_items` table (per-user SM-2 queue seeded by mistakes):
`(id, user_id, course_id, concept, prompt, answer, explanation, source,
ease, interval, repetitions, due_date, last_reviewed, status, created_at)`.
`review_engine.py` reuses SM-2 from `flashcard_engine`. On every wrong quiz/exam
answer, auto-seed a review item (due today). Endpoints: `GET /api/reviews/{course_id}`
(due first), `POST /api/reviews/{item_id}/grade`. Frontend "Review" surface
showing due items across the course. This is the spine of the loop.

### C. Grounded "explain my mistake"  (M)
When an answer is wrong, retrieve the contradicting passage and return a short,
cited explanation of *why* the student's answer is wrong, anchored to their own
materials. Extend quiz `grade_answer` and exam grading to attach `mistake_explanation`
+ `source`. Show it in the feedback card.

### D. Concept prerequisite graph  (L)
`concept_graph.py`: extract the course's concepts + prerequisite edges via
`structured_call` over retrieved material; persist `concept_graph` (course-scoped).
Use it in analytics/readiness: "you're blocked on X because prerequisite Y is weak."

### E. Dynamic planner  (M)
Planner re-prioritizes from *current* state: weak concepts (mastery), readiness
gaps, due `review_items`, and prerequisite order — packed against the exam date.
Add `POST /api/replan` and a "weak-areas-first" plan mode.

---

## Interop / "context layer"

### F. Anki export with SR state  (S)
Extend `exports.build_flashcards_apkg` to export the *persisted* deck preserving
SM-2 state (due dates / intervals) so power users keep progress. Add review-item
export too.

### G. GitHub export / import  (M)
Export notes + decks as versioned Markdown (downloadable bundle; optional push to
a repo given a token). Import course materials from a GitHub repo of lecture notes.

### H. Export to AIs  (L)
The anti-wrapper play:
- **MCP server per course** (`mcp_server.py`, stdio): tools to query a course's
  grounded content + the student's mastery/weak topics, so Claude/Cursor can use
  Vindexa as a study memory.
- **Context-pack export** `GET /api/context-pack/{course_id}/{user_id}`: a
  ready-to-paste prompt + curated excerpts of weak areas for any model / Claude
  Project / Custom GPT.

### I. Canvas LMS deepening  (M)
Extend the existing `importCanvasLms` to pull syllabus + assignment/exam due
dates + materials, auto-feeding exam dates into the planner (kills cold-start).

---

## Order
A → B → C (the loop's spine and headline value) → D → E → F → G → H → I.
Each is its own branch off `main`, verified live, merged. Migrations applied to
the live DB via the Supabase SQL editor (Claude-in-Chrome).
