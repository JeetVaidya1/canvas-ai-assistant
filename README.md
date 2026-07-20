# Canvas AI Assistant

AI study assistant for Canvas LMS courses. Ingests course materials (PDF, PPTX, DOCX, images), builds a per-course knowledge base, and generates grounded quizzes, practice problems, notes, flashcards, and full exam simulations — every answer cites the source material it came from.

**Stack:** FastAPI + Supabase (auth, Postgres) + FAISS vector stores + React/Vite frontend, deployed on Fly.io.

## What it does

- **Multimodal ingestion** — text extraction (pdfplumber/PyMuPDF), slide and document parsing (python-pptx, python-docx), image OCR (Tesseract), and vision-model descriptions of figures and diagrams, with contextual chunking before embedding.
- **Grounded generation** — retrieval-augmented answers, quizzes, and practice problems that stay inside what the course actually taught, with citations back to source chunks.
- **Learning engine** — content-derived topic maps per course ("Course Brain"), two-phase drills, answer calibration, and mastery tracking.
- **Multi-tenant** — Supabase JWT auth with per-course access checks (`require_course_access`) enforced across all 19 routers; rate limiting on every endpoint.

## Engineering

- **246 backend tests** across 30 files (pytest, hermetic — no live keys needed) plus frontend tests (Vitest) and Playwright E2E, gated by **GitHub Actions CI on every push**.
- Layered backend: 19 focused routers over engine/service modules, typed with Pydantic throughout; SQL migrations under `migrations/`.
- React/Vite/Tailwind frontend (~18K LOC) with role-aware routing.

## Running it

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest                     # backend suite, no external services required
uvicorn main:app --reload  # needs SUPABASE_* and ANTHROPIC_API_KEY env vars
```

Course materials are ingested from a local `data/<course>/` directory which is **gitignored** — course content is copyrighted by instructors and never belongs in the repo.

## Status

Active. Originally a single-user prototype; rebuilt into the multi-tenant, CI-gated application described above.
