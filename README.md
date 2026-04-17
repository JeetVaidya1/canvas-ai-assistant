# Canvas AI Assistant

Personal AI study assistant for Canvas LMS. Pulls your course materials, builds a per-course vector index, and generates quizzes, practice problems, notes, and full exam simulations grounded in what your professor actually taught.

## The problem

Canvas scatters your course content across modules, weeks, announcements, and file dumps. Studying for a final means rebuilding context from scratch every session. Generic LLMs don't know your syllabus, and the paid study tools that do usually cost $20/month and still hallucinate half the answers.

## How it works

```
Canvas course pages
       ↓
Ingest (chunk + embed PDFs, DOCX, PPTX, images)
       ↓
FAISS vector store (scoped per course)
       ↓
 ┌─────────────┬─────────────┬──────────────┬──────────┐
 │ Conversational│ Exam gen   │ Notes engine │ Analytics│
 │  RAG engine   │             │              │          │
 └───────────────┴─────────────┴──────────────┴──────────┘
```

Every retrieval is scoped to the current course. The exam generator pulls from the same index every other feature uses, so every practice question is traceable back to the chunk it came from.

## Features

- **Conversational RAG** — ask questions in natural language, get answers grounded in your own materials with citations
- **Quiz generator** — timed quizzes per topic, adjustable difficulty
- **Practice problems** — longer-form questions with step-by-step worked solutions
- **Notes engine** — condenses lectures into studyable summaries
- **Exam simulator** — full-length practice exams with timing and scoring
- **Learning analytics** — tracks which topics you keep getting wrong and biases future questions toward them

## Stack

- Python + FastAPI
- OpenAI embeddings and GPT-4 for generation
- FAISS for vector search
- pdfplumber / PyMuPDF / python-pptx / python-docx / Tesseract OCR for ingestion
- TypeScript frontend

## Status

Prototype. Works end-to-end on my own courses. Not multi-tenant yet — everything assumes one user, one Canvas account.
