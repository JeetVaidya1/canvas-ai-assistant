# Vindexa — AI Study Platform Frontend

## Project Overview

Vindexa is an AI-powered study companion that transforms course materials into intelligent learning tools. Users upload course files (PDF, DOCX, PPTX) and access AI-generated features: chat, quizzes, practice problems, notes, flashcards, exams, analytics, study planner, and audio overviews.

## Tech Stack

- **Framework**: React 19 + TypeScript 5.8 (strict mode)
- **Build**: Vite 7
- **Styling**: Tailwind CSS v4 (uses `@theme` directive in CSS, NOT `tailwind.config.js`)
- **Routing**: react-router-dom v7
- **Data Fetching**: @tanstack/react-query v5
- **Animation**: motion v12 (framer-motion rebranded, imports from `motion/react`)
- **Icons**: lucide-react
- **Backend**: Supabase (@supabase/supabase-js)

## Key Architecture

- **Path alias**: `@/` maps to `src/` (configured in both `vite.config.ts` and `tsconfig.app.json`)
- **Routing**: `src/App.tsx` defines all routes with React.lazy + Suspense code splitting
- **Layout**: `src/components/layout/AppLayout.tsx` wraps all pages with sidebar + topbar
- **Pages**: `src/pages/` — 11 page components, each lazy-loaded
- **Feature Components**: `src/components/` — FlashCards, QuizAsisstant (note: typo in filename is intentional), PracticeMode, NotesCreator, ExamMode, AnalyticsDashboard
- **UI Components**: `src/components/ui/` — reusable visual components
- **Shared Components**: `src/components/shared/` — LoadingSpinner, EmptyState, ConfirmDialog, ErrorBoundary
- **Hooks**: `src/hooks/` — useUser, useCourses, useCourseFiles, useChatSessions (TanStack Query wrappers)
- **API**: `src/lib/api.ts` — all backend API calls

## TypeScript Constraints (CRITICAL)

- `verbatimModuleSyntax` is ON: must use `import type { ... }` for type-only imports
- `noUnusedLocals` is ON: every import must be used or build fails
- `noUnusedParameters` is ON: every function parameter must be used
- Build command: `tsc -b && vite build` — TypeScript must pass before Vite builds

## Styling Conventions — "PAPER & INK" design system (editorial / textbook)

The canonical token + utility source is `src/index.css` (read it first). This REPLACED the old dark navy+cyan system entirely — any `zinc-*`, `cyan-*`, `bg-gradient-brand`, glow, or glass class is LEGACY and must be migrated on sight.

- **Canvas**: warm paper `#f7f5f1` (`bg-paper`); sidebar/wells `bg-paper-deep`; cards are WHITE sheets (`Card` / `.card-surface`: `bg-surface` + `border-line` hairline + faint shadow). Light-first; there is no dark mode.
- **Text ladder**: `text-ink` (primary) / `text-ink-soft` (secondary) / `text-ink-faint` (hints only).
- **Accent — ONE pen blue**: `bg-accent` `#2b4acb` for primary actions; `text-accent` / `text-accent-deep` for links & active states; `bg-accent-wash` + `border-accent-line` for tinted chips. NEVER gradients, NEVER glow, NEVER glass/backdrop-blur.
- **Typography**: UI = Instrument Sans (default). Display = Newsreader serif via `.font-display` — page titles (PageHeader does this), hero numbers, brand moments. Mono = JetBrains Mono (`.section-num`, timers with `.tnum`).
- **Signature motifs (use them — they carry the identity):**
  - `.hl` — highlighter-yellow mark behind a key stat/phrase (1-2 per page max)
  - `.section-head` + `.section-num` — numbered syllabus section headers ("01 — Learn")
  - `.footnote-ref` — footnote-style citation chips for sources
  - `Badge tone="marker"` — highlighter-toned label
- **Semantic (muted, ink-like)**: success `#2f7d5c`, warning `#a8741a`, danger `#bb4444`, info `#3d6a8f`, each with a `-wash` bg token. Use `scoreTone()` from `@/lib/score` for readiness/mastery.
- **Charts (recharts)**: ink `#211f1a` + accent `#2b4acb` series; gridlines `#e7e3d9`; tick text `#8d877b` 11px; tooltip = white card (`#ffffff` bg, `#e7e3d9` border, ink text); one subtle accent area fill max.
- **Primitives (reuse, never re-roll)**: `Button` (primary/secondary/ghost/danger — flat, no ripple), `Card`/`PageHeader`, `SubTabs`, `Select`, `Badge`, `Input`/`Textarea`, `Modal`, `Tooltip`, `ProgressBar`/`ProgressRing`, `EmptyState`/`ErrorState`, `Markdown`, `BrandMark` (serif V on pen-blue square).
- **Motion**: entrances ≤250ms (`.animate-fade-up`), no stagger chains, no whileInView theatrics. Indeterminate progress via `.animate-indeterminate` — never fake percentages.
- **Focus**: `.focus-ring` (pen-blue on paper).

## File Structure

```
src/
  App.tsx              # Router with lazy-loaded routes
  main.tsx             # Entry point with providers (QueryClient, BrowserRouter)
  index.css            # Tailwind + theme variables + base styles
  lib/
    api.ts             # Backend API functions
    supabaseClient.ts  # Supabase client
  hooks/
    useUser.ts         # User ID from localStorage
    useCourses.ts      # Course CRUD via TanStack Query
    useCourseFiles.ts  # File operations via TanStack Query
    useChatSessions.ts # Chat session management
  components/
    layout/
      AppLayout.tsx    # Main layout shell (sidebar + topbar + outlet)
      AppSidebar.tsx   # Collapsible sidebar with course nav
      TopBar.tsx       # Page title + breadcrumbs
    shared/
      LoadingSpinner.tsx
      EmptyState.tsx
      ConfirmDialog.tsx
      ErrorBoundary.tsx
    ui/
      FadeIn.tsx       # Intersection Observer scroll animation
      CountUp.tsx      # Animated number counter
    FlashCards.tsx
    QuizAsisstant.tsx  # (typo is intentional, don't rename)
    PracticeMode.tsx
    NotesCreator.tsx
    ExamMode.tsx
    AnalyticsDashboard.tsx
  pages/
    Dashboard.tsx
    CourseOverview.tsx
    ChatPage.tsx
    QuizPage.tsx
    PracticePage.tsx
    NotesPage.tsx
    ExamsPage.tsx
    AnalyticsPage.tsx
    PlannerPage.tsx
    AudioPage.tsx
    SettingsPage.tsx
```

## Commands

- `npm run dev` — Start dev server (Vite, port 5173)
- `npm run build` — TypeScript check + Vite production build
- `npm run lint` — ESLint
- `npm run preview` — Preview production build

## Design Philosophy

Paper & Ink: a digital study desk — warm paper, ink text, one pen-blue accent, a highlighter. Editorial/textbook character (serif display, numbered sections, footnote citations), density over decoration, zero gradients/glow/glass. Every surface still showcases the backend (RAG citations, adaptive practice, readiness).
