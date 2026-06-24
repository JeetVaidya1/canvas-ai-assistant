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

## Styling Conventions — "Premium Dark" design system (Linear/Vercel)

The canonical token + utility source is `src/index.css` (read it first). Summary:

- **Canvas**: deep navy `#0a0c14` with a soft blue aurora on `body::before` (echoes the landing nebula). The app layout is transparent so the aurora shows — do NOT add opaque `bg-zinc-950` wrappers.
- **Elevation ladder (surfaces)**: cards are SOLID navy `#111521` (via the `Card` component / `.card-surface`, NOT translucent). Elevated/nested surfaces use `bg-white/[0.04]` or `#19202f`; popovers/overlays `#1f2738`. Use `.elev-1/2/3` for layered-shadow depth.
- **Borders**: `border-white/10`, `border-[#18181d]`, or `#21212a`.
- **Accent**: CYAN → BLUE (matches the landing wordmark + CTA). Primary buttons use `.bg-gradient-brand` (cyan→blue, `#06b6d4`→`#3b82f6`). Accent text/icons `text-cyan-300`; active rings `ring-cyan-400/25`; gradient eyebrow text `.text-gradient-brand`. (Quick Quiz keeps an amber ⚡ identity; semantic = emerald/amber/rose.)
- **Text contrast (important)**: titles/key content `text-zinc-50`/`text-zinc-100`; body `text-zinc-300`; secondary `text-zinc-400`; ONLY true hints `text-zinc-500`. Never use zinc-500/600 for primary or content text — that was the old "washed out" bug this rebuild fixed.
- **Semantic colors**: success = emerald, warning = amber, danger = rose (use rose, not red).
- **Font**: Inter via Google Fonts CDN. Headings use tight tracking (`-0.02em`).
- **Border radius**: `rounded-lg`/`rounded-xl`.
- **Primitives** (reuse, don't re-roll): `Button` + `Card`/`PageHeader` (`@/components/ui/Card`), `SubTabs`, `Select`, `Markdown`, `CountUp`, `FadeIn`.
- **Decorative effects ARE used, tastefully**: gradient accents, subtle glow on primary actions, glass top bar (`.glass-bar`), motion v12 entrances. (This supersedes the old "no decorative effects" rule.)
- **Wrapper page headers**: consolidated destinations (Learn/Practice/StudyKit/Progress) use a slim `h-14 border-b border-[#18181d]` bar with `SubTabs` — no redundant eyebrow+course-title block (the TopBar already shows page context).

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

World-class, deep-navy UI that matches the landing page: navy canvas with a soft blue aurora, a real elevation ladder, crisp high-contrast text, and a cyan→blue accent. The core study pages are STRUCTURALLY rebuilt, not restyled — Chat is the reference (center-first composer that docks on send, history as a slide-over, clean prose answers, collapsible sources, follow-up study chips); Notes is a center-first studio (topic prompt + style pills + collapsible sources, not a checkbox form); Practice setup is center-first and tactile. Generous spacing, confident type, subtle motion. Every surface should highlight what the backend can do (RAG citations, adaptive practice, concept graphs, readiness). Avoid flat low-contrast "AI slop" AND avoid recolor-only "rebuilds" — change layout + interaction, not just the palette.
