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

## Styling Conventions

- **Dark theme**: zinc-950 (bg), zinc-900 (cards), zinc-800 (elevated/hover), zinc-700 (borders)
- **Accent**: cyan-600 for primary actions, cyan-400 for active states
- **Font**: Inter via Google Fonts CDN
- **Border radius**: `rounded-lg` consistently
- **Cards**: `bg-zinc-900 border border-zinc-800 rounded-lg`
- **No decorative effects**: no glow shadows, no glass-morphism, no animated backgrounds
- **Buttons**: solid colors, no gradients. Primary = `bg-cyan-600`, Secondary = `bg-zinc-800`

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

Professional, Linear/Notion-inspired dark UI. Clean, purposeful, information-dense. No decorative animations or flashy effects. Cyan accent used sparingly for primary actions and active states only.
