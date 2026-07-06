# V4 — "Professional" (Jeet's mandate, 2026-07-06)

Feedback driving this: app feels laggy; doesn't feel like a full professional website;
shell still reads as the original sidebar app; AI-giveaway icons remain; nothing is
resumable. Decisions made by Jeet (AskUserQuestion):
- Shell → **Command workspace** (Linear-style): slim top bar (brand, course switcher
  dropdown, Cmd+K palette + search, readiness chip, avatar menu) + dense ICON RAIL
  left nav. The wide sidebar dies.
- Perf → **instant paint first** (persist React Query cache; render last-known data
  immediately, refresh quietly), then backend precompute of slow endpoints.
- Site surfaces → **Resume everywhere**, **onboarding tour + empty-state polish**,
  **marketing site depth** (pricing/FAQ/footer/terms/privacy/help).

## Workstreams
A. SHELL: components/shell/ — TopBar (course switcher, Cmd+K trigger, readiness chip,
   avatar menu incl. sign out/settings), IconRail (6 destinations + dashboard, tooltips,
   active states), CommandPalette (Cmd+K: jump to course/destination, actions: start
   drill / new note / ask, recent items; client-side index). AppLayout rebuilt on these.
B. INSTANT PAINT: React Query cache persisted to localStorage (persist-client +
   sync storage persister, 24h maxAge, buster = app build id). Skeletons only on
   cold first visits. (Phase 2 later: precompute readiness/analytics server-side.)
C. RESUME EVERYWHERE: backend GET /api/quiz/in-progress (per user+course, from
   quiz_sessions + responses); useQuizRun server-restore; notes drafts + practice
   session snapshot in localStorage; "Continue" cards (quiz N of M, exam, note draft)
   on Dashboard + CourseHome Today panel. Exams already resume.
D. AI-ICON SWEEP: kill Sparkles/Brain/Zap/Wand/lightning/graduation-tile iconography
   app-wide; replace with mono glyphs, typographic markers, or nothing.
E. ONBOARDING + EMPTY STATES: dismissible first-run coach marks (localStorage),
   every empty state re-checked for a concrete next action.
F. MARKETING DEPTH: /pricing (single tier ~$15/mo + free trial per ROADMAP), FAQ on
   landing, real footer (Product/Company/Legal), /terms + /privacy (standard SaaS
   templates marked DRAFT for review), /help (getting-started docs). Public routes.

Sequencing: A+B+C parallel (disjoint), then D+E+F (touch shell/landing files).
Invariants: Paper & Ink law; all gates green; PR + merge on green (standing auth).
