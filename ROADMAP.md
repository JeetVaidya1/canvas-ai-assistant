# Vindexa — Path to Production Launch

**Goal:** Take Vindexa from a working-but-rough prototype to a hardened, monetized,
polished product safe to put real (paying) students on.

**Decisions (locked):**
- **Monetization:** single paid tier, **~$15–20/mo**, with a **free trial** (no free-forever tier).
- **Sequencing:** **foundation first** — clean + harden the code before adding billing and polish.
- **Hosting:** Vercel (frontend) + Fly.io (backend container, already configured). Real
  `sk-ant-api...` key in prod; Max OAuth only for local dev.

**Starting state (measured):**
- Backend ~12.8k LOC / ~50 files. God-files: `practice_generator.py` (980),
  `exam_generator.py` (843), `exam_session_manager.py` (677), `routers/exams.py` (505),
  `notes_engine.py` (503). ~10 dead/duplicate files incl. 3–4 overlapping RAG/query engines.
  No tests.
- Frontend ~9.4k LOC. Good infra already: `motion` (Framer v12), Tailwind v4, shadcn-style
  `components/ui`, react-query, react-router 7. God-file: `api.ts` (1033). No Stripe.

---

## Phase 0 — Safety net  *(prerequisite for everything else)*
You cannot refactor 12.8k LOC safely with zero tests. Build the net first.
- pytest harness + smoke/integration tests on the critical paths (auth, ingest, query,
  quiz/exam generation, sharing). Target the high-traffic engines first, not 100% coverage.
- GitHub Actions CI to run them on every push.
- **Exit:** green test suite that fails loudly if a refactor breaks a core flow.

## Phase 1 — Backend refactor
- Delete confirmed dead code: `debug_live_api.py`, `test_gpt4v.py`, `check_endpoints.py`,
  `reingest.py`, `enhanced_ingest.py`, and whichever of the duplicate engines are unused.
- Consolidate the overlapping RAG/query engines (`query_engine`, `enhanced_query_engine`,
  `conversational_rag_engine`, `quiz_assistant_engine`) into one.
- Split god-files into focused <400-line modules.
- Kill the `from deps import *` anti-pattern; explicit imports.
- **Exit:** no file >400 lines, no dead code, tests still green.

## Phase 2 — Frontend refactor
- Split `api.ts` (1033) into per-domain API modules.
- Decompose big components (ExamMode 806, NotesCreator 785, QuizMode 594, PracticeMode 521)
  into subcomponents + hooks.
- Consistent patterns, shared UI primitives, typed API layer.
- **Exit:** `tsc -b && vite build` clean; no component >300 lines.

## Phase 3 — Cost controls  *(must land before public trials)*
- **Per-user rate limiting** on all AI endpoints (sliding window keyed by user id), with
  separate trial vs paid limits. Trial is the main abuse surface.
- **Caching to cut API spend:**
  - Anthropic **prompt caching** for repeated system prompts + course context (biggest lever).
  - Content-hash result cache for deterministic generations (same params → cached output).
  - Verify embedding cache is content-hashed (no re-embedding unchanged docs).
- **Model routing audit:** Haiku-first, Sonnet only where it measurably helps.
- **Cost observability:** per-user / per-request token + $ tracking (revive `cost_analysis.py`).
- **Exit:** measured per-active-user cost ceiling that fits a $15–20 price with margin.

## Phase 4 — Hardening ("flawless, no lag")
- Error handling: consistent error envelope, no silent swallows, friendly UI messages.
- Input validation: Pydantic models on every endpoint boundary.
- Performance: stream AI responses; kill N+1 Supabase queries; paginate; async blocking calls.
- Frontend perf: route code-splitting, lazy loading, memoization, skeleton/empty/error states.
- Security: Stripe webhook signature verify, rate-limit bypass checks, re-confirm RLS.
- **Exit:** load-sane, no obvious lag, clean error paths under failure injection.

## Phase 5 — Monetization (Stripe)
- Stripe Checkout subscription with a free trial; Customer Portal for manage/cancel.
- Webhooks (`checkout.session.completed`, `customer.subscription.updated/deleted`) →
  subscription status in Supabase, signature-verified + idempotent.
- Backend gating middleware: trial/active → allow; expired → 402 with upgrade prompt.
- Frontend: pricing page, trial banner/countdown, paywall, manage-subscription UI.
- **Exit:** real test-mode purchase → unlocks; cancel → re-locks at period end.

## Phase 6 — Landing page + visual polish
- Upgrade `LandingPage.tsx`: hero, features, social proof, pricing, CTA, `motion` animations.
- Component polish pass: transitions, consistent green design tokens, responsive, empty/loading/error states.
- **Exit:** looks like a finished, trustworthy product on desktop + mobile.

## Phase 7 — Preview mode + videos + help
- **Preview/demo mode:** sandboxed read-only walkthrough on a sample course, no sign-up (funnel).
- **Preview videos:** short product walkthroughs embedded on landing + help.
- **Help:** in-app help tab / docs, onboarding tour, tooltips.
- **Exit:** a cold visitor can understand and try the product without an account.

## Phase 8 — Launch
- Lock CORS to the Vercel origin; finalize secrets.
- Complete the 4 Supabase dashboard steps (Google OAuth, redirect URLs, SMTP, claim-legacy).
- Full E2E smoke on Fly + Vercel previews.
- Cut over to production.

---

### Notes
- Each phase ships as its own PR(s); nothing proceeds on a red test suite.
- Trial-window cost is the main financial risk → Phase 3 gates Phase 5/8.
- Re-validate retrieval quality after any change touching the RAG/embedding path.
