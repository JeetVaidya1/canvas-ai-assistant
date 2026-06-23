# Phase 6 — Production multi-user: auth, ownership, and database security

Turn the prototype (identity = a spoofable `user_id` form field; courses globally
visible) into a real multi-user system where **users only ever see their own
stuff**, enforced in depth.

## Decisions (locked with the founder)
- **Auth:** Supabase Auth — email+password, magic link, and Google OAuth.
- **Existing data:** assigned to the first account that signs up (claim-on-first-login).
- **Security model:** RLS **and** backend JWT verification (defense-in-depth).

## Architecture
- **Identity** = the Supabase Auth user (`auth.users.id`, a UUID). The client-supplied
  `user_id` is removed everywhere — it was the core hole.
- **Frontend** holds a Supabase session (supabase-js). Every backend call sends
  `Authorization: Bearer <access_token>`. The only direct DB read from the client is
  `courses` (RLS scopes it). Everything else is backend-mediated.
- **Backend** verifies the token on every request via GoTrue (`auth.get_user`) and
  derives identity from it. It uses the service-role key (bypasses RLS) but **scopes
  every query** by the authenticated id / course access. RLS is the second wall.
- **Ownership:** `courses.owner_id`. "My courses" = owned **or** joined (via
  `course_memberships`, already built for shared courses). Course-scoped data
  (notes, flashcards, quizzes, embeddings…) is reachable only with course access.

## PRs
### M1 — Backend auth core
- `auth.py`: `get_current_user` dependency (Bearer → `auth.get_user` → {id, email},
  60s cache); `require_course_access(course_id, user)` (owner or member). `/api/me`.
- Add `SUPABASE_ANON_KEY` to backend env (the publishable key) for token verification.
- Verify by minting a real test user via the admin API.

### M2 — Frontend auth
- `AuthProvider` + `useAuth()` (session via `onAuthStateChange`). Repoint `useUser()`
  to the real auth id so the 11 call sites don't churn.
- `LoginPage`: sign up / sign in (email+pw), magic link, Google button; "check your
  email" states.
- `apiFetch` attaches the bearer token; 401 → sign out.
- `RequireAuth` guard on the app shell; Landing + Login stay public. Logout control.

### M3 — Identity enforcement + ownership
- Migration 0009: `courses.owner_id`.
- Replace all 20 `user_id` form params (11 routers) with `Depends(get_current_user)`;
  drop `user_id` from the matching `api.ts` calls. `/sessions` uses the authed id.
- `create-course` sets `owner_id`; delete/upload/file ops require course access.
- `/api/claim-legacy-data`: first authed user claims all `owner_id IS NULL` courses
  (idempotent). Frontend calls once post-login.

### M4 — RLS
- Migration 0010: enable RLS on every table + policies.
  - `courses`: owner-or-member for SELECT/UPDATE/DELETE; INSERT owner = auth.uid().
  - user-scoped tables: `user_id = auth.uid()::text`.
  - course-scoped tables: access via course ownership/membership subquery; `messages`
    via `chat_sessions`.
- Backend (service-role) is unaffected; the client's only direct read (`courses`)
  gets a SELECT policy so it returns just the user's courses.

### M5 — Isolation verification
- Two real auth users; prove A cannot read or mutate B's data via the API, and that
  the RLS-scoped course list returns only own courses.

## Manual steps for the founder (I'll guide)
- Supabase dashboard → Auth → Providers → enable Google (client id/secret) + redirect URLs.
- Auth → URL config: add the app origin (localhost:5173 + prod) to redirect allow-list.
- Email confirmation: default dev email works (rate-limited); production needs SMTP.
