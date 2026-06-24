# Deploying the Vindexa backend to Fly.io

The frontend goes on Vercel; this is the **backend container** (FastAPI + local
BGE embeddings). Vectors, auth, and data already live in Supabase, so this host
is stateless — it can be destroyed and recreated freely.

## Files in this setup
- `Dockerfile` — Python 3.12 + poppler/tesseract, CPU-only torch, BGE model baked in.
- `.dockerignore` — keeps the frontend, `data/`, and `vectorstores/` out of the image.
- `fly.toml` — single always-on 2GB machine, health check on `/system-status`.

## One-time setup

```bash
# 1. Authenticate (opens a browser)
fly auth login

# 2. Create the app WITHOUT deploying yet (so we can set secrets first).
#    If "vindexa-backend" is taken, pick another name and update fly.toml.
fly apps create vindexa-backend
```

## Secrets (never baked into the image)

Pull the values from your local `.env`. The backend needs these four; the
`MODEL_*` and `CLAUDE_MAX_TOKENS` lines are optional overrides.

```bash
fly secrets set \
  ANTHROPIC_API_KEY="sk-ant-api..." \
  SUPABASE_URL="https://eddozjbdezpdcwuxuzpo.supabase.co" \
  SUPABASE_KEY="<service-role-key>" \
  SUPABASE_ANON_KEY="sb_publishable_..." \
  MODEL_DEFAULT="claude-haiku-4-5-20251001" \
  MODEL_COMPLEX="claude-sonnet-4-6"
```

> **Use a real `sk-ant-api...` key**, not the Max OAuth token — there's no
> macOS keychain on the server, and `resolve_auth()` gives the API key top
> precedence. This is the metered, per-student cost.

## Deploy

```bash
fly deploy
```

First build is slow (downloads torch + bakes the 1.3GB model into the image).
Subsequent deploys reuse cached layers.

## Verify

```bash
fly status                      # machine should be "started" and passing checks
curl https://vindexa-backend.fly.dev/system-status   # → JSON capabilities blob
fly logs                        # watch for OOM kills (see memory note below)
```

## Then point the frontend at it
In the Vercel project's env vars set:
```
VITE_API_BASE_URL = https://vindexa-backend.fly.dev
```
(plus the existing `VITE_SUPABASE_URL` / `VITE_SUPABASE_KEY`) and redeploy.

## Notes / gotchas
- **Memory:** 2GB is the floor for BGE-large. If `fly logs` shows `Out of memory`
  / `OOM`, bump `memory` to `"4gb"` in `fly.toml` and redeploy.
- **First request is slow:** the model loads lazily on the first embedding call
  (~20–40s), then stays warm because the machine never scales to zero.
- **Region:** `fly.toml` defaults to `iad`. Set it to whatever is closest to the
  Supabase project to cut DB round-trips.
- **CORS** is currently `allow_origins=["*"]` — fine for a private preview. Lock
  it to the Vercel origin before real production.
