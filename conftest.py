"""Global pytest setup.

CRITICAL: this runs before any test module imports app code. Nearly every engine
calls ``create_client(SUPABASE_URL, SUPABASE_KEY)`` at *module import* time, so we
inject dummy credentials here. ``create_client`` only constructs the client object
(no network call happens until a query executes), so dummy values are safe and keep
the suite hermetic — it never touches the real Supabase project.

``setdefault`` + ``load_dotenv(override=False)`` (the default used in the engines)
means these dummy values win over anything in a local ``.env``, so tests can never
accidentally hit production credentials.
"""
import os

os.environ.setdefault("SUPABASE_URL", "https://fake-project.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "fake-service-role-key")
os.environ.setdefault("SUPABASE_ANON_KEY", "sb_publishable_fake")
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-api-fake")
os.environ.setdefault("MODEL_DEFAULT", "claude-haiku-4-5-20251001")
os.environ.setdefault("MODEL_COMPLEX", "claude-sonnet-4-6")
