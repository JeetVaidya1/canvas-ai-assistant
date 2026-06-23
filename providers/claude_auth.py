# providers/claude_auth.py
"""
Resolve Anthropic credentials.

Two modes:
  - "api_key": a normal Anthropic API key (sk-ant-api...). Standard billing.
  - "oauth":   a Claude Code / Max subscription OAuth token (sk-ant-oat...),
               read from the macOS keychain. Lets the app run on the Max
               subscription instead of API credits.

OAuth mode requires the Claude-Code identity system prompt and the oauth beta
header (handled in anthropic_chat.py).
"""
from __future__ import annotations

import json
import os
import subprocess
from typing import Tuple

KEYCHAIN_SERVICE = "Claude Code-credentials"


def _read_keychain_token() -> str | None:
    """Read the Claude Code OAuth access token from the macOS keychain."""
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-w"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    return (data.get("claudeAiOauth") or {}).get("accessToken")


def resolve_auth() -> Tuple[str, str]:
    """Return (mode, secret) where mode is 'api_key' or 'oauth'.

    Precedence: a real API key wins; otherwise an explicit oauth env var;
    otherwise the Claude Code keychain token (read fresh so Claude Code's
    background refresh is always picked up).
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key and not api_key.startswith("sk-ant-oat"):
        return ("api_key", api_key)

    env_oauth = os.getenv("CLAUDE_CODE_OAUTH_TOKEN")
    if env_oauth:
        return ("oauth", env_oauth)
    if api_key and api_key.startswith("sk-ant-oat"):
        return ("oauth", api_key)

    token = _read_keychain_token()
    if token:
        return ("oauth", token)

    raise RuntimeError(
        "No Anthropic auth available. Set ANTHROPIC_API_KEY, or sign into "
        "Claude Code so the Max OAuth token is in your keychain."
    )
