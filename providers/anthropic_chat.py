# providers/anthropic_chat.py
"""
Anthropic-backed drop-in for `client.chat.completions.create(...)`.

Accepts the OpenAI-style arguments the existing engines pass (messages,
temperature, max_tokens, response_format, extra_body, ...) and returns an
object exposing `.choices[0].message.content`, so call sites are unchanged.

Handles:
  - system-message extraction (Anthropic takes `system` separately)
  - vision parts ({type:"image_url"}) -> Anthropic image blocks
  - JSON mode (response_format={"type":"json_object"}) via instruction + cleanup
  - merging consecutive same-role messages (Anthropic requires alternation)
  - sane defaults (max_tokens is required by Anthropic; temperature clamped)
"""
from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .claude_auth import resolve_auth
from .model_map import resolve_chat_model

DEFAULT_MAX_TOKENS = int(os.getenv("CLAUDE_MAX_TOKENS", "4096"))

# When authenticating with a Claude Code / Max OAuth token, Anthropic requires
# the oauth beta header and a system prompt that begins with the Claude Code
# identity. Requests without it are rejected for subscription credentials.
_OAUTH_BETA = "oauth-2025-04-20"
_CLAUDE_CODE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."

_client = None
_auth_mode = None
_client_lock = threading.Lock()


def _get_anthropic():
    """Lazily construct the Anthropic SDK client once. Returns (client, mode)."""
    global _client, _auth_mode
    if _client is None:
        with _client_lock:
            if _client is None:
                import anthropic

                mode, secret = resolve_auth()
                if mode == "oauth":
                    _client = anthropic.Anthropic(
                        auth_token=secret,
                        default_headers={"anthropic-beta": _OAUTH_BETA},
                    )
                else:
                    _client = anthropic.Anthropic(api_key=secret)
                _auth_mode = mode
    return _client, _auth_mode


def _reset_client() -> None:
    """Drop the cached client so the next call re-reads a fresh token.

    The Max OAuth token in the keychain rotates (~hourly); without this, a
    long-running process keeps using a client built around a now-expired token
    and every call 401s until manual restart.
    """
    global _client, _auth_mode
    with _client_lock:
        _client = None
        _auth_mode = None


def _is_auth_error(exc: Exception) -> bool:
    if getattr(exc, "status_code", None) == 401:
        return True
    return "authentication" in str(exc).lower() and "401" in str(exc)


def _messages_create_with_retry(client, kwargs):
    """messages.create that, on a 401, refreshes the token once and retries."""
    try:
        return client.messages.create(**kwargs)
    except Exception as exc:  # noqa: BLE001
        if _is_auth_error(exc):
            _reset_client()
            fresh, _ = _get_anthropic()
            return fresh.messages.create(**kwargs)
        raise


# ---------- response shape (mimics OpenAI SDK) ----------
@dataclass(frozen=True)
class _Message:
    content: str
    role: str = "assistant"


@dataclass(frozen=True)
class _Choice:
    message: _Message
    finish_reason: str = "stop"
    index: int = 0


@dataclass(frozen=True)
class _ChatResponse:
    choices: List[_Choice]
    model: str


# ---------- translation helpers ----------
def _parse_data_url(url: str) -> Optional[Dict[str, str]]:
    """Turn 'data:image/png;base64,XXXX' into an Anthropic base64 image source."""
    match = re.match(r"^data:(?P<media>[\w/+.-]+);base64,(?P<data>.+)$", url, re.DOTALL)
    if not match:
        return None
    return {
        "type": "base64",
        "media_type": match.group("media"),
        "data": match.group("data"),
    }


def _translate_content(content: Any) -> Any:
    """Translate an OpenAI message `content` into Anthropic blocks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content

    blocks: List[Dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            blocks.append({"type": "text", "text": str(part)})
            continue
        ptype = part.get("type")
        if ptype == "text":
            blocks.append({"type": "text", "text": part.get("text", "")})
        elif ptype == "image_url":
            url = (part.get("image_url") or {}).get("url", "")
            source = _parse_data_url(url)
            if source is None and url:
                # Remote URL — Anthropic supports url sources directly.
                source = {"type": "url", "url": url}
            if source is not None:
                blocks.append({"type": "image", "source": source})
        else:
            # Unknown part type — best-effort stringify.
            blocks.append({"type": "text", "text": str(part)})
    return blocks


def _split_messages(messages: List[Dict[str, Any]]):
    """Extract system text and return (system, anthropic_messages)."""
    system_parts: List[str] = []
    converted: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            else:
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        system_parts.append(part.get("text", ""))
            continue
        anth_role = "assistant" if role == "assistant" else "user"
        converted.append({"role": anth_role, "content": _translate_content(content)})

    merged = _merge_consecutive(converted)
    system = "\n\n".join(p for p in system_parts if p).strip()
    return system, merged


def _merge_consecutive(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Anthropic requires alternating roles; merge consecutive same-role turns."""
    out: List[Dict[str, Any]] = []
    for msg in messages:
        if out and out[-1]["role"] == msg["role"]:
            prev = out[-1]
            prev_blocks = _as_blocks(prev["content"])
            new_blocks = _as_blocks(msg["content"])
            out[-1] = {"role": prev["role"], "content": prev_blocks + new_blocks}
        else:
            out.append(msg)
    return out


def _as_blocks(content: Any) -> List[Dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return list(content)


_JSON_INSTRUCTION = (
    "Return ONLY a single valid JSON value. Do not wrap it in markdown code "
    "fences and do not add any commentary before or after it."
)


def _coerce_json(text: str) -> str:
    """Strip markdown fences / surrounding prose so json.loads() succeeds."""
    stripped = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL)
    if fence:
        stripped = fence.group(1).strip()
    # Extract the outermost JSON object/array if extra prose leaked in.
    start = min([i for i in (stripped.find("{"), stripped.find("[")) if i != -1], default=-1)
    if start > 0:
        end = max(stripped.rfind("}"), stripped.rfind("]"))
        if end > start:
            stripped = stripped[start:end + 1]
    return stripped


def _extract_text(response) -> str:
    parts = []
    for block in response.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts)


def _prompt_cache_on() -> bool:
    return os.getenv("PROMPT_CACHE_ENABLED", "1").lower() not in ("0", "false", "no")


def _system_field(system: str, mode: str):
    """Build the Anthropic ``system`` field, marking the real system prompt with
    a prompt-cache breakpoint so repeated long system prompts are billed cheaply.

    Anthropic ignores ``cache_control`` below the minimum cacheable prefix size,
    so this is always safe — it only ever helps, never errors.
    """
    cache = _prompt_cache_on()
    if mode == "oauth":
        # Subscription tokens require the Claude Code identity as the first block.
        blocks = [{"type": "text", "text": _CLAUDE_CODE_IDENTITY}]
        if system:
            block = {"type": "text", "text": system}
            if cache:
                block["cache_control"] = {"type": "ephemeral"}
            blocks.append(block)
        return blocks
    if not system:
        return None
    if cache:
        return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
    return system


def _record_usage(response, target_model: str) -> None:
    """Best-effort usage/cost accounting — must never break a real request."""
    try:
        usage = getattr(response, "usage", None)
        if usage is not None:
            import usage_tracker

            usage_tracker.record(target_model, usage)
    except Exception:  # noqa: BLE001
        pass


def _build_call(model, messages, temperature, max_tokens, response_format):
    """Shared prep for both streaming and non-streaming calls.

    Returns (client, kwargs, want_json, target_model).
    """
    client, mode = _get_anthropic()
    target_model = resolve_chat_model(model)
    system, anth_messages = _split_messages(messages)

    want_json = bool(response_format) and response_format.get("type") == "json_object"
    if want_json:
        system = (system + "\n\n" + _JSON_INSTRUCTION).strip() if system else _JSON_INSTRUCTION

    kwargs: Dict[str, Any] = {
        "model": target_model,
        "messages": anth_messages,
        "max_tokens": max_tokens or DEFAULT_MAX_TOKENS,
    }
    system_field = _system_field(system, mode)
    if system_field is not None:
        kwargs["system"] = system_field
    if temperature is not None:
        # Anthropic temperature range is [0, 1].
        kwargs["temperature"] = max(0.0, min(1.0, float(temperature)))
    return client, kwargs, want_json, target_model


def stream_text(messages, *, model=None, temperature=None, max_tokens=None):
    """Yield assistant text deltas as they arrive (Claude streaming)."""
    client, kwargs, _want_json, _target = _build_call(
        model, messages, temperature, max_tokens, None
    )
    yielded = False
    try:
        with client.messages.stream(**kwargs) as stream:
            for delta in stream.text_stream:
                if delta:
                    yielded = True
                    yield delta
    except Exception as exc:  # noqa: BLE001
        # Auth errors surface at stream-open (before any delta) — safe to refresh
        # the token and retry once without double-emitting.
        if _is_auth_error(exc) and not yielded:
            _reset_client()
            fresh, _ = _get_anthropic()
            with fresh.messages.stream(**kwargs) as stream:
                for delta in stream.text_stream:
                    if delta:
                        yield delta
        else:
            raise


class _CompletionsNamespace:
    def create(self, *, model: str = None, messages: List[Dict[str, Any]],
               temperature: float = None, max_tokens: int = None,
               response_format: Dict[str, Any] = None, **_ignored) -> _ChatResponse:
        client, kwargs, want_json, target_model = _build_call(
            model, messages, temperature, max_tokens, response_format
        )
        response = _messages_create_with_retry(client, kwargs)
        _record_usage(response, target_model)
        text = _extract_text(response)
        if want_json:
            text = _coerce_json(text)
        message = _Message(content=text)
        return _ChatResponse(choices=[_Choice(message=message)], model=target_model)


class ChatNamespace:
    """Mirrors `client.chat` exposing `.completions.create(...)`."""

    def __init__(self):
        self.completions = _CompletionsNamespace()
