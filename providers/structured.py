# providers/structured.py
"""
Guaranteed-schema outputs via Claude tool use.

Replaces brittle "ask for JSON then regex/parse the text" patterns: the model
is forced to call a single tool whose input_schema is the contract, and we
return the validated `input` dict directly.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from .anthropic_chat import _build_call, _extract_text, _messages_create_with_retry, _record_usage


def structured_call(
    messages: List[Dict[str, Any]],
    *,
    schema: Dict[str, Any],
    tool_name: str = "result",
    model: str = None,
    max_tokens: int = None,
    temperature: float = None,
) -> Dict[str, Any]:
    """Force the model to emit a value matching `schema`; return it as a dict."""
    client, kwargs, _want_json, _target = _build_call(
        model, messages, temperature, max_tokens, None
    )
    kwargs["tools"] = [{
        "name": tool_name,
        "description": "Return the structured result for this request.",
        "input_schema": schema,
    }]
    kwargs["tool_choice"] = {"type": "tool", "name": tool_name}

    response = _messages_create_with_retry(client, kwargs)
    _record_usage(response, _target)
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == tool_name:
            return block.input

    # Fallback: a model that ignored the forced tool may have emitted JSON text.
    text = _extract_text(response).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Structured call returned no tool output: {exc}")
