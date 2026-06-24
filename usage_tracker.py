"""In-process AI usage + cost observability.

Records token usage and estimated cost from every Claude call so we can see what
the product actually spends (per model, and in total). Thread-safe and best-effort:
recording must NEVER break a real request, so all call sites guard with try/except.

This is an in-memory accumulator on a single always-on instance — fine for launch
visibility. Persistent per-user attribution lands with the Stripe work (Phase 5),
where a Supabase sink can subscribe via ``set_sink``.
"""
from __future__ import annotations

import threading
from typing import Any, Callable, Dict, Optional

from providers.pricing import estimate_cost, usage_to_dict

_lock = threading.Lock()
_totals: Dict[str, Dict[str, float]] = {}
_sink: Optional[Callable[[Dict[str, Any]], None]] = None


def set_sink(fn: Optional[Callable[[Dict[str, Any]], None]]) -> None:
    """Register an optional callback (e.g. a Supabase writer) for each event."""
    global _sink
    _sink = fn


def record(model: str, usage: Any, *, user_id: Optional[str] = None) -> Dict[str, Any]:
    """Record one call's usage. Returns the event dict (also useful for tests)."""
    tokens = usage_to_dict(usage)
    cost = estimate_cost(
        model,
        input_tokens=tokens["input_tokens"],
        output_tokens=tokens["output_tokens"],
        cache_read_tokens=tokens["cache_read_tokens"],
        cache_write_tokens=tokens["cache_write_tokens"],
    )
    event = {"model": model, "user_id": user_id, "cost_usd": cost, **tokens}

    with _lock:
        bucket = _totals.setdefault(
            model,
            {"calls": 0, "input_tokens": 0, "output_tokens": 0,
             "cache_read_tokens": 0, "cache_write_tokens": 0, "cost_usd": 0.0},
        )
        bucket["calls"] += 1
        for k in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"):
            bucket[k] += tokens[k]
        bucket["cost_usd"] = round(bucket["cost_usd"] + cost, 6)

    if _sink is not None:
        try:
            _sink(event)
        except Exception:  # noqa: BLE001 — a sink failure must not affect the request
            pass
    return event


def snapshot() -> Dict[str, Any]:
    """Return a copy of the accumulated totals (per model + grand total)."""
    with _lock:
        by_model = {m: dict(v) for m, v in _totals.items()}
    grand = {
        "calls": sum(v["calls"] for v in by_model.values()),
        "cost_usd": round(sum(v["cost_usd"] for v in by_model.values()), 6),
        "input_tokens": sum(v["input_tokens"] for v in by_model.values()),
        "output_tokens": sum(v["output_tokens"] for v in by_model.values()),
    }
    return {"by_model": by_model, "total": grand}


def reset() -> None:
    """Clear all counters (test helper)."""
    with _lock:
        _totals.clear()
