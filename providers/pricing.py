# providers/pricing.py
"""Token cost estimation for observability.

Prices are USD per 1M tokens and are matched by a substring of the model id so
version suffixes (``claude-haiku-4-5-20251001``) still resolve. These are
ESTIMATES for relative cost tracking — confirm against the current Anthropic
price sheet before using them for billing. Override any value via env, e.g.
``PRICE_HAIKU_IN`` / ``PRICE_SONNET_OUT``.
"""
from __future__ import annotations

import os
from typing import Any, Dict

# (input, output) USD per 1,000,000 tokens.
_DEFAULTS = {
    "haiku": (1.00, 5.00),
    "sonnet": (3.00, 15.00),
    "opus": (15.00, 75.00),
}

# Anthropic prompt-cache multipliers relative to the base input price.
_CACHE_READ_MULT = 0.1    # cached prefix reads are ~10% of input price
_CACHE_WRITE_MULT = 1.25  # writing a cache entry costs ~25% more than input

_PER_MILLION = 1_000_000.0


def _tier(model: str) -> str:
    m = (model or "").lower()
    if "haiku" in m:
        return "haiku"
    if "opus" in m:
        return "opus"
    return "sonnet"  # default / "sonnet" / unknown smart-tier ids


def _rates(model: str) -> tuple[float, float]:
    tier = _tier(model)
    base_in, base_out = _DEFAULTS[tier]
    in_price = float(os.getenv(f"PRICE_{tier.upper()}_IN", base_in))
    out_price = float(os.getenv(f"PRICE_{tier.upper()}_OUT", base_out))
    return in_price, out_price


def estimate_cost(
    model: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """Return the estimated USD cost of one call. Pure function."""
    in_price, out_price = _rates(model)
    cost = (
        input_tokens * in_price
        + output_tokens * out_price
        + cache_read_tokens * in_price * _CACHE_READ_MULT
        + cache_write_tokens * in_price * _CACHE_WRITE_MULT
    ) / _PER_MILLION
    return round(cost, 6)


def usage_to_dict(usage: Any) -> Dict[str, int]:
    """Normalise an Anthropic ``usage`` object into a plain int dict."""
    def _get(name: str) -> int:
        return int(getattr(usage, name, 0) or 0)

    return {
        "input_tokens": _get("input_tokens"),
        "output_tokens": _get("output_tokens"),
        "cache_read_tokens": _get("cache_read_input_tokens"),
        "cache_write_tokens": _get("cache_creation_input_tokens"),
    }
