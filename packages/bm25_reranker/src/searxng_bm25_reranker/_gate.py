"""Heuristics that decide when BM25 should ask sem_reranker for help.

The gate runs before any network call.  It inspects the BM25 scores and the
result text only.  When the gate is disabled, the coordinator keeps the old
behavior: sem is called whenever it is active.
"""

from __future__ import annotations

import logging
import math
import typing as t
from dataclasses import dataclass

from ._tokenizer import cjk_tokenize

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GateConfig:
    enabled: bool = False
    min_coverage: float = 0.5
    min_margin: float = 0.1
    min_scored: int = 2
    top_k: int = 10


def load_gate(settings: t.Mapping[str, t.Any]) -> GateConfig:
    """Read ``rerankers.escalate`` with safe fallbacks."""
    block = settings.get("rerankers")
    raw = block.get("escalate") if isinstance(block, t.Mapping) else None
    if not isinstance(raw, t.Mapping):
        raw = {}

    return GateConfig(
        enabled=_bool(raw.get("enabled"), False, "escalate.enabled"),
        min_coverage=_ratio(raw.get("min_coverage"), 0.5, "escalate.min_coverage"),
        min_margin=_ratio(raw.get("min_margin"), 0.1, "escalate.min_margin"),
        min_scored=_int(raw.get("min_scored"), 2, 1, "escalate.min_scored"),
        top_k=_int(raw.get("top_k"), 10, 1, "escalate.top_k"),
    )


def should_escalate(
    query: str,
    docs: list[str],
    scores: list[float | None],
    valid_indices: list[int],
    config: GateConfig,
) -> tuple[bool, str]:
    """Return ``(escalate, reason)`` for one request.

    The gate opens when any indicator falls below its threshold.  A disabled
    gate always opens, which preserves the previous "always call sem" behavior.
    """
    if not config.enabled:
        return True, "disabled"

    ranked = [index for index in valid_indices if scores[index] is not None]
    ranked.sort(key=lambda index: (-t.cast(float, scores[index]), index))

    if len(ranked) < config.min_scored:
        return True, "scored"

    top1 = t.cast(float, scores[ranked[0]])
    top2 = t.cast(float, scores[ranked[1]]) if len(ranked) > 1 else 0.0

    if top1 <= 0:
        return True, "top_score"

    margin = (top1 - top2) / top1
    if margin < config.min_margin:
        return True, "margin"

    coverage = best_coverage(query, docs, ranked[: config.top_k])
    if coverage < config.min_coverage:
        return True, "coverage"

    return False, "ok"


def best_coverage(query: str, docs: list[str], indices: list[int]) -> float:
    """Best fraction of query tokens found in one result."""
    tokens = set(cjk_tokenize(query))
    if not tokens:
        return 1.0

    best = 0.0
    for index in indices:
        doc_tokens = set(cjk_tokenize(docs[index]))
        if not doc_tokens:
            continue
        best = max(best, len(tokens & doc_tokens) / len(tokens))
    return best


def _bool(raw: t.Any, default: bool, name: str) -> bool:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return default
    logger.warning("rerankers.%s=%r is not a bool; using %r", name, raw, default)
    return default


def _ratio(raw: t.Any, default: float, name: str) -> float:
    if raw is None:
        return default
    if isinstance(raw, bool):
        logger.warning("rerankers.%s=%r is not a number", name, raw)
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        logger.warning("rerankers.%s=%r is not a number; using %r", name, raw, default)
        return default
    if not math.isfinite(value) or value < 0 or value > 1:
        logger.warning(
            "rerankers.%s must be in [0, 1], got %r; using %r", name, raw, default
        )
        return default
    return value


def _int(raw: t.Any, default: int, minimum: int, name: str) -> int:
    if raw is None:
        return default
    if isinstance(raw, bool):
        logger.warning("rerankers.%s=%r is not a number", name, raw)
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        logger.warning("rerankers.%s=%r is not a number; using %r", name, raw, default)
        return default
    if value < minimum:
        logger.warning(
            "rerankers.%s must be >= %d, got %r; using %r",
            name,
            minimum,
            raw,
            default,
        )
        return default
    return value
