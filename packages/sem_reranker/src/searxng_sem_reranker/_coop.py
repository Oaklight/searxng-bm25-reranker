"""Coordinator selection for two plugins that both write ``positions``.

SearXNG keeps plugins in a set, so ``post_search`` order is not stable.  The
configured coordinator does all the work; the other plugin returns immediately.
"""

from __future__ import annotations

import logging
import typing as t

logger = logging.getLogger(__name__)

COORDINATORS = ("bm25_reranker", "sem_reranker")


def active_plugins(search: t.Any) -> list[str] | None:
    """Return the per-request plugin ids, or ``None`` when unavailable."""
    active = getattr(search, "user_plugins", None)
    if active is None:
        return None
    return list(active)


def coordinator_setting(settings: t.Mapping[str, t.Any]) -> str | None:
    block = settings.get("rerankers")
    if not isinstance(block, t.Mapping):
        return None
    value = block.get("coordinator")
    if isinstance(value, str) and value in COORDINATORS:
        return value
    if value is not None:
        logger.warning(
            "Invalid rerankers.coordinator=%r; expected one of %s",
            value,
            ", ".join(COORDINATORS),
        )
    return None


def is_coordinator(
    settings: t.Mapping[str, t.Any],
    search: t.Any,
    own_id: str,
) -> bool:
    """Decide whether ``own_id`` should perform this request's reranking."""
    active = active_plugins(search)
    if active is None:
        return True

    configured = coordinator_setting(settings)
    if configured is not None:
        if configured in active:
            return configured == own_id
        logger.warning(
            "rerankers.coordinator=%s is not active; %s coordinates instead",
            configured,
            own_id,
        )
        return True

    if "bm25_reranker" in active and "sem_reranker" in active:
        logger.warning(
            "Both rerankers are active and rerankers.coordinator is unset; "
            "sem_reranker coordinates"
        )
        return own_id == "sem_reranker"

    return True
