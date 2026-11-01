"""Settings parsing for the semantic reranker.

There are no provider presets.  Each stage declares its full endpoint URL, key,
format, and model explicitly, so embedding and rerank can come from different
hosts and use different keys.

SearXNG does not validate unknown top-level settings blocks, so this module
validates every value itself and falls back to a safe default with a warning.
"""

from __future__ import annotations

import logging
import math
import os
import typing as t
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

EMBED_FORMATS = ("openai", "cohere", "jina", "voyage")
RERANK_FORMATS = ("jina", "cohere", "voyage")


@dataclass(frozen=True)
class StageConfig:
    enabled: bool
    format: str
    url: str
    model: str
    api_key_env: str
    api_key: str = field(repr=False)
    top_k: int
    weight: float
    max_chars: int
    timeout_ms: int

    def resolve_key(self) -> str:
        """Return the env key first, then the settings value."""
        env_value = os.environ.get(self.api_key_env, "") if self.api_key_env else ""
        return env_value or self.api_key


@dataclass(frozen=True)
class FusionConfig:
    rrf_k: int = 20
    engine_weight: float = 1.0
    bm25_weight: float = 1.0
    neural_weight: float = 1.0


@dataclass(frozen=True)
class CacheConfig:
    ttl: int = 300
    max_entries: int = 512


@dataclass(frozen=True)
class Config:
    embedding: StageConfig
    rerank: StageConfig
    fusion: FusionConfig
    cache: CacheConfig


def load_config(settings: t.Mapping[str, t.Any]) -> Config:
    """Build a validated config from the ``sem_reranker:`` settings block."""
    block = settings.get("sem_reranker")
    if not isinstance(block, t.Mapping):
        block = {}

    return Config(
        embedding=_stage(
            block.get("embedding"),
            EMBED_FORMATS,
            "embedding",
            default_top_k=0,
        ),
        rerank=_stage(
            block.get("rerank"),
            RERANK_FORMATS,
            "rerank",
            default_top_k=10,
        ),
        fusion=_fusion(block.get("fusion")),
        cache=_cache(block.get("cache")),
    )


def _stage(
    raw: t.Any,
    formats: tuple[str, ...],
    name: str,
    *,
    default_top_k: int,
) -> StageConfig:
    if not isinstance(raw, t.Mapping):
        raw = {}

    return StageConfig(
        enabled=_bool(raw.get("enabled"), False, f"{name}.enabled"),
        format=_choice(raw.get("format"), formats, f"{name}.format"),
        url=_url(raw.get("url"), f"{name}.url"),
        model=_string(raw.get("model"), "", f"{name}.model"),
        api_key_env=_string(raw.get("api_key_env"), "", f"{name}.api_key_env"),
        api_key=_string(raw.get("api_key"), "", f"{name}.api_key"),
        top_k=_int(raw.get("top_k"), default_top_k, 0, f"{name}.top_k"),
        weight=_float(raw.get("weight"), 1.0, 0.0, f"{name}.weight"),
        max_chars=_int(raw.get("max_chars"), 4000, 1, f"{name}.max_chars"),
        timeout_ms=_int(raw.get("timeout_ms"), 3000, 1, f"{name}.timeout_ms"),
    )


def _fusion(raw: t.Any) -> FusionConfig:
    if not isinstance(raw, t.Mapping):
        raw = {}
    return FusionConfig(
        rrf_k=_int(raw.get("rrf_k"), 20, 1, "fusion.rrf_k"),
        engine_weight=_float(
            raw.get("engine_weight"), 1.0, 0.0, "fusion.engine_weight"
        ),
        bm25_weight=_float(raw.get("bm25_weight"), 1.0, 0.0, "fusion.bm25_weight"),
        neural_weight=_float(
            raw.get("neural_weight"), 1.0, 0.0, "fusion.neural_weight"
        ),
    )


def _cache(raw: t.Any) -> CacheConfig:
    if not isinstance(raw, t.Mapping):
        raw = {}
    return CacheConfig(
        ttl=_int(raw.get("ttl"), 300, 0, "cache.ttl"),
        max_entries=_int(raw.get("max_entries"), 512, 1, "cache.max_entries"),
    )


def _bool(raw: t.Any, default: bool, name: str) -> bool:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return default
    logger.warning("sem_reranker.%s=%r is not a bool; using %r", name, raw, default)
    return default


def _int(raw: t.Any, default: int, minimum: int, name: str) -> int:
    if raw is None:
        return default
    if isinstance(raw, bool):
        logger.warning("sem_reranker.%s=%r is not a number", name, raw)
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        logger.warning(
            "sem_reranker.%s=%r is not a number; using %r", name, raw, default
        )
        return default
    if value < minimum:
        logger.warning(
            "sem_reranker.%s must be >= %d, got %r; using %r",
            name,
            minimum,
            raw,
            default,
        )
        return default
    return value


def _float(raw: t.Any, default: float, minimum: float, name: str) -> float:
    if raw is None:
        return default
    if isinstance(raw, bool):
        logger.warning("sem_reranker.%s=%r is not a number", name, raw)
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        logger.warning(
            "sem_reranker.%s=%r is not a number; using %r", name, raw, default
        )
        return default
    if not math.isfinite(value) or value < minimum:
        logger.warning(
            "sem_reranker.%s must be >= %s, got %r; using %r",
            name,
            minimum,
            raw,
            default,
        )
        return default
    return value


def _string(raw: t.Any, default: str, name: str) -> str:
    if raw is None:
        return default
    if isinstance(raw, str):
        return raw
    logger.warning("sem_reranker.%s=%r is not a string; using %r", name, raw, default)
    return default


def _choice(raw: t.Any, choices: tuple[str, ...], name: str) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str) and raw in choices:
        return raw
    logger.warning(
        "sem_reranker.%s=%r is not one of %s; stage will be skipped",
        name,
        raw,
        ", ".join(choices),
    )
    return ""


def _url(raw: t.Any, name: str) -> str:
    if raw is None or raw == "":
        return ""
    if isinstance(raw, str) and raw.startswith(("http://", "https://")):
        return raw
    logger.warning(
        "sem_reranker.%s=%r is not an HTTP URL; stage will be skipped", name, raw
    )
    return ""
