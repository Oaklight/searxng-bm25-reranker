"""Semantic embedding and cross-encoder reranking for SearXNG.

The scoring library imports without SearXNG.  ``SXNGPlugin`` is loaded lazily
from :mod:`searxng_sem_reranker._plugin` so the BM25 coordinator can import
``score_neural`` in a process that does not load the plugin class.

Runtime code uses only the standard library and vendored zerodep modules.
"""

from __future__ import annotations

__version__ = "0.1.0"

import hashlib
import html as html_lib
import logging
import re
import threading
import typing as t

from ._config import Config, StageConfig, load_config
from ._providers import ProviderError
from ._providers import embed as embed_texts
from ._providers import rerank as rerank_documents
from ._vendor.cache import TTLCache
from ._vendor.semantic_search import Result, cosine_similarity

__all__ = ["score_neural", "rank_by_score"]

logger = logging.getLogger(__name__)

_TAG_RE = re.compile(r"</?[a-zA-Z](?:[^>\"']|\"[^\"]*\"|'[^']*')*>")

_CACHE_LOCK = threading.Lock()
_EMBED_CACHE: TTLCache | None = None
_RERANK_CACHE: TTLCache | None = None
_CACHE_SHAPE: tuple[int, int] | None = None


def __getattr__(name: str) -> t.Any:
    if name == "SXNGPlugin":
        from ._plugin import SXNGPlugin

        return SXNGPlugin
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def score_neural(
    query: str,
    results: list[t.Any],
    config: Config | None = None,
) -> list[float | None]:
    """Return one neural score per result, or ``None`` where unavailable.

    Embedding similarity is computed for every result with text.  When the
    rerank stage is enabled, it overrides the top candidates; other results keep
    their embedding score.  Provider failures leave the affected scores ``None``
    so the caller can fall back to BM25 or engine order.
    """
    cfg = config if config is not None else load_config(_settings())
    count = len(results)
    scores: list[float | None] = [None] * count
    texts = [_text(result) for result in results]

    if cfg.embedding.enabled and _usable(cfg.embedding, "embedding"):
        embedding_scores = _embedding_scores(query, texts, cfg)
        if embedding_scores is not None:
            scores = embedding_scores

    _apply_rerank(query, texts, scores, cfg)
    return scores


def _apply_rerank(
    query: str,
    texts: list[str],
    scores: list[float | None],
    cfg: Config,
) -> None:
    """Override the top candidates with cross-encoder scores, in place."""
    if not cfg.rerank.enabled or not _usable(cfg.rerank, "rerank"):
        return

    if any(score is not None for score in scores):
        candidates = rank_by_score(scores)
    else:
        candidates = [index for index, text in enumerate(texts) if text]

    limit = cfg.rerank.top_k or len(candidates)
    candidates = candidates[:limit]
    if not candidates:
        return

    documents = [texts[index][: cfg.rerank.max_chars] for index in candidates]
    try:
        rerank_scores = _cached_rerank(cfg, query, documents)
    except ProviderError as exc:
        logger.warning("Rerank request failed: %s", exc)
        return

    for index, score in zip(candidates, rerank_scores):
        if score is not None:
            scores[index] = score


def rank_by_score(scores: list[float | None]) -> list[int]:
    """Sort indices by descending score, breaking ties by original index."""
    return sorted(
        (index for index, score in enumerate(scores) if score is not None),
        key=lambda index: (-t.cast(float, scores[index]), index),
    )


def _embedding_scores(
    query: str,
    texts: list[str],
    cfg: Config,
) -> list[float | None] | None:
    try:
        vectors = _cached_embeddings(cfg, query, texts)
    except ProviderError as exc:
        logger.warning("Embedding request failed: %s", exc)
        return None

    query_vector = vectors[0]
    if query_vector is None:
        return None

    scores: list[float | None] = [None] * len(texts)
    for index, vector in enumerate(vectors[1:]):
        if vector is None:
            continue
        if len(vector) != len(query_vector):
            logger.warning(
                "Embedding dimension mismatch: query=%d document=%d",
                len(query_vector),
                len(vector),
            )
            continue
        scores[index] = cosine_similarity(query_vector, vector)
    return scores


def _cached_embeddings(
    cfg: Config,
    query: str,
    texts: list[str],
) -> list[list[float] | None]:
    """Embed the query and documents, reusing cached vectors per text."""
    stage = cfg.embedding
    cache, _ = _caches(cfg)
    query = query[: stage.max_chars]
    clipped = [text[: stage.max_chars] for text in texts]
    nonempty = [index for index, text in enumerate(clipped) if text]

    vectors: list[list[float] | None] = [None] * (len(texts) + 1)

    if stage.format == "openai":
        batch = [query] + [clipped[index] for index in nonempty]
        fetched = _fetch_vectors(stage, batch, query=True, cache=cache)
        vectors[0] = fetched[0]
        for index, vector in zip(nonempty, fetched[1:]):
            vectors[index + 1] = vector
        return vectors

    vectors[0] = _fetch_vectors(stage, [query], query=True, cache=cache)[0]
    if nonempty:
        documents = [clipped[index] for index in nonempty]
        fetched = _fetch_vectors(stage, documents, query=False, cache=cache)
        for index, vector in zip(nonempty, fetched):
            vectors[index + 1] = vector
    return vectors


def _fetch_vectors(
    stage: StageConfig,
    texts: list[str],
    *,
    query: bool,
    cache: TTLCache,
) -> list[list[float] | None]:
    keys = [_embedding_key(stage, text, query) for text in texts]
    with _CACHE_LOCK:
        cached: list[list[float] | None] = [cache.get(key) for key in keys]

    missing = [index for index, vector in enumerate(cached) if vector is None]
    if missing:
        vectors = embed_texts(stage, [texts[index] for index in missing], query=query)
        for index, vector in zip(missing, vectors):
            cached[index] = vector
            with _CACHE_LOCK:
                cache[keys[index]] = vector
    return cached


def _cached_rerank(
    cfg: Config,
    query: str,
    documents: list[str],
) -> list[float | None]:
    stage = cfg.rerank
    _, cache = _caches(cfg)
    query = query[: stage.max_chars]
    keys = [_rerank_key(stage, query, document) for document in documents]

    with _CACHE_LOCK:
        cached = [cache.get(key) for key in keys]
    if all(score is not None for score in cached):
        return cached

    scores = rerank_documents(stage, query, documents)
    for key, score in zip(keys, scores):
        if score is not None:
            with _CACHE_LOCK:
                cache[key] = score
    return scores


def _caches(cfg: Config) -> tuple[TTLCache, TTLCache]:
    global _EMBED_CACHE, _RERANK_CACHE, _CACHE_SHAPE

    shape = (cfg.cache.max_entries, cfg.cache.ttl)
    with _CACHE_LOCK:
        if _CACHE_SHAPE != shape or _EMBED_CACHE is None or _RERANK_CACHE is None:
            _EMBED_CACHE = TTLCache(maxsize=shape[0], ttl=shape[1])
            _RERANK_CACHE = TTLCache(maxsize=shape[0], ttl=shape[1])
            _CACHE_SHAPE = shape
        return _EMBED_CACHE, _RERANK_CACHE


def _embedding_key(stage: StageConfig, text: str, query: bool) -> str:
    raw = "\0".join(
        (
            stage.format,
            stage.model,
            stage.url,
            "query" if query else "document",
            text,
        )
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _rerank_key(stage: StageConfig, query: str, document: str) -> str:
    raw = "\0".join((stage.format, stage.model, stage.url, query, document))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _usable(stage: StageConfig, name: str) -> bool:
    if not stage.format:
        logger.warning(
            "sem_reranker.%s.format is unset or invalid; stage skipped", name
        )
        return False
    if not stage.url:
        logger.warning("sem_reranker.%s.url is empty; stage skipped", name)
        return False
    if stage.api_key_env and not stage.resolve_key():
        logger.warning(
            "sem_reranker.%s: %s is unset and api_key is empty; stage skipped",
            name,
            stage.api_key_env,
        )
        return False
    return True


def _bm25_scores(
    query: str,
    results: list[t.Any],
    search: t.Any,
) -> list[float | None] | None:
    active = getattr(search, "user_plugins", None)
    if active is not None and "bm25_reranker" not in list(active):
        return None

    try:
        from searxng_bm25_reranker import score_results  # ty: ignore[unresolved-import]
    except ImportError:
        return None

    try:
        return score_results(query, results)
    except Exception:
        logger.exception("BM25 scoring failed, continuing with neural scores only")
        return None


def _apply_bm25_url_priority(
    fused: list[Result],
    results: list[t.Any],
) -> list[Result]:
    try:
        from searxng_bm25_reranker import (  # ty: ignore[unresolved-import]
            _apply_url_priority,
            _compile_url_priority,
            _plugin_settings,
        )
    except ImportError:
        return fused

    patterns = _compile_url_priority(
        _plugin_settings("bm25_reranker").get("url_priority")
    )
    if not patterns:
        return fused
    return _apply_url_priority(fused, results, patterns)


def _text(result: t.Any) -> str:
    title = _get_text(result, "title")
    content = _get_text(result, "content")
    return (title + "\n" + content).strip()


def _get_text(result: t.Any, field: str) -> str:
    """Extract a text field and strip engine HTML before sending it out."""
    try:
        value = result[field]
    except (KeyError, IndexError, TypeError):
        value = getattr(result, field, "")
    if not isinstance(value, str):
        return ""
    return _TAG_RE.sub(" ", html_lib.unescape(value))


def _settings() -> t.Mapping[str, t.Any]:
    """Read SearXNG settings when available; return an empty mapping otherwise."""
    try:
        from searx import settings  # ty: ignore[unresolved-import]
    except ImportError:
        return {}
    return settings
