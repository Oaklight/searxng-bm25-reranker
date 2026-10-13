"""BM25 reranking plugin for SearXNG.

Uses sparse_search (zerodep BM25 implementation) to rerank search results
by text relevance, with RRF fusion to preserve engine ranking signals.
"""

from __future__ import annotations

import logging
import math
import re
import typing as t

from searx import settings  # ty: ignore[unresolved-import]
from searx.plugins import Plugin, PluginInfo  # ty: ignore[unresolved-import]

from ._tokenizer import cjk_tokenize
from ._vendor.sparse_search import Result as SparseResult
from ._vendor.sparse_search import SparseIndex, rrf

if t.TYPE_CHECKING:
    from searx.extended_types import SXNG_Request  # ty: ignore[unresolved-import]
    from searx.plugins import PluginCfg  # ty: ignore[unresolved-import]
    from searx.search import SearchWithPlugins  # ty: ignore[unresolved-import]

__version__ = "0.1.0"

logger = logging.getLogger(__name__)


class SXNGPlugin(Plugin):
    """Rerank search results using BM25 scoring with RRF fusion."""

    id = "bm25_reranker"

    def __init__(self, plg_cfg: PluginCfg) -> None:
        super().__init__(plg_cfg)
        self._url_priority = _compile_url_priority(
            _plugin_settings(self.id).get("url_priority")
        )
        self.info = PluginInfo(
            id=self.id,
            name="BM25 Reranker",
            description="Reranks search results using BM25 text relevance scoring with RRF fusion.",
            preference_section="general",
        )

    def post_search(self, request: SXNG_Request, search: SearchWithPlugins) -> None:
        """Rerank results by BM25 relevance fused with original engine ranking.

        Accesses main_results_map before close() calculates scores,
        builds a temporary BM25F index, and rewrites positions to
        influence the final scoring formula.
        """
        results_map = search.result_container.main_results_map
        if len(results_map) < 2:
            return None

        query = search.search_query.query
        if not query or not query.strip():
            return None

        try:
            self._rerank(query, results_map)
        except Exception:
            logger.exception("BM25 reranking failed, keeping original order")

        logger.debug(
            "BM25 reranker processed %d results for: %s", len(results_map), query[:50]
        )

        return None

    def _rerank(self, query: str, results_map: dict) -> None:
        """Core reranking logic.

        Args:
            query: Original search query.
            results_map: Dict of result hash -> MainResult/LegacyResult objects.
        """
        results = list(results_map.values())

        # Build temporary BM25F index with title boost
        idx = SparseIndex(
            variant="bm25",
            field_weights={"title": 2.0, "content": 1.0},
            tokenize=cjk_tokenize,
        )

        valid_indices: list[int] = []
        for i, r in enumerate(results):
            title = _get_text(r, "title")
            content = _get_text(r, "content")
            if not title and not content:
                continue
            idx.add(str(i), {"title": title, "content": content})
            valid_indices.append(i)

        if len(valid_indices) < 2:
            return

        # BM25 retrieval
        bm25_results = idx.search(query, top_k=len(valid_indices))

        # Build engine ranking from original positions
        engine_ranking = [
            SparseResult(doc_id=str(i), score=1.0 / (rank + 1))
            for rank, i in enumerate(valid_indices)
        ]

        # RRF fusion: combine engine ranking + BM25 ranking
        fused = rrf(engine_ranking, bm25_results, k=60)

        # Multiply fused scores for URLs matching a configured pattern
        fused = _apply_url_priority(fused, results, self._url_priority)

        # Rewrite positions to influence calculate_score()
        for new_pos, fused_r in enumerate(fused, start=1):
            idx_int = int(fused_r.doc_id)
            r = results[idx_int]
            # Preserve positions list length (multi-engine boost) but update values
            n_positions = len(r["positions"]) if r["positions"] else 1
            r["positions"] = [new_pos] * max(n_positions, 1)

        logger.debug("BM25 reranked %d results for query: %s", len(fused), query[:50])


def _get_text(result: t.Any, field: str) -> str:
    """Safely extract text field from a result object supporting [] access."""
    try:
        val = result[field]
    except (KeyError, TypeError):
        val = getattr(result, field, "")
    return val or ""


def _plugin_settings(plugin_id: str) -> t.Mapping[str, t.Any]:
    """Read a plugin's own settings block from SearXNG's settings.

    Plugin-specific options do not live under ``plugins:`` -- that mapping
    only carries ``active``, and ``PluginCfg`` is a dataclass that rejects
    unknown keys. SearXNG's convention (see the in-tree hostnames plugin) is
    a top-level block keyed by the plugin id:

    .. code:: yaml

       bm25_reranker:
         url_priority:
           "(.*\\.)?arxiv\\.org$": 1.2

    Args:
        plugin_id: The plugin's :py:attr:`Plugin.id`.

    Returns:
        The settings mapping, or an empty mapping when the block is absent
        or is not a mapping.
    """
    block = settings.get(plugin_id)
    if not isinstance(block, t.Mapping):
        return {}
    return block


def _compile_url_priority(raw: t.Any) -> list[tuple[re.Pattern[str], float]]:
    """Compile the ``url_priority`` mapping into usable patterns.

    ``url_priority`` maps a regular expression, matched against each
    result's URL, to a score multiplier. Multipliers never remove results;
    a multiplier below 1.0 demotes matching URLs, above 1.0 promotes them,
    and 0.0 pushes them to the bottom.

    Malformed entries are skipped with a warning rather than raising: a
    typo in settings.yml must not break search. That covers an invalid
    regex, a multiplier that is not a number, a negative one, and NaN.
    NaN is rejected explicitly -- it is not negative, so it would slip
    through, and a single NaN poisons the sort into an arbitrary order.
    Positive infinity is accepted; it simply pins the URL to the top.

    Args:
        raw: The raw ``url_priority`` value from settings.

    Returns:
        A list of ``(compiled pattern, multiplier)`` pairs. Empty when
        *raw* is absent, is not a mapping, or holds no valid entry.
    """
    if raw is None:
        return []
    if not isinstance(raw, t.Mapping):
        logger.warning(
            "url_priority must be a mapping, ignoring %r", type(raw).__name__
        )
        return []

    compiled: list[tuple[re.Pattern[str], float]] = []
    for pattern, multiplier in raw.items():
        try:
            value = float(multiplier)
        except (TypeError, ValueError):
            logger.warning(
                "url_priority: ignoring non-numeric multiplier %r for %r",
                multiplier,
                pattern,
            )
            continue
        if math.isnan(value):
            logger.warning("url_priority: ignoring NaN multiplier for %r", pattern)
            continue
        if value < 0:
            logger.warning(
                "url_priority: ignoring negative multiplier %r for %r",
                multiplier,
                pattern,
            )
            continue
        try:
            compiled.append((re.compile(str(pattern)), value))
        except re.error:
            logger.warning("url_priority: ignoring invalid regex %r", pattern)

    return compiled


def _apply_url_priority(
    fused: list[SparseResult],
    results: list[t.Any],
    patterns: list[tuple[re.Pattern[str], float]],
) -> list[SparseResult]:
    """Multiply fused scores for results whose URL matches a pattern.

    Every matching pattern applies, so a URL matching several entries has
    its score multiplied by each of them in turn. Ties keep their incoming
    RRF order, because Python's sort is stable.

    Args:
        fused: RRF-fused ranking, where each ``doc_id`` indexes *results*.
        results: List of result objects (supporting ``[]`` access for
            ``"url"``).
        patterns: Compiled ``(pattern, multiplier)`` pairs from
            :func:`_compile_url_priority`.

    Returns:
        A new ranking sorted by descending adjusted score. The input list
        is returned unchanged when *patterns* is empty.
    """
    if not patterns:
        return fused

    adjusted: list[SparseResult] = []
    for item in fused:
        url = _get_text(results[int(item.doc_id)], "url")
        multiplier = 1.0
        for pattern, value in patterns:
            if pattern.search(url):
                multiplier *= value
        adjusted.append(SparseResult(doc_id=item.doc_id, score=item.score * multiplier))

    adjusted.sort(key=lambda r: r.score, reverse=True)
    return adjusted
