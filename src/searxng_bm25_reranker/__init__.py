"""BM25 reranking plugin for SearXNG.

Uses sparse_search (zerodep BM25 implementation) to rerank search results
by text relevance, with RRF fusion to preserve engine ranking signals.
"""

from __future__ import annotations

import logging
import typing as t

from searx.plugins import Plugin, PluginInfo

from ._tokenizer import cjk_tokenize
from ._vendor.sparse_search import Result as SparseResult
from ._vendor.sparse_search import SparseIndex, rrf

if t.TYPE_CHECKING:
    from searx.extended_types import SXNG_Request
    from searx.plugins import PluginCfg
    from searx.search import SearchWithPlugins

__version__ = "0.1.0"

logger = logging.getLogger(__name__)


class SXNGPlugin(Plugin):
    """Rerank search results using BM25 scoring with RRF fusion."""

    id = "bm25_reranker"

    def __init__(self, plg_cfg: "PluginCfg") -> None:
        super().__init__(plg_cfg)
        self.info = PluginInfo(
            id=self.id,
            name="BM25 Reranker",
            description="Reranks search results using BM25 text relevance scoring with RRF fusion.",
            preference_section="general",
        )

    def post_search(self, request: "SXNG_Request", search: "SearchWithPlugins") -> None:
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

        logger.debug("BM25 reranker processed %d results for: %s", len(results_map), query[:50])

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
        engine_ranking = [SparseResult(doc_id=str(i), score=1.0 / (rank + 1)) for rank, i in enumerate(valid_indices)]

        # RRF fusion: combine engine ranking + BM25 ranking
        fused = rrf(engine_ranking, bm25_results, k=60)

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
