"""SearXNG plugin class.

Kept separate from ``__init__`` so the scoring library can be imported without
SearXNG.  The package exposes ``SXNGPlugin`` lazily through ``__getattr__``.
"""

from __future__ import annotations

import logging
import typing as t

import searxng_sem_reranker as sem
from searx import settings  # ty: ignore[unresolved-import]
from searx.plugins import Plugin, PluginInfo  # ty: ignore[unresolved-import]

from ._config import load_config
from ._coop import is_coordinator
from ._positions import rewrite_positions
from ._vendor.semantic_search import Result, rrf

if t.TYPE_CHECKING:
    from searx.extended_types import SXNG_Request  # ty: ignore[unresolved-import]
    from searx.plugins import PluginCfg  # ty: ignore[unresolved-import]
    from searx.search import SearchWithPlugins  # ty: ignore[unresolved-import]

logger = logging.getLogger(__name__)


class SXNGPlugin(Plugin):
    """Rerank results with embedding similarity, a cross-encoder, or both."""

    id = "sem_reranker"

    def __init__(self, plg_cfg: PluginCfg) -> None:
        super().__init__(plg_cfg)
        self.info = PluginInfo(
            id=self.id,
            name="Semantic Reranker",
            description="Reranks search results with embedding and cross-encoder scores.",
            preference_section="general",
        )

    def post_search(self, request: SXNG_Request, search: SearchWithPlugins) -> None:
        """Rerank once when this plugin is the configured coordinator."""
        try:
            if not is_coordinator(settings, search, self.id):
                return None

            results_map = search.result_container.main_results_map
            if len(results_map) < 2:
                return None

            query = search.search_query.query
            if not query or not query.strip():
                return None

            self._rerank(query, results_map, search)
            logger.debug(
                "Semantic reranker processed %d results for: %s",
                len(results_map),
                query[:50],
            )
        except Exception:
            logger.exception("Semantic reranking failed, keeping original order")

        return None

    def _rerank(
        self,
        query: str,
        results_map: dict[str, t.Any],
        search: SearchWithPlugins | None = None,
    ) -> None:
        results = list(results_map.values())
        cfg = load_config(settings)
        valid = [index for index, result in enumerate(results) if sem._text(result)]
        if len(valid) < 2:
            return

        neural = sem.score_neural(query, results, cfg)
        bm25 = sem._bm25_scores(query, results, search)
        has_neural = any(score is not None for score in neural)
        has_bm25 = bm25 is not None and any(score is not None for score in bm25)

        if not has_neural and not has_bm25:
            return

        engine_ranking = [
            Result(doc_id=str(index), score=1.0 / (rank + 1))
            for rank, index in enumerate(valid)
        ]
        lists = [engine_ranking]
        weights = [cfg.fusion.engine_weight]

        if has_bm25:
            assert bm25 is not None
            lists.append(
                [
                    Result(doc_id=str(index), score=t.cast(float, bm25[index]))
                    for index in sem.rank_by_score(bm25)
                ]
            )
            weights.append(cfg.fusion.bm25_weight)

        if has_neural:
            lists.append(
                [
                    Result(doc_id=str(index), score=t.cast(float, neural[index]))
                    for index in sem.rank_by_score(neural)
                ]
            )
            weights.append(cfg.fusion.neural_weight)

        fused = rrf(*lists, k=cfg.fusion.rrf_k, weights=weights)
        fused.sort(key=lambda item: (-item.score, int(item.doc_id)))

        if has_bm25:
            fused = sem._apply_bm25_url_priority(fused, results)

        rewrite_positions(results, [int(item.doc_id) for item in fused])

        logger.debug("Semantic reranked %d results for: %s", len(fused), query[:50])
