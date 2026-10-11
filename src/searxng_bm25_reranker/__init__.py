"""BM25 reranking plugin for SearXNG.

Uses sparse_search (zerodep BM25 implementation) to rerank search results
by text relevance, with RRF fusion to preserve engine ranking signals.
"""

from __future__ import annotations

import html as html_lib
import logging
import re
import typing as t

from searx.plugins import Plugin, PluginInfo  # ty: ignore[unresolved-import]

from ._tokenizer import cjk_tokenize
from ._vendor.sparse_search import Result as SparseResult
from ._vendor.sparse_search import SparseIndex, rrf

if t.TYPE_CHECKING:
    from searx.extended_types import SXNG_Request  # ty: ignore[unresolved-import]
    from searx.plugins import PluginCfg  # ty: ignore[unresolved-import]
    from searx.search import SearchWithPlugins  # ty: ignore[unresolved-import]

__version__ = "0.2.0"

logger = logging.getLogger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")

# Tuning defaults. Each one can be overridden per instance in settings.yml, e.g.
#   plugins: { searxng_bm25_reranker.SXNGPlugin: { active: true, rrf_k: 10 } }
# All values must be positive numbers.
DEFAULTS: dict[str, float] = {
    # RRF constant. The Cormack et al. default of 60 was tuned on retrieval runs
    # of ~1000 candidates; SearXNG pages hold ~30, where 60 flattens rank 1
    # against rank 30 to 1.48x and mutes the relevance signal.
    "rrf_k": 20.0,
    "engine_weight": 1.0,
    "bm25_weight": 1.0,
    "title_weight": 2.0,
    "content_weight": 1.0,
}


class SXNGPlugin(Plugin):
    """Rerank search results using BM25 scoring with RRF fusion."""

    id = "bm25_reranker"

    def __init__(self, plg_cfg: PluginCfg) -> None:
        super().__init__(plg_cfg)
        self.info = PluginInfo(
            id=self.id,
            name="BM25 Reranker",
            description="Reranks search results using BM25 text relevance scoring with RRF fusion.",
            preference_section="general",
        )

    def _setting(self, name: str) -> float:
        """Read an optional numeric setting, falling back to the default.

        SearXNG hands over a PluginCfg that may or may not carry the key, and
        a bad value in settings.yml must not take the whole reranker down.
        """
        raw = getattr(self.plg_cfg, name, None)
        if raw is None:
            getter = getattr(self.plg_cfg, "get", None)
            raw = getter(name) if callable(getter) else None
        if raw is None:
            return DEFAULTS[name]
        try:
            value = float(raw)
        except (TypeError, ValueError):
            logger.warning("Invalid %s=%r, using %s", name, raw, DEFAULTS[name])
            return DEFAULTS[name]
        if value <= 0:
            logger.warning(
                "%s must be positive, got %r, using %s", name, raw, DEFAULTS[name]
            )
            return DEFAULTS[name]
        return value

    def post_search(self, request: SXNG_Request, search: SearchWithPlugins) -> None:
        """Rerank results by BM25 relevance fused with original engine ranking.

        Every touch of the SearXNG API stays inside the try block: a SearXNG
        upgrade must degrade to "original order", never break the search.
        """
        try:
            results_map = search.result_container.main_results_map
            if len(results_map) < 2:
                return None

            query = search.search_query.query
            if not query or not query.strip():
                return None

            self._rerank(query, results_map)
            logger.debug(
                "BM25 reranker processed %d results for: %s",
                len(results_map),
                query[:50],
            )
        except Exception:
            logger.exception("BM25 reranking failed, keeping original order")

        return None

    def _rerank(self, query: str, results_map: dict[str, t.Any]) -> None:
        """Core reranking logic.

        Args:
            query: Original search query.
            results_map: Dict of result hash -> MainResult/LegacyResult objects.
        """
        results = list(results_map.values())

        # Temporary BM25F index with a title boost. delta=0 keeps this classic
        # Okapi BM25, matching the documented default.
        idx = SparseIndex(
            variant="bm25",
            delta=0.0,
            field_weights={
                "title": self._setting("title_weight"),
                "content": self._setting("content_weight"),
            },
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

        bm25_results = idx.search(query, top_k=len(valid_indices))

        # Engine ranking mirrors the original insertion (engine) order.
        engine_ranking = [
            SparseResult(doc_id=str(i), score=1.0 / (rank + 1))
            for rank, i in enumerate(valid_indices)
        ]

        fused = rrf(
            engine_ranking,
            bm25_results,
            k=int(self._setting("rrf_k")),
            weights=[
                self._setting("engine_weight"),
                self._setting("bm25_weight"),
            ],
        )

        _rewrite_positions(results, fused)

        logger.debug("BM25 reranked %d results for query: %s", len(fused), query[:50])


def _rewrite_positions(results: list[t.Any], fused: list[SparseResult]) -> None:
    """Give every result a fresh, contiguous position.

    Results with no indexable text are absent from ``fused``. Leaving their raw
    engine positions untouched would collide with the rewritten ones, so two
    results could end up with the same position and therefore the same
    ``calculate_score`` -- an arbitrary tie instead of a ranking.
    """
    new_positions = {int(f.doc_id): pos for pos, f in enumerate(fused, start=1)}
    next_position = len(new_positions) + 1

    for i, r in enumerate(results):
        # Preserve list length: SearXNG multiplies weight by len(positions),
        # which is how a result found by several engines keeps its boost.
        old = r["positions"] or []
        count = max(len(old), 1)
        if i in new_positions:
            r["positions"] = [new_positions[i]] * count
        else:
            r["positions"] = [next_position] * count
            next_position += 1


def _get_text(result: t.Any, field: str) -> str:
    """Extract a text field from an object supporting ``[]`` or attribute access.

    SearXNG hands over raw engine HTML, so tags are dropped and entities
    decoded; otherwise markup noise (div, href, www, com) drowns real terms
    and inflates the term frequency of the ones that matter.
    """
    try:
        val = result[field]
    except (KeyError, IndexError, TypeError):
        val = getattr(result, field, "")

    if not isinstance(val, str):
        return ""
    if "<" in val and ">" in val:
        val = _TAG_RE.sub(" ", val)
    return html_lib.unescape(val)
