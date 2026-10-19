"""BM25 reranking plugin for SearXNG.

Uses sparse_search (zerodep BM25 implementation) to rerank search results
by text relevance, with RRF fusion to preserve engine ranking signals.
"""

from __future__ import annotations

import html as html_lib
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

# Only an actual tag: `<x`, `</x`, with quoted attribute values allowed to
# contain `>`. A bare `<` followed by punctuation or space (e.g.
# `a < b and c > d`) must survive untouched.
_TAG_RE = re.compile(r"</?[a-zA-Z](?:[^>\"']|\"[^\"]*\"|'[^']*')*>")

# Tuning defaults. Each one can be overridden per instance in settings.yml via a
# top-level block keyed by the plugin id, e.g.
#   bm25_reranker: { rrf_k: 10 }
# Values must be numbers >= 1.
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
        self._url_priority = _compile_url_priority(
            _plugin_settings(self.id).get("url_priority")
        )
        self.info = PluginInfo(
            id=self.id,
            name="BM25 Reranker",
            description="Reranks search results using BM25 text relevance scoring with RRF fusion.",
            preference_section="general",
        )

    def _setting(self, name: str, *, integer: bool = False) -> float:
        """Read an optional numeric setting, falling back to the default.

        Tunables live in a top-level ``bm25_reranker:`` block in settings.yml,
        not under ``plugins:`` -- that mapping only carries ``active``, and
        ``PluginCfg`` is a dataclass that rejects unknown keys (see
        :func:`_plugin_settings`). A bad value must not take the reranker down.

        Values below 1 are rejected: every tunable is a fusion constant or a
        relative weight, and a sub-unit ``rrf_k`` truncates to 0 under ``int()``
        and makes ``rrf()`` raise. Booleans are rejected because ``float(True)``
        is ``1.0`` and would slip past the bound check. With *integer* set the
        value must also be whole; ``rrf_k`` is passed to :func:`rrf` through
        ``int()``, so ``1.9`` would otherwise truncate to ``1`` silently.
        """
        raw = _plugin_settings(self.id).get(name)
        if raw is None:
            return DEFAULTS[name]
        if isinstance(raw, bool):
            logger.warning("Invalid %s=%r, using %s", name, raw, DEFAULTS[name])
            return DEFAULTS[name]
        try:
            value = float(raw)
        except (TypeError, ValueError):
            logger.warning("Invalid %s=%r, using %s", name, raw, DEFAULTS[name])
            return DEFAULTS[name]
        if value < 1:
            logger.warning(
                "%s must be >= 1, got %r, using %s", name, raw, DEFAULTS[name]
            )
            return DEFAULTS[name]
        if integer and not value.is_integer():
            logger.warning(
                "%s must be an integer, got %r, using %s", name, raw, DEFAULTS[name]
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
            k=int(self._setting("rrf_k", integer=True)),
            weights=[
                self._setting("engine_weight"),
                self._setting("bm25_weight"),
            ],
        )

        # Multiply fused scores for URLs matching a configured pattern
        fused = _apply_url_priority(fused, results, self._url_priority)

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
    # Decode entities before stripping. Stripping first lets `&lt;script&gt;`
    # survive the regex and then unescape into a real tag, whose words would be
    # indexed as content -- the noise this function exists to remove.
    val = html_lib.unescape(val)
    return _TAG_RE.sub(" ", val)


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
    regex, a multiplier that is not a number, a negative one, NaN, and a
    boolean. NaN and booleans are rejected explicitly because both would
    otherwise slip through: NaN is not negative, and a bool is a legal
    ``float()`` argument even though ``false`` reads as "switch this rule
    off". A single NaN poisons the sort into an arbitrary order. Positive
    infinity is accepted; it simply pins the URL to the top.

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
        if isinstance(multiplier, bool):
            # bool is a legal float() argument, so it slips past the
            # "not a number" contract. False is the one that bites: it
            # becomes 0.0, which buries the domain instead of switching the
            # rule off, and YAML's no/off/false all parse to it.
            logger.warning(
                "url_priority: ignoring boolean multiplier %r for %r",
                multiplier,
                pattern,
            )
            continue
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
        A new ranking sorted by descending adjusted score. When *patterns*
        is empty the input list is returned as-is, by identity, so the
        caller can rely on no reordering having happened.
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
        if math.isnan(multiplier):
            # Each entry passed the input guard, but the product can still
            # be NaN (inf * 0.0). One NaN is enough to make the sort below
            # non-transitive, so fall back to the unadjusted score.
            logger.warning(
                "url_priority: NaN from overlapping patterns for %r, leaving score unadjusted",
                url,
            )
            multiplier = 1.0
        adjusted.append(SparseResult(doc_id=item.doc_id, score=item.score * multiplier))

    adjusted.sort(key=lambda r: r.score, reverse=True)
    return adjusted
