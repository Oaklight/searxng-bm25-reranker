"""BM25 reranking plugin for SearXNG.

Uses sparse_search (zerodep BM25 implementation) to rerank search results
by text relevance, with RRF fusion to preserve engine ranking signals.

The module also exposes the pure scoring helpers that ``sem_reranker`` uses
when it coordinates a combined BM25 + neural ranking.
"""

from __future__ import annotations

import html as html_lib
import logging
import math
import re
import typing as t

from searx import settings  # ty: ignore[unresolved-import]
from searx.plugins import Plugin, PluginInfo  # ty: ignore[unresolved-import]

from ._gate import load_gate, should_escalate
from ._tokenizer import cjk_tokenize
from ._vendor.sparse_search import Result as SparseResult
from ._vendor.sparse_search import SparseIndex, rrf

if t.TYPE_CHECKING:
    from searx.extended_types import SXNG_Request  # ty: ignore[unresolved-import]
    from searx.plugins import PluginCfg  # ty: ignore[unresolved-import]
    from searx.search import SearchWithPlugins  # ty: ignore[unresolved-import]

__version__ = "0.3.0"

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

_COORDINATORS = ("bm25_reranker", "sem_reranker")


class _Ranked(t.Protocol):
    """Minimal shape shared by sparse_search and semantic_search results."""

    doc_id: str
    score: float


_RankedT = t.TypeVar("_RankedT", bound=_Ranked)


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
        """
        return _read_setting(name, _plugin_settings(self.id), integer=integer)

    def post_search(self, request: SXNG_Request, search: SearchWithPlugins) -> None:
        """Rerank results by BM25 relevance fused with original engine ranking.

        Every touch of the SearXNG API stays inside the try block: a SearXNG
        upgrade must degrade to "original order", never break the search.
        """
        try:
            if not _is_coordinator(search, self.id):
                return None

            results_map = search.result_container.main_results_map
            if len(results_map) < 2:
                return None

            query = search.search_query.query
            if not query or not query.strip():
                return None

            self._rerank(query, results_map, search)
            logger.debug(
                "BM25 reranker processed %d results for: %s",
                len(results_map),
                query[:50],
            )
        except Exception:
            logger.exception("BM25 reranking failed, keeping original order")

        return None

    def _rerank(
        self,
        query: str,
        results_map: dict[str, t.Any],
        search: SearchWithPlugins | None = None,
    ) -> None:
        """Core reranking logic.

        Args:
            query: Original search query.
            results_map: Dict of result hash -> MainResult/LegacyResult objects.
            search: Per-request SearXNG object, used only to detect sem_reranker.
        """
        results = list(results_map.values())
        valid_indices = _valid_indices(results)

        if len(valid_indices) < 2:
            return

        scores = score_results(query, results)

        # Engine ranking mirrors the original insertion (engine) order.
        engine_ranking = [
            SparseResult(doc_id=str(i), score=1.0 / (rank + 1))
            for rank, i in enumerate(valid_indices)
        ]

        bm25_ranking = [
            SparseResult(doc_id=str(i), score=t.cast(float, scores[i]))
            for i in rank_results(query, results)
        ]

        lists = [engine_ranking, bm25_ranking]
        weights = [
            self._setting("engine_weight"),
            self._setting("bm25_weight"),
        ]

        gate = load_gate(settings)
        docs = (
            []
            if not gate.enabled
            else [
                _get_text(r, "title") + "\n" + _get_text(r, "content") for r in results
            ]
        )
        escalate, reason = should_escalate(query, docs, scores, valid_indices, gate)
        if escalate:
            logger.info("BM25 escalation gate open: %s", reason)
        else:
            logger.debug("BM25 escalation gate closed: %s", reason)

        neural_scores = _neural_scores(query, results, search) if escalate else None
        if neural_scores is not None and any(
            score is not None for score in neural_scores
        ):
            neural_ranking = [
                SparseResult(doc_id=str(i), score=t.cast(float, neural_scores[i]))
                for i in rank_by_score(neural_scores)
            ]
            lists.append(neural_ranking)
            weights.append(_neural_weight())

        fused = rrf(
            *lists,
            k=int(self._setting("rrf_k", integer=True)),
            weights=weights,
        )

        # RRF sorts by score only; ties fall back to insertion order. Sort by
        # original index instead so the result is deterministic.
        fused.sort(key=lambda r: (-r.score, int(r.doc_id)))

        # Multiply fused scores for URLs matching a configured pattern
        fused = _apply_url_priority(fused, results, self._url_priority)

        _rewrite_positions(results, fused)

        logger.debug("BM25 reranked %d results for query: %s", len(fused), query[:50])


def score_results(
    query: str,
    results: list[t.Any],
    config: t.Mapping[str, t.Any] | None = None,
) -> list[float | None]:
    """Return one BM25 score per result, or ``None`` when none is available.

    The index is built for this request only. A result with no title and no
    content is absent from the index. An indexed result that the search does
    not return also scores ``None``; callers that need the engine ranking use
    :func:`_valid_indices` separately.
    """
    cfg = _plugin_settings("bm25_reranker") if config is None else config

    idx = SparseIndex(
        variant="bm25",
        delta=0.0,
        field_weights={
            "title": _read_setting("title_weight", cfg),
            "content": _read_setting("content_weight", cfg),
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

    scores: list[float | None] = [None] * len(results)

    if len(valid_indices) < 2:
        return scores

    for item in idx.search(query, top_k=len(valid_indices)):
        scores[int(item.doc_id)] = item.score

    return scores


def _valid_indices(results: list[t.Any]) -> list[int]:
    """Return the indices of results with indexable title or content text."""
    valid: list[int] = []
    for i, result in enumerate(results):
        if _get_text(result, "title") or _get_text(result, "content"):
            valid.append(i)
    return valid


def rank_results(
    query: str,
    results: list[t.Any],
    config: t.Mapping[str, t.Any] | None = None,
) -> list[int]:
    """Return result indices sorted by BM25 score, then original order."""
    return rank_by_score(score_results(query, results, config))


def rank_by_score(scores: list[float | None]) -> list[int]:
    """Sort indices by descending score, breaking ties by original index."""
    return sorted(
        (i for i, score in enumerate(scores) if score is not None),
        key=lambda i: (-t.cast(float, scores[i]), i),
    )


def rewrite_positions(results: list[t.Any], order: list[int]) -> None:
    """Give every result a fresh, contiguous position.

    Results absent from ``order`` keep their relative order below the ranked
    ones. ``len(positions)`` is preserved because SearXNG multiplies weight by
    it, which is how a result found by several engines keeps its boost.
    """
    new_positions = {idx: pos for pos, idx in enumerate(order, start=1)}
    next_position = len(new_positions) + 1

    for i, r in enumerate(results):
        old = r["positions"] or []
        count = max(len(old), 1)
        if i in new_positions:
            r["positions"] = [new_positions[i]] * count
        else:
            r["positions"] = [next_position] * count
            next_position += 1


def _rewrite_positions(results: list[t.Any], fused: list[SparseResult]) -> None:
    """Compatibility wrapper: rewrite from a fused result list."""
    rewrite_positions(results, [int(f.doc_id) for f in fused])


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


def _read_setting(
    name: str,
    config: t.Mapping[str, t.Any],
    *,
    integer: bool = False,
) -> float:
    """Read one numeric tunable from a settings mapping with safe fallback."""
    raw = config.get(name)
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
        logger.warning("%s must be >= 1, got %r, using %s", name, raw, DEFAULTS[name])
        return DEFAULTS[name]
    if integer and not value.is_integer():
        logger.warning(
            "%s must be an integer, got %r, using %s", name, raw, DEFAULTS[name]
        )
        return DEFAULTS[name]
    return value


def _is_coordinator(search: t.Any, own_id: str) -> bool:
    """Decide whether this plugin should do the work for this request.

    ``search.user_plugins`` is absent in unit tests and in non-web callers; in
    that case the running plugin coordinates.
    """
    active = getattr(search, "user_plugins", None)
    if active is None:
        return True

    active = list(active)
    configured = _coordinator_setting()

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


def _coordinator_setting() -> str | None:
    block = settings.get("rerankers")
    if not isinstance(block, t.Mapping):
        return None
    value = block.get("coordinator")
    if isinstance(value, str) and value in _COORDINATORS:
        return value
    if value is not None:
        logger.warning(
            "Invalid rerankers.coordinator=%r; expected one of %s",
            value,
            ", ".join(_COORDINATORS),
        )
    return None


def _neural_scores(
    query: str,
    results: list[t.Any],
    search: t.Any,
) -> list[float | None] | None:
    """Ask the semantic package for scores when it is active and installed."""
    active = getattr(search, "user_plugins", None)
    if active is not None and "sem_reranker" not in list(active):
        return None

    try:
        from searxng_sem_reranker import score_neural  # ty: ignore[unresolved-import]
    except ImportError:
        return None

    try:
        return score_neural(query, results)
    except Exception:
        logger.exception("Semantic scoring failed, continuing with BM25 only")
        return None


def _neural_weight() -> float:
    block = settings.get("sem_reranker")
    if not isinstance(block, t.Mapping):
        return 1.0
    fusion = block.get("fusion")
    if not isinstance(fusion, t.Mapping):
        return 1.0
    raw = fusion.get("neural_weight")
    if raw is None or isinstance(raw, bool):
        return 1.0
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        return 1.0
    if not math.isfinite(value) or value < 0:
        return 1.0
    return value


def _apply_url_priority(
    fused: list[_RankedT],
    results: list[t.Any],
    patterns: list[tuple[re.Pattern[str], float]],
) -> list[_RankedT]:
    """Multiply fused scores for URLs matching configured patterns."""
    if not patterns:
        return fused

    for item in fused:
        result = results[int(item.doc_id)]
        try:
            url = result["url"]
        except (KeyError, IndexError, TypeError):
            url = getattr(result, "url", "")
        if not isinstance(url, str):
            continue

        score = item.score
        for pattern, multiplier in patterns:
            if pattern.search(url):
                score *= multiplier
        if not math.isnan(score):
            item.score = score

    return sorted(fused, key=lambda r: (-r.score, int(r.doc_id)))


def _compile_url_priority(raw: t.Any) -> list[tuple[re.Pattern[str], float]]:
    """Compile ``url_priority`` entries, skipping anything malformed."""
    if not isinstance(raw, t.Mapping):
        return []

    compiled: list[tuple[re.Pattern[str], float]] = []
    for pattern, multiplier in raw.items():
        if not isinstance(pattern, str):
            continue
        if isinstance(multiplier, bool):
            logger.warning("url_priority %r: bool is not a multiplier", pattern)
            continue
        try:
            value = float(multiplier)
        except (TypeError, ValueError):
            logger.warning("url_priority %r: %r is not a number", pattern, multiplier)
            continue
        if math.isnan(value) or value < 0:
            logger.warning(
                "url_priority %r: invalid multiplier %r", pattern, multiplier
            )
            continue
        try:
            compiled.append((re.compile(pattern), value))
        except re.error:
            logger.warning("url_priority %r: invalid regular expression", pattern)
    return compiled
