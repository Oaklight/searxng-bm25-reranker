"""Tests for the url_priority score multipliers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import searx

from searxng_bm25_reranker import (
    SXNGPlugin,
    _apply_url_priority,
    _compile_url_priority,
    _plugin_settings,
)
from searxng_bm25_reranker._vendor.sparse_search import Result as SparseResult


@pytest.fixture(autouse=True)
def _clean_settings() -> None:
    """Keep the stubbed searx.settings from leaking between tests."""
    searx.settings.clear()


def make_result(url: str, title: str = "alpha", content: str = "alpha") -> dict:
    """Build a result in the []-accessible shape SearXNG hands the plugin."""
    return {"url": url, "title": title, "content": content, "positions": [1]}


def make_plugin(url_priority: object = None) -> SXNGPlugin:
    if url_priority is not None:
        searx.settings["bm25_reranker"] = {"url_priority": url_priority}
    return SXNGPlugin(SimpleNamespace())


def ranked_by(urls: list[str], url_priority: object) -> list[str]:
    """Run urls through the pattern multipliers and return the new order."""
    results = [make_result(u) for u in urls]
    fused = [SparseResult(doc_id=str(i), score=1.0 / (i + 1)) for i in range(len(urls))]
    patterns = _compile_url_priority(url_priority)
    ordered = _apply_url_priority(fused, results, patterns)
    return [results[int(r.doc_id)]["url"] for r in ordered]


class TestPluginSettings:
    def test_reads_the_top_level_block_keyed_by_id(self) -> None:
        searx.settings["bm25_reranker"] = {"url_priority": {"a": 2}}
        assert _plugin_settings("bm25_reranker") == {"url_priority": {"a": 2}}

    def test_absent_block_is_empty(self) -> None:
        assert _plugin_settings("bm25_reranker") == {}

    @pytest.mark.parametrize("block", [None, "nonsense", 3, ["a"]])
    def test_non_mapping_block_is_empty(self, block: object) -> None:
        searx.settings["bm25_reranker"] = block
        assert _plugin_settings("bm25_reranker") == {}


class TestCompile:
    def test_absent_value_compiles_to_nothing(self) -> None:
        assert _compile_url_priority(None) == []

    def test_non_mapping_value_compiles_to_nothing(self) -> None:
        assert _compile_url_priority("nope") == []

    def test_invalid_regex_is_skipped(self) -> None:
        assert _compile_url_priority({"(unclosed": 2.0}) == []

    @pytest.mark.parametrize("bad", ["high", None, [1], {"a": 1}])
    def test_non_numeric_multiplier_is_skipped(self, bad: object) -> None:
        assert _compile_url_priority({"example.com": bad}) == []

    @pytest.mark.parametrize("bad", [True, False])
    def test_boolean_multiplier_is_skipped(self, bad: bool) -> None:
        """A bool is a legal float() argument, so it needs its own guard.

        False is the one that bites: it becomes 0.0, which buries the domain
        rather than switching the rule off, and YAML's false/no/off all
        parse to it.
        """
        assert _compile_url_priority({"example.com": bad}) == []

    def test_negative_multiplier_is_skipped(self) -> None:
        assert _compile_url_priority({"example.com": -1.0}) == []

    def test_zero_multiplier_is_kept(self) -> None:
        assert len(_compile_url_priority({"example.com": 0})) == 1

    def test_fractional_multiplier_keeps_its_fraction(self) -> None:
        """0.1 must demote, not truncate to 0 (guards float() vs int())."""
        results = [make_result("https://a.example/1")]
        fused = [SparseResult(doc_id="0", score=1.0)]
        patterns = _compile_url_priority({"a\\.example": 0.1})

        assert _apply_url_priority(fused, results, patterns)[0].score == pytest.approx(
            0.1
        )

    def test_quoted_multiplier_is_accepted(self) -> None:
        """settings.yml quoting a number must not silently drop the entry."""
        assert [m for _, m in _compile_url_priority({"example\\.com": "0.5"})] == [0.5]

    def test_nan_multiplier_is_skipped(self) -> None:
        """NaN is not negative, so it needs its own guard."""
        assert _compile_url_priority({"example\\.com": float("nan")}) == []

    def test_infinite_multiplier_is_kept(self) -> None:
        assert len(_compile_url_priority({"example\\.com": float("inf")})) == 1

    def test_valid_entries_survive_alongside_invalid_ones(self) -> None:
        compiled = _compile_url_priority({"good\\.org": 2, "(bad": 3})
        assert [p.pattern for p, _ in compiled] == ["good\\.org"]


class TestApply:
    def test_no_patterns_leaves_the_ranking_alone(self) -> None:
        urls = ["https://a.example/1", "https://b.example/2"]
        assert ranked_by(urls, None) == urls

    def test_no_patterns_returns_the_very_same_list(self) -> None:
        """The docstring promises the input list back, so pin the identity."""
        results = [make_result("https://a.example/1")]
        fused = [SparseResult(doc_id="0", score=1.0)]

        assert _apply_url_priority(fused, results, []) is fused

    def test_compounded_nan_falls_back_to_the_unadjusted_score(self) -> None:
        """inf and 0.0 each pass the input guard, but inf * 0.0 is NaN.

        Without the product check, one such pair makes the sort
        non-transitive and the order arbitrary.
        """
        results = [make_result(f"https://spam.example/{i}") for i in range(4)]
        fused = [
            SparseResult(doc_id=str(i), score=s)
            for i, s in enumerate([1.0, 0.9, 0.8, 0.7])
        ]
        patterns = _compile_url_priority({"spam": float("inf"), "spam\\.example": 0.0})

        got = _apply_url_priority(fused, results, patterns)

        assert [r.score for r in got] == [1.0, 0.9, 0.8, 0.7]
        assert [r.doc_id for r in got] == ["0", "1", "2", "3"]

    def test_matching_url_is_promoted(self) -> None:
        got = ranked_by(
            ["https://plain.example/1", "https://good.example/2"],
            {"good\\.example": 10.0},
        )
        assert got == ["https://good.example/2", "https://plain.example/1"]

    def test_multiplier_below_one_demotes(self) -> None:
        got = ranked_by(
            ["https://spam.example/1", "https://plain.example/2"],
            {"spam\\.example": 0.01},
        )
        assert got[-1] == "https://spam.example/1"

    def test_zero_multiplier_pushes_to_the_bottom(self) -> None:
        got = ranked_by(
            ["https://spam.example/1", "https://plain.example/2"],
            {"spam\\.example": 0.0},
        )
        assert got[-1] == "https://spam.example/1"

    def test_overlapping_patterns_multiply(self) -> None:
        """A URL matching two entries is multiplied by both, not by the last.

        Asserting the score rather than the order is deliberate: with these
        inputs "multiply by both" (1.0 * 2 * 3) and "keep the last match"
        (1.0 * 3) put the same URL first, so an order assertion would not
        notice the difference.
        """
        results = [make_result("https://host.example/1")]
        fused = [SparseResult(doc_id="0", score=1.0)]
        patterns = _compile_url_priority({"host\\.example": 2.0, "example": 3.0})

        assert _apply_url_priority(fused, results, patterns)[0].score == pytest.approx(
            6.0
        )

    def test_compounding_can_change_the_order(self) -> None:
        """Compounding must be able to outrank a higher incoming score."""
        results = [
            make_result("https://host.example/1"),
            make_result("https://other.example/2"),
        ]
        fused = [
            SparseResult(doc_id="0", score=0.5),
            SparseResult(doc_id="1", score=1.0),
        ]
        patterns = _compile_url_priority({"host\\.example": 3.0, "example": 2.0})

        got = [
            results[int(r.doc_id)]["url"]
            for r in _apply_url_priority(fused, results, patterns)
        ]

        # 0.5 * 3 * 2 = 3.0 clears 1.0 * 2 = 2.0; keeping only the last match
        # would leave 1.5 against 2.0 and invert this.
        assert got == ["https://host.example/1", "https://other.example/2"]

    def test_non_matching_results_keep_their_relative_order(self) -> None:
        urls = ["https://a.example/1", "https://b.example/2", "https://c.example/3"]
        assert ranked_by(urls, {"nowhere\\.invalid": 5.0}) == urls

    def test_results_without_a_url_are_untouched(self) -> None:
        results = [{"title": "alpha", "content": "alpha", "positions": [1]}]
        fused = [SparseResult(doc_id="0", score=1.0)]
        patterns = _compile_url_priority({"example": 2.0})
        assert _apply_url_priority(fused, results, patterns)[0].score == 1.0


class TestWiring:
    def test_rerank_applies_configured_multipliers(self) -> None:
        """End-to-end: the boost flips an engine-favoured result down."""
        plugin = make_plugin({"boosted\\.example": 5.0, "demoted\\.example": 0.1})
        results = {
            "0": make_result("https://demoted.example/a"),
            "1": make_result("https://boosted.example/b"),
        }
        plugin._rerank("alpha", results)

        assert results["0"]["positions"] == [2]
        assert results["1"]["positions"] == [1]

    def test_rerank_without_configuration_keeps_engine_order(self) -> None:
        plugin = make_plugin()
        results = {
            "0": make_result("https://a.example/a", title="alpha alpha"),
            "1": make_result("https://b.example/b", title="alpha beta"),
        }
        plugin._rerank("alpha", results)

        assert results["0"]["positions"] == [1]
