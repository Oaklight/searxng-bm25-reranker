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

    def test_negative_multiplier_is_skipped(self) -> None:
        assert _compile_url_priority({"example.com": -1.0}) == []

    def test_zero_multiplier_is_kept(self) -> None:
        assert len(_compile_url_priority({"example.com": 0})) == 1

    def test_valid_entries_survive_alongside_invalid_ones(self) -> None:
        compiled = _compile_url_priority({"good\\.org": 2, "(bad": 3})
        assert [p.pattern for p, _ in compiled] == ["good\\.org"]


class TestApply:
    def test_no_patterns_leaves_the_ranking_alone(self) -> None:
        urls = ["https://a.example/1", "https://b.example/2"]
        assert ranked_by(urls, None) == urls

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
        """A URL matching two entries is multiplied by both."""
        got = ranked_by(
            ["https://host.example/1", "https://other.example/2"],
            {"host\\.example": 2.0, "example": 3.0},
        )
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
