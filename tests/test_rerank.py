"""Reranking tests, focused on the two P0 defects and the new safeguards."""

from __future__ import annotations

from types import SimpleNamespace

from searx import settings
from searx.plugins import PluginCfg

from searxng_bm25_reranker import DEFAULTS, SXNGPlugin, _get_text


def make_result(title: str = "", content: str = "", n_positions: int = 1) -> dict:
    """Build a result in the []-accessible shape SearXNG hands the plugin."""
    return {"title": title, "content": content, "positions": [1] * n_positions}


def make_plugin(**overrides: object) -> SXNGPlugin:
    """Build a plugin with the given tunables in its settings block.

    Tunables live in a top-level ``bm25_reranker:`` block in settings.yml, not
    on the PluginCfg -- that dataclass only carries ``active``. Each call
    replaces the block so tests cannot leak state into each other.
    """
    settings.clear()
    settings["bm25_reranker"] = dict(overrides) if overrides else {}
    return SXNGPlugin(PluginCfg())


def make_search(results: dict, query: str) -> SimpleNamespace:
    return SimpleNamespace(
        result_container=SimpleNamespace(main_results_map=results),
        search_query=SimpleNamespace(query=query),
    )


def rerank(results: list[dict], query: str, **settings: object) -> None:
    make_plugin(**settings)._rerank(query, {str(i): r for i, r in enumerate(results)})


def positions(results: list[dict]) -> list[int]:
    return [r["positions"][0] for r in results]


class TestPositionRewrite:
    def test_every_result_gets_a_unique_position(self) -> None:
        """Regression: text-less results kept raw engine positions and collided.

        Before the fix this produced positions [1, 1, 2, 3] -- a duplicate top
        score, decided by SearXNG's tie-break rather than by relevance.
        """
        results = [
            make_result(),
            make_result("Cafe da manha", "Padaria e cafe"),
            make_result("Musica", "Sinfonias"),
            make_result("Esportes", "Futebol"),
        ]
        rerank(results, "pingueiros gol")

        got = positions(results)
        assert len(set(got)) == len(results)
        assert sorted(got) == [1, 2, 3, 4]

    def test_textless_result_is_pushed_below_the_reranked_ones(self) -> None:
        results = [
            make_result(),
            make_result("pingueiros", "quadro de pingueiros"),
            make_result("Musica", "Sinfonias"),
        ]
        rerank(results, "pingueiros")
        assert results[0]["positions"] == [3]

    def test_multi_engine_boost_is_preserved(self) -> None:
        results = [
            make_result("pingueiros", "quadro", n_positions=3),
            make_result("Musica", "Sinfonias"),
            make_result("Futebol", "Gols"),
        ]
        rerank(results, "pingueiros")
        # SearXNG multiplies weight by len(positions); the list length must stay.
        assert len(results[0]["positions"]) == 3
        assert len(set(results[0]["positions"])) == 1

    def test_missing_positions_key_still_gets_one(self) -> None:
        results = [
            {"title": "pingueiros", "content": "quadro", "positions": []},
            {"title": "Musica", "content": "Sinfonias", "positions": []},
        ]
        rerank(results, "pingueiros")
        assert sorted(positions(results)) == [1, 2]


class TestRerankQuality:
    def test_relevant_result_is_promoted(self) -> None:
        results = [
            make_result("Receita de bolo", "bolo de cenoura com chocolate"),
            make_result("Previsao do tempo", "chuva a tarde"),
            make_result(
                "python asyncio tutorial", "guia completo de asyncio em python"
            ),
            make_result("Cardapio", "arroz feijao salada"),
        ]
        rerank(results, "python asyncio tutorial")
        assert results[2]["positions"][0] == 1

    def test_query_matching_nothing_keeps_engine_order(self) -> None:
        results = [
            make_result("alpha", "first document"),
            make_result("beta", "second document"),
            make_result("gamma", "third document"),
        ]
        rerank(results, "zzzzqqqq")
        assert positions(results) == [1, 2, 3]

    def test_single_indexable_result_is_a_noop(self) -> None:
        results = [make_result("pingueiros", "quadro"), make_result()]
        rerank(results, "pingueiros")
        assert positions(results) == [1, 1]  # unchanged: fewer than two indexed


class TestHtmlStripping:
    def test_markup_never_reaches_the_tokens(self) -> None:
        html = '<div class="r"><a href="https://www.example.com/x">Guide</a></div>'
        cleaned = _get_text({"content": html}, "content")
        assert "Guide" in cleaned
        assert "div" not in cleaned
        assert "href" not in cleaned
        assert "example" not in cleaned

    def test_entities_are_decoded(self) -> None:
        assert _get_text({"content": "AT&amp;T"}, "content") == "AT&T"

    def test_missing_or_non_string_fields_are_safe(self) -> None:
        assert _get_text({"content": None}, "content") == ""
        assert _get_text({}, "content") == ""
        assert _get_text({"content": ["a", "b"]}, "content") == ""

    def test_plain_text_is_untouched(self) -> None:
        assert _get_text({"content": "python & co"}, "content") == "python & co"

    def test_comparisons_survive_tag_stripping(self) -> None:
        """Regression: `"<" in val and ">" in val` ate plain text.

        `a < b and c > d` came back as `a   d`. Only an actual tag shape is
        removed now.
        """
        assert _get_text({"content": "a < b and c > d"}, "content") == "a < b and c > d"

    def test_half_comparison_keeps_the_angle_bracket(self) -> None:
        assert _get_text({"content": "x < 5"}, "content") == "x < 5"


class TestSettings:
    def test_overrides_are_used(self) -> None:
        plugin = make_plugin(rrf_k=5, title_weight=9.0)
        assert plugin._setting("rrf_k") == 5.0
        assert plugin._setting("title_weight") == 9.0

    def test_missing_setting_uses_default(self) -> None:
        assert make_plugin()._setting("bm25_weight") == DEFAULTS["bm25_weight"]

    def test_bad_values_fall_back_to_default(self) -> None:
        plugin = make_plugin(rrf_k="not-a-number", title_weight=-3)
        assert plugin._setting("rrf_k") == DEFAULTS["rrf_k"]
        assert plugin._setting("title_weight") == DEFAULTS["title_weight"]

    def test_fractional_rrf_k_is_rejected(self) -> None:
        """Regression: `int(self._setting("rrf_k"))` truncated 0.5 to 0.

        ``rrf()`` then raised and the rerank was skipped. Anything below 1 is
        rejected before it can be truncated.
        """
        plugin = make_plugin(rrf_k=0.5)
        assert plugin._setting("rrf_k") == DEFAULTS["rrf_k"]

    def test_tunables_read_the_plugin_settings_block(self) -> None:
        """Regression: tunables must come from `settings[bm25_reranker]`.

        ``PluginCfg`` only carries ``active``, and ``Plugin.__init__``
        discards the object -- there is no ``self.plg_cfg`` on a real
        instance.
        """
        plugin = make_plugin(engine_weight=3.0)
        assert not hasattr(plugin, "plg_cfg")
        assert plugin._setting("engine_weight") == 3.0


class TestPostSearchRobustness:
    def test_broken_search_object_does_not_raise(self) -> None:
        """Regression: the SearXNG API access sat outside the try/except.

        Any SearXNG API change raised straight out of post_search and broke
        the search instead of degrading to the original order.
        """
        plugin = make_plugin()
        assert plugin.post_search(object(), SimpleNamespace()) is None

    def test_fewer_than_two_results_is_a_noop(self) -> None:
        plugin = make_plugin()
        results = {"a": make_result("x", "y")}
        assert plugin.post_search(object(), make_search(results, "query")) is None
        assert results["a"]["positions"] == [1]

    def test_blank_query_is_a_noop(self) -> None:
        results = {str(i): make_result(f"t{i}", f"c{i}") for i in range(3)}
        before = {k: list(v["positions"]) for k, v in results.items()}
        plugin = make_plugin()
        assert plugin.post_search(object(), make_search(results, "   ")) is None
        assert {k: list(v["positions"]) for k, v in results.items()} == before

    def test_happy_path_reranks(self) -> None:
        results = {
            "0": make_result("Receita de bolo", "bolo de cenoura"),
            "1": make_result("Previsao", "chuva"),
            "2": make_result("python asyncio tutorial", "guia de asyncio em python"),
        }
        plugin = make_plugin()
        assert (
            plugin.post_search(object(), make_search(results, "python asyncio")) is None
        )
        assert results["2"]["positions"][0] == 1
