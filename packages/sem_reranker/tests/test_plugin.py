"""Plugin hook behavior and exception boundaries."""

from __future__ import annotations

from types import SimpleNamespace

import searx

import searxng_sem_reranker as sem
from searxng_sem_reranker import SXNGPlugin


def make_results() -> dict[str, dict]:
    return {
        "0": {
            "title": "alpha",
            "content": "",
            "url": "https://a.test",
            "positions": [1],
        },
        "1": {
            "title": "beta",
            "content": "",
            "url": "https://b.test",
            "positions": [1],
        },
        "2": {
            "title": "gamma",
            "content": "",
            "url": "https://c.test",
            "positions": [1],
        },
    }


def make_search(results: dict, query: str, plugins: list[str] | None = None):
    return SimpleNamespace(
        result_container=SimpleNamespace(main_results_map=results),
        search_query=SimpleNamespace(query=query),
        user_plugins=plugins if plugins is not None else ["sem_reranker"],
    )


def positions(results: dict) -> dict[str, int]:
    return {key: value["positions"][0] for key, value in results.items()}


def test_coordinator_reranks(monkeypatch) -> None:
    monkeypatch.setattr(sem, "score_neural", lambda q, r, c=None: [0.1, 0.9, 0.2])
    results = make_results()

    SXNGPlugin(SimpleNamespace()).post_search(object(), make_search(results, "query"))

    assert positions(results) == {"0": 2, "1": 1, "2": 3}


def test_non_coordinator_leaves_positions_untouched(monkeypatch) -> None:
    searx.settings["rerankers"] = {"coordinator": "bm25_reranker"}
    monkeypatch.setattr(sem, "score_neural", _explode)
    results = make_results()

    SXNGPlugin(SimpleNamespace()).post_search(
        object(), make_search(results, "query", ["bm25_reranker", "sem_reranker"])
    )

    assert positions(results) == {"0": 1, "1": 1, "2": 1}


def test_fewer_than_two_results_is_a_noop(monkeypatch) -> None:
    monkeypatch.setattr(sem, "score_neural", _explode)
    results = {"0": {"title": "alpha", "content": "", "positions": [1]}}

    SXNGPlugin(SimpleNamespace()).post_search(object(), make_search(results, "query"))

    assert results["0"]["positions"] == [1]


def test_blank_query_is_a_noop(monkeypatch) -> None:
    monkeypatch.setattr(sem, "score_neural", _explode)
    results = make_results()

    SXNGPlugin(SimpleNamespace()).post_search(object(), make_search(results, "   "))

    assert positions(results) == {"0": 1, "1": 1, "2": 1}


def test_broken_search_object_does_not_raise() -> None:
    assert (
        SXNGPlugin(SimpleNamespace()).post_search(object(), SimpleNamespace()) is None
    )


def test_scoring_exception_keeps_original_order(monkeypatch) -> None:
    monkeypatch.setattr(sem, "score_neural", _explode)
    results = make_results()

    SXNGPlugin(SimpleNamespace()).post_search(object(), make_search(results, "query"))

    assert positions(results) == {"0": 1, "1": 1, "2": 1}


def test_all_scores_none_leaves_positions_untouched(monkeypatch) -> None:
    monkeypatch.setattr(sem, "score_neural", lambda q, r, c=None: [None, None, None])
    results = make_results()

    SXNGPlugin(SimpleNamespace()).post_search(object(), make_search(results, "query"))

    assert positions(results) == {"0": 1, "1": 1, "2": 1}


def _explode(*args, **kwargs):
    raise AssertionError("must not be called")
