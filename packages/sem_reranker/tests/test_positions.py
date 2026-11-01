"""Position rewrite invariants shared with the BM25 plugin."""

from __future__ import annotations

from searxng_sem_reranker._positions import rewrite_positions


def make_result(positions: list[int] | None = None) -> dict:
    return {"title": "t", "content": "c", "positions": positions or [1]}


def test_positions_are_unique_and_contiguous() -> None:
    results = [make_result(), make_result(), make_result()]
    rewrite_positions(results, [2, 0, 1])
    assert [result["positions"][0] for result in results] == [2, 3, 1]
    assert sorted(result["positions"][0] for result in results) == [1, 2, 3]


def test_unranked_results_are_pushed_below() -> None:
    results = [make_result(), make_result(), make_result()]
    rewrite_positions(results, [2])
    assert results[2]["positions"] == [1]
    assert results[0]["positions"] == [2]
    assert results[1]["positions"] == [3]


def test_multi_engine_length_is_preserved() -> None:
    results = [make_result([1, 1, 1]), make_result()]
    rewrite_positions(results, [1, 0])
    assert results[0]["positions"] == [2, 2, 2]
    assert results[1]["positions"] == [1]


def test_empty_positions_get_one() -> None:
    results = [make_result([]), make_result([])]
    rewrite_positions(results, [1, 0])
    assert results[0]["positions"] == [2]
    assert results[1]["positions"] == [1]
