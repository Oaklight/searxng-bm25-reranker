"""Batch stress tests for the BM25 plugin.

These are excluded from the default test run.  Run them with ``make stress``.
"""

from __future__ import annotations

import time

import pytest

from searxng_bm25_reranker import _get_text, _url_text, rewrite_positions, score_results

pytestmark = pytest.mark.stress


def make_results(count: int, *, cjk: bool = False) -> list[dict]:
    if cjk:
        return [
            {
                "title": f"Python 异步编程指南 {i}",
                "content": "asyncio 协程 事件循环 " * 12,
                "url": f"https://example.com/zh/异步/{i}",
                "positions": [1],
            }
            for i in range(count)
        ]
    return [
        {
            "title": f"python asyncio guide {i}",
            "content": "asyncio tutorial event loop " * 12,
            "url": f"https://example.com/python/asyncio/{i}",
            "positions": [1],
        }
        for i in range(count)
    ]


@pytest.mark.parametrize("count", [100, 500, 1000])
def test_bm25_scales(count: int) -> None:
    results = make_results(count)
    start = time.perf_counter()
    scores = score_results("python asyncio", results)
    elapsed = time.perf_counter() - start

    assert sum(score is not None for score in scores) == count
    assert elapsed < 10.0


def test_positions_are_a_permutation_on_large_input() -> None:
    results = make_results(1000)
    rewrite_positions(results, list(range(len(results))))

    first = [result["positions"][0] for result in results]
    assert sorted(first) == list(range(1, len(results) + 1))


def test_cjk_large_input() -> None:
    results = make_results(500, cjk=True)
    scores = score_results("python 异步 协程", results)

    assert sum(score is not None for score in scores) == len(results)


@pytest.mark.parametrize(
    "payload",
    [
        "<a " + '"' * 50000,
        "<a " + "x" * 50000,
        "<div>" * 20000,
        "word " * 20000,
    ],
)
def test_html_stripping_stays_fast(payload: str) -> None:
    start = time.perf_counter()
    _get_text({"content": payload}, "content")

    assert time.perf_counter() - start < 2.0


@pytest.mark.parametrize(
    "result",
    [
        {},
        {"title": None, "content": None, "url": None},
        {"title": ["a"], "content": {"b": 1}, "url": 42},
        {"title": "a", "content": "b", "url": "not a url"},
        {"title": "a", "content": "b", "url": "https://[::1"},
    ],
)
def test_malformed_results_never_raise(result: dict) -> None:
    assert isinstance(_get_text(result, "title"), str)
    assert isinstance(_url_text(_get_text(result, "url")), str)

    scores = score_results("query", [result, {"title": "query", "content": "x"}])

    assert len(scores) == 2
