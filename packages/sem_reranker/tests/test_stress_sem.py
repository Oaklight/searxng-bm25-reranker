"""Batch stress tests for the semantic pipeline.

These are excluded from the default test run.  Run them with ``make stress``.
"""

from __future__ import annotations

import concurrent.futures
import threading
import time

import pytest

import searxng_sem_reranker as sem
from searxng_sem_reranker._config import load_config
from searxng_sem_reranker._providers import ProviderError

pytestmark = pytest.mark.stress


def make_config(*, embedding: bool = True, rerank: bool = False):
    return load_config(
        {
            "sem_reranker": {
                "embedding": {
                    "enabled": embedding,
                    "format": "openai",
                    "url": "https://example.test/embeddings",
                    "model": "test-model",
                    "max_tokens": 128,
                },
                "rerank": {
                    "enabled": rerank,
                    "format": "jina",
                    "url": "https://example.test/rerank",
                    "model": "test-model",
                    "top_k": 10,
                },
                "cache": {"ttl": 300, "max_entries": 512},
            }
        }
    )


def results(count: int) -> list[dict]:
    return [
        {
            "title": f"python asyncio guide {i}",
            "content": "asyncio tutorial event loop " * 10,
            "url": f"https://example.com/{i}",
            "positions": [1],
        }
        for i in range(count)
    ]


def fake_embed(stage, texts, *, query):
    return [[float(len(text) % 17), 1.0, 0.5] for text in texts]


def fake_rerank(stage, query, docs):
    return [0.5] * len(docs)


def _explode(*args, **kwargs):
    raise AssertionError("provider must not be called")


def test_large_embedding_and_rerank(monkeypatch) -> None:
    monkeypatch.setattr(sem, "embed_texts", fake_embed)
    monkeypatch.setattr(sem, "rerank_documents", fake_rerank)
    data = results(1000)

    start = time.perf_counter()
    scores = sem.score_neural(
        "python asyncio", data, make_config(embedding=True, rerank=True)
    )
    elapsed = time.perf_counter() - start

    assert len(scores) == len(data)
    assert all(score is not None for score in scores)
    assert elapsed < 10.0


def test_cache_avoids_repeated_calls(monkeypatch) -> None:
    calls = {"embed": 0, "rerank": 0}

    def embed(stage, texts, *, query):
        calls["embed"] += 1
        return fake_embed(stage, texts, query=query)

    def rerank(stage, query, docs):
        calls["rerank"] += 1
        return fake_rerank(stage, query, docs)

    monkeypatch.setattr(sem, "embed_texts", embed)
    monkeypatch.setattr(sem, "rerank_documents", rerank)
    cfg = make_config(embedding=True, rerank=True)

    sem.score_neural("python asyncio", results(200), cfg)
    sem.score_neural("python asyncio", results(200), cfg)

    assert calls == {"embed": 1, "rerank": 1}


def test_concurrent_scoring(monkeypatch) -> None:
    lock = threading.Lock()
    calls = 0

    def embed(stage, texts, *, query):
        nonlocal calls
        with lock:
            calls += 1
        return fake_embed(stage, texts, query=query)

    monkeypatch.setattr(sem, "embed_texts", embed)
    monkeypatch.setattr(sem, "rerank_documents", fake_rerank)
    cfg = make_config(embedding=True)

    def run(index: int) -> int:
        return len(sem.score_neural(f"query {index}", results(50), cfg))

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        lengths = list(pool.map(run, range(32)))

    assert lengths == [50] * 32
    assert calls == 32


def test_failure_storm_never_raises(monkeypatch) -> None:
    def fail(stage, texts, *, query):
        raise ProviderError("HTTP 500", status_code=500, retryable=True)

    monkeypatch.setattr(sem, "embed_texts", fail)
    monkeypatch.setattr(sem, "rerank_documents", fail)
    cfg = make_config(embedding=True)

    for _ in range(50):
        assert sem.score_neural("query", results(50), cfg) == [None] * 50


def test_zero_vectors_do_not_raise(monkeypatch) -> None:
    monkeypatch.setattr(
        sem, "embed_texts", lambda stage, texts, *, query: [[0.0, 0.0] for _ in texts]
    )
    monkeypatch.setattr(sem, "rerank_documents", fake_rerank)

    scores = sem.score_neural("query", results(20), make_config(embedding=True))

    assert len(scores) == 20


@pytest.mark.parametrize(
    "block",
    [
        {},
        {
            "embedding": {
                "enabled": True,
                "format": "nope",
                "url": "ftp://bad",
                "model": "",
            }
        },
        {
            "embedding": {
                "enabled": True,
                "format": "openai",
                "url": "https://example.test/embeddings",
                "model": "m",
                "api_key_env": "MISSING_KEY_FOR_STRESS",
            }
        },
        {
            "rerank": {
                "enabled": True,
                "format": "jina",
                "url": "https://example.test/rerank",
                "model": "m",
                "api_key_env": "MISSING_KEY_FOR_STRESS",
                "top_k": -5,
            }
        },
    ],
)
def test_config_fuzz_never_raises(monkeypatch, block: dict) -> None:
    monkeypatch.setattr(sem, "embed_texts", _explode)
    monkeypatch.setattr(sem, "rerank_documents", _explode)
    monkeypatch.delenv("MISSING_KEY_FOR_STRESS", raising=False)

    cfg = load_config({"sem_reranker": block})

    assert sem.score_neural("query", results(5), cfg) == [None] * 5
