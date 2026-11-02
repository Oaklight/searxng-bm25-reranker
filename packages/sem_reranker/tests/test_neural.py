"""Neural scoring pipeline tests, with the provider boundary stubbed."""

from __future__ import annotations

import pytest

import searxng_sem_reranker as sem
from searxng_sem_reranker._config import Config, load_config
from searxng_sem_reranker._providers import ProviderError


def make_config(
    *,
    embedding: bool = False,
    rerank: bool = False,
    top_k: int = 2,
    max_chars: int = 4000,
    max_tokens: int = 0,
    query_prefix: str = "",
    document_prefix: str = "",
    model: str = "test-model",
    url: str = "https://example.test/embeddings",
    api_key_env: str = "",
) -> Config:
    return load_config(
        {
            "sem_reranker": {
                "embedding": {
                    "enabled": embedding,
                    "format": "openai",
                    "api_key_env": api_key_env,
                    "url": url,
                    "model": model,
                    "max_chars": max_chars,
                    "max_tokens": max_tokens,
                    "query_prefix": query_prefix,
                    "document_prefix": document_prefix,
                },
                "rerank": {
                    "enabled": rerank,
                    "format": "jina",
                    "api_key_env": api_key_env,
                    "url": "https://example.test/rerank",
                    "model": model,
                    "top_k": top_k,
                    "max_chars": max_chars,
                },
            }
        }
    )


def results() -> list[dict]:
    return [
        {"title": "a", "content": "", "positions": [1]},
        {"title": "bb", "content": "", "positions": [1]},
        {"title": "ccc", "content": "", "positions": [1]},
    ]


def fake_vectors(stage, texts, *, query):
    return [[float(len(text)), 1.0] for text in texts]


def test_stages_off_make_no_provider_calls(monkeypatch) -> None:
    monkeypatch.setattr(sem, "embed_texts", _explode)
    monkeypatch.setattr(sem, "rerank_documents", _explode)
    assert sem.score_neural("query", results(), make_config()) == [None, None, None]


def test_embedding_only_scores_every_text(monkeypatch) -> None:
    monkeypatch.setattr(sem, "embed_texts", fake_vectors)
    monkeypatch.setattr(sem, "rerank_documents", _explode)

    scores = sem.score_neural("query", results(), make_config(embedding=True))

    assert scores[0] == pytest.approx(6.0 / (26**0.5 * 2**0.5))
    assert scores[1] == pytest.approx(11.0 / (26**0.5 * 5**0.5))
    assert scores[2] == pytest.approx(16.0 / (26**0.5 * 10**0.5))


def test_empty_text_scores_none(monkeypatch) -> None:
    monkeypatch.setattr(sem, "embed_texts", fake_vectors)
    monkeypatch.setattr(sem, "rerank_documents", _explode)

    data = results()
    data[1]["title"] = ""
    scores = sem.score_neural("query", data, make_config(embedding=True))

    assert scores[1] is None


def test_embedding_prefixes_reach_the_provider(monkeypatch) -> None:
    seen: list[list[str]] = []

    def capture(stage, texts, *, query):
        seen.append(list(texts))
        return fake_vectors(stage, texts, query=query)

    monkeypatch.setattr(sem, "embed_texts", capture)
    monkeypatch.setattr(sem, "rerank_documents", _explode)

    sem.score_neural(
        "hello",
        results(),
        make_config(
            embedding=True,
            query_prefix="query: ",
            document_prefix="passage: ",
        ),
    )

    assert seen == [["query: hello", "passage: a", "passage: bb", "passage: ccc"]]


def test_max_tokens_bounds_the_embedding_payload(monkeypatch) -> None:
    seen: list[str] = []

    def capture(stage, texts, *, query):
        seen.extend(texts)
        return fake_vectors(stage, texts, query=query)

    monkeypatch.setattr(sem, "embed_texts", capture)
    monkeypatch.setattr(sem, "rerank_documents", _explode)

    long_text = "word " * 300
    data = [
        {"title": long_text, "content": "", "positions": [1]},
        {"title": "short", "content": "", "positions": [1]},
    ]
    sem.score_neural("q", data, make_config(embedding=True, max_tokens=10))

    assert len(seen[1]) <= 40
    assert seen[2] == "short"


def test_truncate_tokens_is_script_aware() -> None:
    assert sem._truncate_tokens("中" * 100, 10) == "中" * 10
    assert len(sem._truncate_tokens("word " * 300, 10)) <= 40


def test_rerank_only_scores_top_k_in_engine_order(monkeypatch) -> None:
    monkeypatch.setattr(sem, "embed_texts", _explode)
    monkeypatch.setattr(
        sem,
        "rerank_documents",
        lambda stage, query, docs: [0.9, 0.1][: len(docs)],
    )

    scores = sem.score_neural("query", results(), make_config(rerank=True, top_k=2))

    assert scores == [0.9, 0.1, None]


def test_both_stages_rerank_the_embedding_top_k(monkeypatch) -> None:
    monkeypatch.setattr(sem, "embed_texts", fake_vectors)
    seen: list[list[str]] = []

    def fake_rerank(stage, query, docs):
        seen.append(docs)
        return [0.9, 0.1]

    monkeypatch.setattr(sem, "rerank_documents", fake_rerank)

    scores = sem.score_neural(
        "query", results(), make_config(embedding=True, rerank=True, top_k=2)
    )

    # "ccc" and "bb" are the two closest embedding vectors; "a" keeps its
    # embedding score.
    assert seen == [["ccc", "bb"]]
    assert scores[0] is not None
    assert scores[1] == 0.1
    assert scores[2] == 0.9


def test_embedding_failure_falls_back_to_rerank(monkeypatch) -> None:
    monkeypatch.setattr(sem, "embed_texts", _raise_provider_error)
    monkeypatch.setattr(
        sem,
        "rerank_documents",
        lambda stage, query, docs: [0.9, 0.1][: len(docs)],
    )

    scores = sem.score_neural(
        "query", results(), make_config(embedding=True, rerank=True, top_k=2)
    )

    assert scores == [0.9, 0.1, None]


def test_rerank_failure_keeps_embedding_scores(monkeypatch) -> None:
    monkeypatch.setattr(sem, "embed_texts", fake_vectors)
    monkeypatch.setattr(sem, "rerank_documents", _raise_provider_error)

    scores = sem.score_neural(
        "query", results(), make_config(embedding=True, rerank=True, top_k=2)
    )

    assert all(score is not None for score in scores)


def test_stages_keep_their_own_url_and_key(monkeypatch) -> None:
    cfg = load_config(
        {
            "sem_reranker": {
                "embedding": {
                    "enabled": True,
                    "format": "openai",
                    "url": "https://embed.example/embeddings",
                    "model": "embed-model",
                    "api_key_env": "EMBED_KEY",
                },
                "rerank": {
                    "enabled": True,
                    "format": "jina",
                    "url": "https://rerank.example/rerank",
                    "model": "rerank-model",
                    "api_key_env": "RERANK_KEY",
                    "top_k": 2,
                },
            }
        }
    )
    monkeypatch.setenv("EMBED_KEY", "embed-secret")
    monkeypatch.setenv("RERANK_KEY", "rerank-secret")
    seen: dict[str, object] = {}

    def fake_embed(stage, texts, *, query):
        seen["embed"] = stage
        return [[1.0] for _ in texts]

    def fake_rerank(stage, query, docs):
        seen["rerank"] = stage
        return [0.9] * len(docs)

    monkeypatch.setattr(sem, "embed_texts", fake_embed)
    monkeypatch.setattr(sem, "rerank_documents", fake_rerank)

    sem.score_neural("query", results(), cfg)

    assert seen["embed"].url == "https://embed.example/embeddings"
    assert seen["embed"].resolve_key() == "embed-secret"
    assert seen["rerank"].url == "https://rerank.example/rerank"
    assert seen["rerank"].resolve_key() == "rerank-secret"


def test_empty_model_skips_the_stage(monkeypatch) -> None:
    monkeypatch.setattr(sem, "embed_texts", _explode)
    monkeypatch.setattr(sem, "rerank_documents", _explode)

    scores = sem.score_neural("query", results(), make_config(embedding=True, model=""))

    assert scores == [None, None, None]


def test_non_http_url_skips_the_stage(monkeypatch) -> None:
    monkeypatch.setattr(sem, "embed_texts", _explode)
    monkeypatch.setattr(sem, "rerank_documents", _explode)

    scores = sem.score_neural(
        "query", results(), make_config(embedding=True, url="not-a-url")
    )

    assert scores == [None, None, None]


def test_config_error_disables_the_stage(monkeypatch) -> None:
    calls: list[int] = []

    def fail(stage, texts, *, query):
        calls.append(1)
        raise ProviderError("HTTP 401", status_code=401)

    monkeypatch.setattr(sem, "embed_texts", fail)
    monkeypatch.setattr(sem, "rerank_documents", _explode)
    cfg = make_config(embedding=True)

    assert sem.score_neural("query", results(), cfg) == [None, None, None]
    assert sem.score_neural("query", results(), cfg) == [None, None, None]
    assert calls == [1]


def test_transient_error_keeps_the_stage_enabled(monkeypatch) -> None:
    calls: list[int] = []

    def fail(stage, texts, *, query):
        calls.append(1)
        raise ProviderError("HTTP 503", status_code=503)

    monkeypatch.setattr(sem, "embed_texts", fail)
    monkeypatch.setattr(sem, "rerank_documents", _explode)
    cfg = make_config(embedding=True)

    sem.score_neural("query", results(), cfg)
    sem.score_neural("query", results(), cfg)

    assert len(calls) == 2


def test_changed_config_resets_the_disable(monkeypatch) -> None:
    calls: list[int] = []

    def fail(stage, texts, *, query):
        calls.append(1)
        raise ProviderError("HTTP 401", status_code=401)

    monkeypatch.setattr(sem, "embed_texts", fail)
    monkeypatch.setattr(sem, "rerank_documents", _explode)

    sem.score_neural("query", results(), make_config(embedding=True))
    sem.score_neural(
        "query",
        results(),
        make_config(embedding=True, url="https://example.test/other"),
    )

    assert len(calls) == 2


def test_embedding_cache_avoids_a_second_request(monkeypatch) -> None:
    calls: list[list[str]] = []

    def counting_embed(stage, texts, *, query):
        calls.append(texts)
        return fake_vectors(stage, texts, query=query)

    monkeypatch.setattr(sem, "embed_texts", counting_embed)
    cfg = make_config(embedding=True)

    sem.score_neural("query", results(), cfg)
    sem.score_neural("query", results(), cfg)

    assert len(calls) == 1


def test_rerank_cache_avoids_a_second_request(monkeypatch) -> None:
    calls: list[list[str]] = []

    def counting_rerank(stage, query, docs):
        calls.append(docs)
        return [0.9, 0.1]

    monkeypatch.setattr(sem, "rerank_documents", counting_rerank)
    cfg = make_config(rerank=True, top_k=2)

    sem.score_neural("query", results(), cfg)
    sem.score_neural("query", results(), cfg)

    assert len(calls) == 1


def test_missing_key_skips_the_stage(monkeypatch) -> None:
    monkeypatch.delenv("SEM_TEST_MISSING_KEY", raising=False)
    monkeypatch.setattr(sem, "embed_texts", _explode)
    monkeypatch.setattr(sem, "rerank_documents", _explode)

    cfg = make_config(embedding=True, rerank=True, api_key_env="SEM_TEST_MISSING_KEY")

    assert sem.score_neural("query", results(), cfg) == [None, None, None]


def test_max_chars_truncates_before_the_request(monkeypatch) -> None:
    seen: list[list[str]] = []

    def recording_embed(stage, texts, *, query):
        seen.append(texts)
        return fake_vectors(stage, texts, query=query)

    monkeypatch.setattr(sem, "embed_texts", recording_embed)
    sem.score_neural("query", results(), make_config(embedding=True, max_chars=2))

    assert all(len(text) <= 2 for text in seen[0])


def _explode(*args, **kwargs):
    raise AssertionError("provider must not be called")


def _raise_provider_error(*args, **kwargs):
    raise ProviderError("boom")
