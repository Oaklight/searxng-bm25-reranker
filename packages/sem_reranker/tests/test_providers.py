"""Provider payload, response, transport, and per-stage source tests."""

from __future__ import annotations

import pytest

from searxng_sem_reranker._config import load_config
from searxng_sem_reranker._providers import (
    ProviderError,
    embed,
    embedding_body,
    parse_embeddings,
    parse_rerank,
    post_json,
    rerank,
    rerank_body,
)
from searxng_sem_reranker._vendor import httpclient


def stage(fmt: str = "jina", **overrides):
    block = {
        "format": fmt,
        "url": "https://host/v1/rerank",
        "model": "rerank-model",
        "api_key_env": "",
        "api_key": "",
        "top_k": 10,
    }
    block.update(overrides)
    return load_config({"sem_reranker": {"rerank": block}}).rerank


def embedding_stage(fmt: str = "openai", **overrides):
    block = {
        "format": fmt,
        "url": "https://host/v1/embeddings",
        "model": "embed-model",
        "api_key_env": "",
        "api_key": "",
    }
    block.update(overrides)
    return load_config({"sem_reranker": {"embedding": block}}).embedding


class TestBodies:
    def test_openai_embedding(self) -> None:
        body = embedding_body(embedding_stage("openai"), ["a", "b"], query=True)
        assert body["model"] == "embed-model"
        assert body["input"] == ["a", "b"]
        assert body["encoding_format"] == "float"

    def test_cohere_embedding_uses_input_type(self) -> None:
        body = embedding_body(embedding_stage("cohere"), ["a"], query=True)
        assert body["input_type"] == "search_query"
        assert body["embedding_types"] == ["float"]
        assert body["texts"] == ["a"]

    def test_jina_embedding_uses_task(self) -> None:
        assert (
            embedding_body(embedding_stage("jina"), ["a"], query=False)["task"]
            == "retrieval.passage"
        )
        assert (
            embedding_body(embedding_stage("jina"), ["a"], query=True)["task"]
            == "retrieval.query"
        )

    def test_voyage_embedding_uses_input_type(self) -> None:
        assert (
            embedding_body(embedding_stage("voyage"), ["a"], query=False)["input_type"]
            == "document"
        )

    def test_jina_rerank_uses_top_n(self) -> None:
        body = rerank_body(stage("jina"), "q", ["a", "b"])
        assert body["top_n"] == 10
        assert "top_k" not in body

    def test_voyage_rerank_uses_top_k(self) -> None:
        body = rerank_body(stage("voyage"), "q", ["a", "b"])
        assert body["top_k"] == 10
        assert "top_n" not in body

    def test_zero_top_k_sends_all_documents(self) -> None:
        cfg = stage("jina", top_k=0)
        assert rerank_body(cfg, "q", ["a", "b", "c"])["top_n"] == 3


class TestParseEmbeddings:
    def test_openai_data_is_sorted_by_index(self) -> None:
        payload = {
            "data": [
                {"index": 1, "embedding": [0.3, 0.4]},
                {"index": 0, "embedding": [0.1, 0.2]},
            ]
        }
        assert parse_embeddings("openai", payload, 2) == [[0.1, 0.2], [0.3, 0.4]]

    def test_missing_index_preserves_order(self) -> None:
        payload = {"data": [{"embedding": [1.0]}, {"embedding": [2.0]}]}
        assert parse_embeddings("jina", payload, 2) == [[1.0], [2.0]]

    def test_cohere_float_list(self) -> None:
        payload = {"embeddings": {"float": [[1.0, 2.0], [3.0, 4.0]]}}
        assert parse_embeddings("cohere", payload, 2) == [[1.0, 2.0], [3.0, 4.0]]

    def test_malformed_index_raises(self) -> None:
        payload = {"data": [{"index": "not-a-number", "embedding": [1.0]}]}
        with pytest.raises(ProviderError):
            parse_embeddings("openai", payload, 1)

    def test_wrong_count_raises(self) -> None:
        payload = {"data": [{"embedding": [1.0]}]}
        with pytest.raises(ProviderError):
            parse_embeddings("openai", payload, 2)

    def test_malformed_vector_raises(self) -> None:
        payload = {"data": [{"embedding": []}]}
        with pytest.raises(ProviderError):
            parse_embeddings("openai", payload, 1)

    @pytest.mark.parametrize("value", ["nan", "inf", "-inf", "not-a-number", None])
    def test_non_finite_or_non_numeric_values_raise(self, value) -> None:
        payload = {"data": [{"embedding": [1.0, value]}]}
        with pytest.raises(ProviderError):
            parse_embeddings("openai", payload, 1)


class TestParseRerank:
    def test_results_are_placed_by_index(self) -> None:
        payload = {
            "results": [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "score": 0.5},
            ]
        }
        assert parse_rerank(payload, 2) == [0.5, 0.9]

    def test_voyage_uses_data(self) -> None:
        payload = {"data": [{"index": 2, "relevance_score": 0.7}]}
        assert parse_rerank(payload, 3) == [None, None, 0.7]

    def test_out_of_range_index_is_ignored(self) -> None:
        payload = {"results": [{"index": 9, "relevance_score": 1.0}]}
        assert parse_rerank(payload, 2) == [None, None]

    @pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_scores_are_skipped(self, score) -> None:
        payload = {"results": [{"index": 0, "relevance_score": score}]}
        assert parse_rerank(payload, 1) == [None]

    def test_missing_results_raises(self) -> None:
        with pytest.raises(ProviderError):
            parse_rerank({}, 2)


class TestPostJson:
    def test_bearer_header_is_sent(self, monkeypatch) -> None:
        captured: dict = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            captured.update(kwargs)
            return _FakeResponse({"ok": True})

        monkeypatch.setattr(httpclient, "post", fake_post)
        assert post_json("https://host/v1/rerank", {"a": 1}, "secret", 1.0) == {
            "ok": True
        }
        assert captured["headers"]["Authorization"] == "Bearer secret"
        assert captured["json"] == {"a": 1}
        assert captured["timeout"] == 1.0
        assert captured["max_redirects"] == 0

    def test_no_key_sends_no_authorization(self, monkeypatch) -> None:
        captured: dict = {}

        def fake_post(url, **kwargs):
            captured.update(kwargs)
            return _FakeResponse({"ok": True})

        monkeypatch.setattr(httpclient, "post", fake_post)
        post_json("https://host/v1/rerank", {}, "", 1.0)
        assert "Authorization" not in captured["headers"]


class TestDirectApiKey:
    def test_api_key_works_without_an_env_name(self, monkeypatch) -> None:
        cfg = embedding_stage("openai", api_key_env="", api_key="direct-secret")
        captured: dict = {}

        def fake_post(url, **kwargs):
            captured.update(kwargs)
            return _FakeResponse({"data": [{"index": 0, "embedding": [1.0]}]})

        monkeypatch.setattr(httpclient, "post", fake_post)
        embed(cfg, ["text"], query=False)

        assert captured["headers"]["Authorization"] == "Bearer direct-secret"

    def test_env_wins_when_both_are_set(self, monkeypatch) -> None:
        monkeypatch.setenv("EMBED_KEY", "from-env")
        cfg = embedding_stage("openai", api_key_env="EMBED_KEY", api_key="direct")
        assert cfg.resolve_key() == "from-env"


class TestFullEndpoint:
    def test_url_is_used_verbatim(self, monkeypatch) -> None:
        cfg = embedding_stage("openai", url="https://gateway.example/custom/embed?x=1")
        captured: dict = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            return _FakeResponse({"data": [{"index": 0, "embedding": [1.0]}]})

        monkeypatch.setattr(httpclient, "post", fake_post)
        embed(cfg, ["text"], query=False)

        assert captured["url"] == "https://gateway.example/custom/embed?x=1"


class TestIndependentSources:
    def test_stages_use_their_own_url_and_key(self, monkeypatch) -> None:
        embed_cfg = embedding_stage(
            "openai",
            url="https://embed.example/v1/embeddings",
            api_key_env="EMBED_KEY",
        )
        rerank_cfg = stage(
            "jina",
            url="https://rerank.example/v1/rerank",
            api_key_env="RERANK_KEY",
        )
        monkeypatch.setenv("EMBED_KEY", "embed-secret")
        monkeypatch.setenv("RERANK_KEY", "rerank-secret")

        calls: list[tuple[str, str | None]] = []

        def fake_post(url, **kwargs):
            calls.append((url, kwargs["headers"].get("Authorization")))
            if url.endswith("/embeddings"):
                return _FakeResponse({"data": [{"index": 0, "embedding": [1.0, 2.0]}]})
            return _FakeResponse({"results": [{"index": 0, "relevance_score": 0.5}]})

        monkeypatch.setattr(httpclient, "post", fake_post)

        embed(embed_cfg, ["text"], query=False)
        rerank(rerank_cfg, "query", ["document"])

        assert calls == [
            ("https://embed.example/v1/embeddings", "Bearer embed-secret"),
            ("https://rerank.example/v1/rerank", "Bearer rerank-secret"),
        ]


class _FakeResponse(httpclient.Response):
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload
