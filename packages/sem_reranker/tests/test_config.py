"""Config parsing and validation tests.

There are no presets and no base-url/path split: each stage declares one full
endpoint URL and its own key.
"""

from __future__ import annotations

import pytest
from searx.plugins import PluginCfg

from searxng_sem_reranker._config import load_config


def test_defaults_keep_both_stages_off() -> None:
    cfg = load_config({})
    assert cfg.embedding.enabled is False
    assert cfg.embedding.format == ""
    assert cfg.embedding.url == ""
    assert cfg.embedding.model == ""
    assert cfg.embedding.max_tokens == 0
    assert cfg.embedding.query_prefix == ""
    assert cfg.embedding.document_prefix == ""
    assert cfg.rerank.enabled is False
    assert cfg.rerank.format == ""
    assert cfg.rerank.top_k == 10
    assert cfg.fusion.rrf_k == 20
    assert cfg.cache.ttl == 300


def test_stages_are_explicit_and_independent() -> None:
    cfg = load_config(
        {
            "sem_reranker": {
                "embedding": {
                    "enabled": True,
                    "format": "openai",
                    "url": "https://embed.example/v1/embeddings",
                    "model": "embed-model",
                    "api_key_env": "EMBED_KEY",
                },
                "rerank": {
                    "enabled": True,
                    "format": "jina",
                    "url": "https://rerank.example/v1/rerank",
                    "model": "rerank-model",
                    "api_key_env": "RERANK_KEY",
                    "top_k": 5,
                },
            }
        }
    )
    assert cfg.embedding.url == "https://embed.example/v1/embeddings"
    assert cfg.rerank.url == "https://rerank.example/v1/rerank"
    assert cfg.embedding.model == "embed-model"
    assert cfg.rerank.model == "rerank-model"
    assert cfg.embedding.api_key_env == "EMBED_KEY"
    assert cfg.rerank.api_key_env == "RERANK_KEY"
    assert cfg.rerank.top_k == 5


def test_embedding_prefixes_and_token_budget() -> None:
    cfg = load_config(
        {
            "sem_reranker": {
                "embedding": {
                    "query_prefix": "query: ",
                    "document_prefix": "passage: ",
                    "max_tokens": 512,
                }
            }
        }
    )
    assert cfg.embedding.query_prefix == "query: "
    assert cfg.embedding.document_prefix == "passage: "
    assert cfg.embedding.max_tokens == 512
    assert cfg.rerank.max_tokens == 0


def test_environment_key_wins_per_stage(monkeypatch) -> None:
    monkeypatch.setenv("EMBED_KEY", "embed-env")
    monkeypatch.setenv("RERANK_KEY", "rerank-env")
    cfg = load_config(
        {
            "sem_reranker": {
                "embedding": {
                    "api_key_env": "EMBED_KEY",
                    "api_key": "embed-settings",
                },
                "rerank": {
                    "api_key_env": "RERANK_KEY",
                    "api_key": "rerank-settings",
                },
            }
        }
    )
    assert cfg.embedding.resolve_key() == "embed-env"
    assert cfg.rerank.resolve_key() == "rerank-env"


def test_settings_key_is_the_fallback_per_stage(monkeypatch) -> None:
    monkeypatch.delenv("EMBED_KEY", raising=False)
    monkeypatch.delenv("RERANK_KEY", raising=False)
    cfg = load_config(
        {
            "sem_reranker": {
                "embedding": {
                    "api_key_env": "EMBED_KEY",
                    "api_key": "embed-settings",
                },
                "rerank": {
                    "api_key_env": "RERANK_KEY",
                    "api_key": "rerank-settings",
                },
            }
        }
    )
    assert cfg.embedding.resolve_key() == "embed-settings"
    assert cfg.rerank.resolve_key() == "rerank-settings"


def test_invalid_values_fall_back_to_neutral_defaults() -> None:
    cfg = load_config(
        {
            "sem_reranker": {
                "rerank": {
                    "format": "not-a-format",
                    "url": "ftp://example.com",
                    "top_k": -1,
                    "max_chars": 0,
                    "max_tokens": -1,
                    "timeout_ms": 0,
                    "weight": -1,
                },
                "fusion": {"rrf_k": 0, "engine_weight": -1},
                "cache": {"ttl": -1, "max_entries": 0},
            }
        }
    )
    assert cfg.rerank.format == ""
    assert cfg.rerank.url == ""
    assert cfg.rerank.top_k == 10
    assert cfg.rerank.max_chars == 4000
    assert cfg.rerank.max_tokens == 0
    assert cfg.rerank.timeout_ms == 3000
    assert cfg.rerank.weight == 1.0
    assert cfg.fusion.rrf_k == 20
    assert cfg.fusion.engine_weight == 1.0
    assert cfg.cache.ttl == 300
    assert cfg.cache.max_entries == 512


def test_url_is_kept_verbatim() -> None:
    cfg = load_config(
        {
            "sem_reranker": {
                "rerank": {"url": "https://gateway.example/custom/rerank?x=1"}
            }
        }
    )
    assert cfg.rerank.url == "https://gateway.example/custom/rerank?x=1"


def test_non_http_url_is_rejected() -> None:
    cfg = load_config({"sem_reranker": {"rerank": {"url": "ftp://example.com"}}})
    assert cfg.rerank.url == ""


def test_non_mapping_blocks_are_ignored() -> None:
    cfg = load_config(
        {
            "sem_reranker": {
                "embedding": "nonsense",
                "rerank": ["nonsense"],
                "fusion": 3,
                "cache": None,
            }
        }
    )
    assert cfg.embedding.enabled is False
    assert cfg.rerank.enabled is False
    assert cfg.fusion.rrf_k == 20
    assert cfg.cache.max_entries == 512


def test_non_finite_weight_falls_back() -> None:
    cfg = load_config({"sem_reranker": {"fusion": {"neural_weight": float("inf")}}})
    assert cfg.fusion.neural_weight == 1.0


def test_api_key_is_not_in_repr(monkeypatch) -> None:
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    cfg = load_config({"sem_reranker": {"rerank": {"api_key": "secret-value"}}})
    assert "secret-value" not in repr(cfg)


def test_plugin_cfg_rejects_tunables_under_plugins() -> None:
    """SearXNG calls ``PluginCfg(**settings)``; extra keys raise TypeError."""
    with pytest.raises(TypeError):
        PluginCfg(active=True, rrf_k=20)
