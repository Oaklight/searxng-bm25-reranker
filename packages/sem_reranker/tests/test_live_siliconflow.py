"""Live SiliconFlow checks.  Skipped unless SILICONFLOW_API_KEY is set."""

from __future__ import annotations

import os

import pytest

from searxng_sem_reranker._config import load_config
from searxng_sem_reranker._providers import embed, rerank

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.environ.get("SILICONFLOW_API_KEY"),
        reason="SILICONFLOW_API_KEY is not set",
    ),
]


@pytest.mark.parametrize(
    "model",
    ["BAAI/bge-reranker-v2-m3", "Qwen/Qwen3-Reranker-0.6B"],
)
def test_rerank_live(model: str) -> None:
    cfg = load_config(
        {
            "sem_reranker": {
                "rerank": {
                    "enabled": True,
                    "format": "jina",
                    "url": "https://api.siliconflow.cn/v1/rerank",
                    "model": model,
                    "api_key_env": "SILICONFLOW_API_KEY",
                    "top_k": 10,
                    "timeout_ms": 10000,
                }
            }
        }
    )
    scores = rerank(
        cfg.rerank,
        "python asyncio",
        [
            "python asyncio tutorial",
            "banana bread recipe",
            "weather forecast",
        ],
    )
    assert len(scores) == 3
    assert any(score is not None for score in scores)


@pytest.mark.parametrize(
    "model",
    ["BAAI/bge-m3", "Qwen/Qwen3-Embedding-0.6B"],
)
def test_embedding_live(model: str) -> None:
    cfg = load_config(
        {
            "sem_reranker": {
                "embedding": {
                    "enabled": True,
                    "format": "openai",
                    "url": "https://api.siliconflow.cn/v1/embeddings",
                    "model": model,
                    "api_key_env": "SILICONFLOW_API_KEY",
                    "timeout_ms": 10000,
                }
            }
        }
    )
    vectors = embed(cfg.embedding, ["hello", "world"], query=False)
    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1]) > 0
