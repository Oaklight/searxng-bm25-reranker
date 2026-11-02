"""Shared fixtures for the semantic reranker tests."""

from __future__ import annotations

import typing as t

import pytest
import searx

import searxng_sem_reranker as sem


@pytest.fixture(autouse=True)
def _reset_state() -> t.Iterator[None]:
    searx.settings.clear()
    sem._EMBED_CACHE = None
    sem._RERANK_CACHE = None
    sem._CACHE_SHAPE = None
    sem._DISABLED.clear()
    yield
    searx.settings.clear()


@pytest.fixture
def make_result() -> t.Callable[..., dict[str, t.Any]]:
    def _make(
        title: str = "",
        content: str = "",
        positions: list[int] | None = None,
        url: str = "",
    ) -> dict[str, t.Any]:
        return {
            "title": title,
            "content": content,
            "url": url,
            "positions": positions if positions is not None else [1],
        }

    return _make
