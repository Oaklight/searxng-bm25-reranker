"""Provider wire formats and transport for the semantic reranker.

Payload shapes follow the converters used by ``llm-rosetta``.  HTTP, retry,
and response parsing come from vendored zerodep modules.
"""

from __future__ import annotations

import math
import typing as t

from ._config import StageConfig
from ._vendor import httpclient, retry


class ProviderError(Exception):
    """A provider request failed; the caller keeps the previous ranking."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


def embed(
    stage: StageConfig,
    texts: list[str],
    *,
    query: bool,
) -> list[list[float]]:
    """Embed a batch of texts with the stage's configured format."""
    url = _require_endpoint(stage)
    body = embedding_body(stage, texts, query=query)
    payload = post_json(url, body, stage.resolve_key(), _timeout(stage))
    return parse_embeddings(stage.format, payload, len(texts))


def rerank(
    stage: StageConfig,
    query: str,
    documents: list[str],
) -> list[float | None]:
    """Score documents against a query; missing indices stay ``None``."""
    url = _require_endpoint(stage)
    body = rerank_body(stage, query, documents)
    payload = post_json(url, body, stage.resolve_key(), _timeout(stage))
    return parse_rerank(payload, len(documents))


def embedding_body(stage: StageConfig, texts: list[str], *, query: bool) -> dict:
    if stage.format == "openai":
        return {
            "model": stage.model,
            "input": texts,
            "encoding_format": "float",
        }
    if stage.format == "cohere":
        return {
            "model": stage.model,
            "texts": texts,
            "input_type": "search_query" if query else "search_document",
            "embedding_types": ["float"],
        }
    if stage.format == "jina":
        return {
            "model": stage.model,
            "input": texts,
            "task": "retrieval.query" if query else "retrieval.passage",
        }
    return {
        "model": stage.model,
        "input": texts,
        "input_type": "query" if query else "document",
    }


def rerank_body(stage: StageConfig, query: str, documents: list[str]) -> dict:
    top_k = stage.top_k or len(documents)
    if stage.format == "voyage":
        return {
            "model": stage.model,
            "query": query,
            "documents": documents,
            "top_k": top_k,
        }
    return {
        "model": stage.model,
        "query": query,
        "documents": documents,
        "top_n": top_k,
    }


def parse_embeddings(
    fmt: str,
    payload: t.Mapping[str, t.Any],
    expected: int,
) -> list[list[float]]:
    """Extract vectors from an embedding response and preserve input order."""
    vectors = _vectors_from_payload(fmt, payload)
    if len(vectors) != expected:
        raise ProviderError(
            f"embedding response has {len(vectors)} vectors, expected {expected}"
        )
    return [_parse_vector(vector) for vector in vectors]


def _vectors_from_payload(
    fmt: str,
    payload: t.Mapping[str, t.Any],
) -> list[t.Any]:
    """Locate the raw vector list in any supported response shape."""
    if fmt == "cohere":
        embeddings = payload.get("embeddings")
        if isinstance(embeddings, t.Mapping):
            embeddings = embeddings.get("float")
        vectors = embeddings
    else:
        items = payload.get("data")
        if not isinstance(items, list):
            raise ProviderError("embedding response has no data list")
        if all(isinstance(item, t.Mapping) and "index" in item for item in items):
            try:
                items = sorted(items, key=lambda item: int(item["index"]))
            except (TypeError, ValueError) as exc:
                raise ProviderError("embedding response has a malformed index") from exc
        vectors = [
            item.get("embedding") if isinstance(item, t.Mapping) else None
            for item in items
        ]

    if not isinstance(vectors, list):
        raise ProviderError("embedding response has no vector list")
    return vectors


def _parse_vector(vector: t.Any) -> list[float]:
    """Convert one raw vector to floats, rejecting NaN and infinity."""
    if not isinstance(vector, list) or not vector:
        raise ProviderError("embedding response contains a malformed vector")
    numbers: list[float] = []
    for value in t.cast(list[t.Any], vector):
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProviderError(
                "embedding response contains a non-numeric value"
            ) from exc
        if not math.isfinite(number):
            raise ProviderError("embedding response contains a non-finite value")
        numbers.append(number)
    return numbers


def parse_rerank(
    payload: t.Mapping[str, t.Any],
    expected: int,
) -> list[float | None]:
    """Extract rerank scores into a full-length list indexed by document."""
    items = payload.get("results")
    if not isinstance(items, list):
        items = payload.get("data")
    if not isinstance(items, list):
        raise ProviderError("rerank response has no results list")

    scores: list[float | None] = [None] * expected
    for item in items:
        if not isinstance(item, t.Mapping):
            continue
        try:
            index = int(item["index"])
            score = item.get("relevance_score", item.get("score"))
            value = float(score)
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(value):
            continue
        if 0 <= index < expected:
            scores[index] = value
    return scores


def post_json(
    url: str,
    body: t.Mapping[str, t.Any],
    api_key: str,
    timeout_s: float,
) -> dict[str, t.Any]:
    """POST JSON with one bounded retry on 429, 5xx, and transport errors."""
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    def _do() -> dict[str, t.Any]:
        response = httpclient.post(
            url,
            json=body,
            headers=headers,
            timeout=timeout_s,
            max_redirects=0,
        )
        if not isinstance(response, httpclient.Response):
            raise ProviderError("provider returned a streaming response")
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ProviderError("provider response is not a JSON object")
        return payload

    try:
        return retry.retry_call(
            _do,
            max_retries=1,
            base_delay=0.2,
            retry_on=_retryable,
        )
    except ProviderError:
        raise
    except httpclient.HTTPError as exc:
        raise ProviderError(
            f"HTTP {exc.status_code} from {url}: {_excerpt(exc.body)}",
            status_code=exc.status_code,
        ) from exc
    except (httpclient.HttpClientError, ValueError, OSError) as exc:
        raise ProviderError(f"request to {url} failed: {exc}", retryable=True) from exc


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpclient.HTTPError):
        return exc.status_code == 429 or 500 <= exc.status_code < 600
    return isinstance(
        exc, (httpclient.HttpConnectionError, httpclient.HttpTimeoutError)
    )


def _require_endpoint(stage: StageConfig) -> str:
    url = stage.url
    if not url.startswith(("http://", "https://")):
        raise ProviderError("url is empty or not an HTTP URL")
    return url


def _timeout(stage: StageConfig) -> float:
    return max(stage.timeout_ms / 1000.0, 0.001)


def _excerpt(body: t.Any) -> str:
    text = body if isinstance(body, str) else str(body)
    return text[:200].replace("\n", " ")
