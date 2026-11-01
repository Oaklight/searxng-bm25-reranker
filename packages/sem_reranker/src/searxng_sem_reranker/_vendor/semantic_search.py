# /// zerodep
# version = "0.2.0"
# deps = []
# tier = "subsystem"
# category = "text"
# note = "Install/update via `zerodep add semantic_search`"
# ///
"""Zero-dependency dense vector search with Flat, IVF, and LSH indices.

Part of zerodep: https://github.com/Oaklight/zerodep
Copyright (c) 2026 Peng Ding. MIT License.

Dense vector similarity search engine supporting exact (Flat) and approximate
(IVF, LSH) nearest neighbor search with optional SQ8 quantization. Includes
pluggable embedding function support and built-in API helpers for OpenAI,
Cohere, Voyage, and Jina. Designed for use in LLM/Agent/RAG pipelines.

Basic usage::

    index = SemanticIndex(dim=384, metric="cosine")
    index.add("doc1", vector=[0.1, 0.2, ...])
    index.add("doc2", vector=[0.3, 0.4, ...])
    results = index.search(vector=[0.15, 0.25, ...], top_k=5)

With embedding function::

    embed = openai_embed(api_key="sk-...", model="text-embedding-3-small")
    index = SemanticIndex(dim=1536, embed=embed)
    index.add("doc1", text="the quick brown fox")
    results = index.search(text="fast fox", top_k=5)

IVF index for larger collections::

    index = SemanticIndex(dim=384, index_type="ivf", n_clusters=32, nprobe=4)
    for doc_id, vec in vectors.items():
        index.add(doc_id, vector=vec)
    index.build_index()
    results = index.search(vector=query_vec)

Persistence::

    index.save("index.json")
    index = SemanticIndex.load("index.json")

Hybrid sparse+dense search::

    from sparse_search import SparseIndex
    sparse_results = sparse_idx.search("query text")
    dense_results = semantic_idx.search(vector=query_vec)
    combined = rrf(sparse_results, dense_results)
"""

from __future__ import annotations

import array
import base64
import heapq
import json
import math
import random
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

__all__ = [
    "Result",
    "SemanticIndex",
    "cosine_similarity",
    "detect_zh_en",
    "jina_embed",
    "cohere_embed",
    "mmr",
    "openai_embed",
    "rrf",
    "voyage_embed",
]

_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Distance Metrics
# ---------------------------------------------------------------------------


def _normalize(v: list[float]) -> list[float]:
    """L2-normalize a vector. Returns zero vector if norm is zero."""
    norm = math.sqrt(sum(x * x for x in v))
    if norm == 0.0:
        return v
    inv = 1.0 / norm
    return [x * inv for x in v]


def _cosine_distance(
    vectors: array.array[float],
    query: list[float],
    dim: int,
    offset: int,
) -> float:
    """Cosine distance (1 - dot) on pre-normalized vectors.

    Args:
        vectors: Flat float32 array holding all stored vectors.
        query: Query vector (should be pre-normalized).
        dim: Dimensionality.
        offset: Element offset into *vectors* for the target vector.

    Returns:
        Distance value in [0, 2].
    """
    base = offset
    dot = 0.0
    for j in range(dim):
        dot += vectors[base + j] * query[j]
    return 1.0 - dot


def _euclidean_distance(
    vectors: array.array[float],
    query: list[float],
    dim: int,
    offset: int,
) -> float:
    """Euclidean (L2) distance.

    Args:
        vectors: Flat float32 array holding all stored vectors.
        query: Query vector.
        dim: Dimensionality.
        offset: Element offset into *vectors* for the target vector.

    Returns:
        Non-negative distance.
    """
    base = offset
    acc = 0.0
    for j in range(dim):
        d = vectors[base + j] - query[j]
        acc += d * d
    return math.sqrt(acc)


def _inner_product_distance(
    vectors: array.array[float],
    query: list[float],
    dim: int,
    offset: int,
) -> float:
    """Negated inner-product distance (smaller = better).

    Args:
        vectors: Flat float32 array holding all stored vectors.
        query: Query vector.
        dim: Dimensionality.
        offset: Element offset into *vectors* for the target vector.

    Returns:
        Negated dot product.
    """
    base = offset
    dot = 0.0
    for j in range(dim):
        dot += vectors[base + j] * query[j]
    return -dot


_DISTANCE_FNS: dict[
    str,
    Callable[[array.array[float], list[float], int, int], float],
] = {
    "cosine": _cosine_distance,
    "euclidean": _euclidean_distance,
    "inner_product": _inner_product_distance,
}


# ---------------------------------------------------------------------------
# Vector Utilities
# ---------------------------------------------------------------------------


def _quantize_sq8(
    vectors: array.array[float],
    dim: int,
    n: int,
) -> tuple[bytes, array.array[float], array.array[float]]:
    """Scalar quantization to 8-bit unsigned integers.

    For each dimension, maps the value range [min, max] linearly to [0, 255].

    Args:
        vectors: Flat float32 array of *n* vectors each of *dim* dimensions.
        dim: Dimensionality.
        n: Number of vectors.

    Returns:
        Tuple of (quantized_bytes, min_vals_per_dim, step_vals_per_dim).
    """
    min_vals = array.array("f", [float("inf")] * dim)
    max_vals = array.array("f", [float("-inf")] * dim)

    for i in range(n):
        base = i * dim
        for j in range(dim):
            v = vectors[base + j]
            if v < min_vals[j]:
                min_vals[j] = v
            if v > max_vals[j]:
                max_vals[j] = v

    step_vals = array.array("f", [0.0] * dim)
    for j in range(dim):
        r = max_vals[j] - min_vals[j]
        step_vals[j] = r / 255.0 if r > 0.0 else 0.0

    quantized = bytearray(n * dim)
    for i in range(n):
        base = i * dim
        for j in range(dim):
            if step_vals[j] == 0.0:
                quantized[base + j] = 0
            else:
                val = (vectors[base + j] - min_vals[j]) / step_vals[j]
                quantized[base + j] = max(0, min(255, round(val)))

    return bytes(quantized), min_vals, step_vals


def _dequantize_sq8_single(
    data: bytes,
    min_vals: array.array[float],
    step_vals: array.array[float],
    dim: int,
    idx: int,
) -> list[float]:
    """Dequantize a single vector from SQ8 data.

    Args:
        data: Quantized byte buffer.
        min_vals: Per-dimension minimum values.
        step_vals: Per-dimension step sizes.
        dim: Dimensionality.
        idx: Index of the vector to dequantize.

    Returns:
        Reconstructed float vector.
    """
    base = idx * dim
    return [min_vals[j] + data[base + j] * step_vals[j] for j in range(dim)]


def _dequantize_sq8_all(
    data: bytes,
    min_vals: array.array[float],
    step_vals: array.array[float],
    dim: int,
    n: int,
) -> array.array[float]:
    """Dequantize every stored vector back into a flat float32 array.

    Args:
        data: Quantized byte buffer.
        min_vals: Per-dimension minimum values.
        step_vals: Per-dimension step sizes.
        dim: Dimensionality.
        n: Number of vectors.

    Returns:
        Flat float32 array holding *n* reconstructed vectors.
    """
    out = array.array("f")
    for i in range(n):
        base = i * dim
        for j in range(dim):
            out.append(min_vals[j] + data[base + j] * step_vals[j])
    return out


def _sq8_norms(
    data: bytes,
    min_vals: array.array[float],
    step_vals: array.array[float],
    dim: int,
    n: int,
) -> array.array[float]:
    """Precompute the L2 norm of each dequantized vector.

    Cosine distance needs the candidate norm because quantization does not
    preserve unit length.

    Args:
        data: Quantized byte buffer.
        min_vals: Per-dimension minimum values.
        step_vals: Per-dimension step sizes.
        dim: Dimensionality.
        n: Number of vectors.

    Returns:
        Float32 array of *n* norms.
    """
    norms = array.array("f")
    for i in range(n):
        base = i * dim
        acc = 0.0
        for j in range(dim):
            v = min_vals[j] + data[base + j] * step_vals[j]
            acc += v * v
        norms.append(math.sqrt(acc))
    return norms


def _sq8_distance_factory(
    metric: str,
    data: bytes,
    min_vals: array.array[float],
    step_vals: array.array[float],
    norms: array.array[float],
) -> Callable[[array.array[float], list[float], int, int], float]:
    """Build a distance function that reads vectors from SQ8 data.

    The returned callable keeps the ``(vectors, query, dim, offset)``
    signature of the full-precision distance functions so the index
    implementations can use it unchanged: *vectors* is ignored and the
    stored vector is reconstructed from the quantized buffer instead.
    """

    def _sq8_cosine(
        vectors: array.array[float],
        query: list[float],
        dim: int,
        offset: int,
    ) -> float:
        dot = 0.0
        for j in range(dim):
            dot += (min_vals[j] + data[offset + j] * step_vals[j]) * query[j]
        norm = norms[offset // dim]
        return 1.0 - dot / norm if norm > 0.0 else 1.0

    def _sq8_euclidean(
        vectors: array.array[float],
        query: list[float],
        dim: int,
        offset: int,
    ) -> float:
        acc = 0.0
        for j in range(dim):
            d = (min_vals[j] + data[offset + j] * step_vals[j]) - query[j]
            acc += d * d
        return math.sqrt(acc)

    def _sq8_inner_product(
        vectors: array.array[float],
        query: list[float],
        dim: int,
        offset: int,
    ) -> float:
        dot = 0.0
        for j in range(dim):
            dot += (min_vals[j] + data[offset + j] * step_vals[j]) * query[j]
        return -dot

    if metric == "cosine":
        return _sq8_cosine
    if metric == "euclidean":
        return _sq8_euclidean
    return _sq8_inner_product


# ---------------------------------------------------------------------------
# Language Detection
# ---------------------------------------------------------------------------


def detect_zh_en(text: str, threshold: float = 0.3) -> str:
    """Detect whether text is predominantly Chinese or English.

    Counts CJK unified ideographs versus Latin letters, Japanese kana,
    and Korean Hangul.  Returns ``"zh"`` when the CJK ratio among all
    counted characters exceeds *threshold*, otherwise ``"en"``.

    Args:
        text: Input text to classify.
        threshold: Minimum ratio of CJK characters to total counted
            characters to classify as Chinese.  Default 0.3.

    Returns:
        ``"zh"`` or ``"en"``.
    """
    cjk = 0
    other = 0
    for ch in text:
        cp = ord(ch)
        if (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF):
            cjk += 1
        elif (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A):
            other += 1
        elif (0x3040 <= cp <= 0x309F) or (0x30A0 <= cp <= 0x30FF):
            other += 1
        elif 0xAC00 <= cp <= 0xD7AF:
            other += 1

    total = cjk + other
    if total == 0:
        return "en"
    return "zh" if cjk / total >= threshold else "en"


# ---------------------------------------------------------------------------
# Embedding Protocol & API Helpers
# ---------------------------------------------------------------------------


@runtime_checkable
class EmbeddingFunction(Protocol):
    """Protocol for embedding functions.

    Any callable with signature ``list[str] -> list[list[float]]`` satisfies
    this protocol.
    """

    def __call__(self, texts: list[str]) -> list[list[float]]: ...


def _api_post(url: str, headers: dict[str, str], payload: dict) -> Any:
    """POST JSON to an API endpoint and return parsed response.

    Args:
        url: Full endpoint URL.
        headers: HTTP headers.
        payload: JSON-serializable request body.

    Returns:
        Parsed JSON response.

    Raises:
        ValueError: If *payload* carries a non-finite float; ``NaN`` and
            ``Infinity`` are not valid JSON, so nothing is sent.
        urllib.error.URLError: On network or HTTP errors.
    """
    data = json.dumps(payload, allow_nan=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _openai_compat_embed(
    base_url: str,
    api_key: str,
    model: str,
) -> Callable[[list[str]], list[list[float]]]:
    """Create an embedding function for OpenAI-compatible ``/v1/embeddings`` APIs.

    Args:
        base_url: Base URL of the API (without trailing slash).
        api_key: API key for authentication.
        model: Model name.

    Returns:
        Embedding function ``list[str] -> list[list[float]]``.
    """

    def _embed(texts: list[str]) -> list[list[float]]:
        url = f"{base_url.rstrip('/')}/v1/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {"input": texts, "model": model}
        resp = _api_post(url, headers, payload)
        items = sorted(resp["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in items]

    return _embed


def openai_embed(
    api_key: str = "",
    model: str = "text-embedding-3-small",
    *,
    base_url: str = "https://api.openai.com",
) -> Callable[[list[str]], list[list[float]]]:
    """Create an OpenAI-compatible embedding function.

    Returns a closure that calls the ``/v1/embeddings`` endpoint.

    Args:
        api_key: API key for authentication.
        model: Model name.
        base_url: Base URL of the API (without trailing slash).

    Returns:
        Embedding function ``list[str] -> list[list[float]]``.
    """
    return _openai_compat_embed(base_url, api_key, model)


def cohere_embed(
    api_key: str,
    model: str = "embed-english-v3.0",
    *,
    input_type: str = "search_document",
) -> Callable[[list[str]], list[list[float]]]:
    """Create a Cohere embedding function.

    Returns a closure that calls the Cohere ``/v2/embed`` endpoint.

    Args:
        api_key: API key for authentication.
        model: Model name.
        input_type: Input type (e.g. ``"search_document"``,
            ``"search_query"``).

    Returns:
        Embedding function ``list[str] -> list[list[float]]``.
    """

    def _embed(texts: list[str]) -> list[list[float]]:
        url = "https://api.cohere.com/v2/embed"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "texts": texts,
            "model": model,
            "input_type": input_type,
            "embedding_types": ["float"],
        }
        resp = _api_post(url, headers, payload)
        return resp["embeddings"]["float"]

    return _embed


def voyage_embed(
    api_key: str,
    model: str = "voyage-3-lite",
) -> Callable[[list[str]], list[list[float]]]:
    """Create a Voyage AI embedding function.

    Returns a closure that calls the Voyage ``/v1/embeddings`` endpoint.

    Args:
        api_key: API key for authentication.
        model: Model name.

    Returns:
        Embedding function ``list[str] -> list[list[float]]``.
    """
    return _openai_compat_embed("https://api.voyageai.com", api_key, model)


def jina_embed(
    api_key: str,
    model: str = "jina-embeddings-v3",
) -> Callable[[list[str]], list[list[float]]]:
    """Create a Jina AI embedding function.

    Returns a closure that calls the Jina ``/v1/embeddings`` endpoint.

    Args:
        api_key: API key for authentication.
        model: Model name.

    Returns:
        Embedding function ``list[str] -> list[list[float]]``.
    """
    return _openai_compat_embed("https://api.jina.ai", api_key, model)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class Result:
    """A single search result."""

    doc_id: str
    score: float
    metadata: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Internal Index Implementations
# ---------------------------------------------------------------------------


class _FlatIndex:
    """Brute-force exact nearest neighbor search."""

    def search(
        self,
        query: list[float],
        vectors: array.array[float],
        dim: int,
        dist_fn: Callable[[array.array[float], list[float], int, int], float],
        n_docs: int,
        top_k: int,
        sq8_dist: Callable[[array.array[float], list[float], int, int], float]
        | None = None,
    ) -> list[tuple[int, float]]:
        """Return top_k nearest (index, distance) pairs.

        Args:
            query: Query vector.
            vectors: Flat float32 array of all vectors.
            dim: Dimensionality.
            dist_fn: Distance function.
            n_docs: Number of stored vectors.
            top_k: Number of results.
            sq8_dist: Optional distance function reading quantized storage;
                when given it replaces *dist_fn* for candidate distances.

        Returns:
            List of (vector_index, distance) sorted by ascending distance.
        """
        cand_dist = sq8_dist or dist_fn
        dists = [(i, cand_dist(vectors, query, dim, i * dim)) for i in range(n_docs)]
        return heapq.nsmallest(top_k, dists, key=lambda x: x[1])

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {"type": "flat"}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _FlatIndex:
        """Deserialize from dict."""
        return cls()


def _extract_vector(
    vectors: array.array[float],
    idx: int,
    dim: int,
) -> list[float]:
    """Extract a single vector from the flat array."""
    base = idx * dim
    return [vectors[base + j] for j in range(dim)]


def _kmeanspp_init(
    vectors: array.array[float],
    dim: int,
    k: int,
    n: int,
    rng: random.Random,
) -> list[list[float]]:
    """K-means++ centroid initialization."""
    first = rng.randrange(n)
    centroids: list[list[float]] = [_extract_vector(vectors, first, dim)]

    for _ in range(1, k):
        dists_sq: list[float] = []
        for i in range(n):
            base = i * dim
            min_d = float("inf")
            for c in centroids:
                d = sum((vectors[base + j] - c[j]) ** 2 for j in range(dim))
                if d < min_d:
                    min_d = d
            dists_sq.append(min_d)

        total = sum(dists_sq)
        if total == 0.0:
            break

        threshold = rng.random() * total
        cumulative = 0.0
        chosen = n - 1
        for i, dsq in enumerate(dists_sq):
            cumulative += dsq
            if cumulative >= threshold:
                chosen = i
                break

        centroids.append(_extract_vector(vectors, chosen, dim))

    return centroids


def _kmeans(
    vectors: array.array[float],
    dim: int,
    k: int,
    dist_fn: Callable[[array.array[float], list[float], int, int], float],
    max_iter: int = 20,
    seed: int = 42,
) -> array.array[float]:
    """K-means clustering with k-means++ initialization.

    Args:
        vectors: Flat float32 array of all vectors.
        dim: Dimensionality.
        k: Number of clusters.
        dist_fn: Distance function.
        max_iter: Maximum number of Lloyd iterations.
        seed: Random seed for reproducibility.

    Returns:
        Centroids as a flat float32 array of shape (k, dim).
    """
    rng = random.Random(seed)
    n = len(vectors) // dim

    if k >= n:
        return array.array("f", vectors)

    centroids = _kmeanspp_init(vectors, dim, k, n, rng)

    centroid_arr = array.array("f", [0.0] * (len(centroids) * dim))
    for ci, c in enumerate(centroids):
        for j in range(dim):
            centroid_arr[ci * dim + j] = c[j]

    actual_k = len(centroids)
    assignments = [0] * n

    for _ in range(max_iter):
        changed = False
        for i in range(n):
            vec_i = _extract_vector(vectors, i, dim)
            best_cluster = 0
            best_dist = float("inf")
            for ci in range(actual_k):
                d = dist_fn(centroid_arr, vec_i, dim, ci * dim)
                if d < best_dist:
                    best_dist = d
                    best_cluster = ci
            if assignments[i] != best_cluster:
                assignments[i] = best_cluster
                changed = True

        if not changed:
            break

        counts = [0] * actual_k
        new_centroids = [0.0] * (actual_k * dim)
        for i in range(n):
            ci = assignments[i]
            counts[ci] += 1
            base = i * dim
            c_base = ci * dim
            for j in range(dim):
                new_centroids[c_base + j] += vectors[base + j]

        for ci in range(actual_k):
            c_base = ci * dim
            if counts[ci] > 0:
                inv = 1.0 / counts[ci]
                for j in range(dim):
                    centroid_arr[c_base + j] = new_centroids[c_base + j] * inv

    return centroid_arr


class _IVFIndex:
    """Inverted File Index for approximate nearest neighbor search."""

    def __init__(self, n_clusters: int = 64, nprobe: int = 4) -> None:
        self._n_clusters = n_clusters
        self._nprobe = nprobe
        self._centroids: array.array[float] = array.array("f")
        self._inverted_lists: dict[int, list[int]] = {}
        self._actual_k: int = 0

    def build(
        self,
        vectors: array.array[float],
        dim: int,
        dist_fn: Callable[[array.array[float], list[float], int, int], float],
        n_docs: int,
        seed: int = 42,
    ) -> None:
        """Build the IVF index by clustering vectors.

        Args:
            vectors: Flat float32 array of all vectors.
            dim: Dimensionality.
            dist_fn: Distance function.
            n_docs: Number of vectors.
            seed: Random seed.
        """
        k = min(self._n_clusters, n_docs)
        self._centroids = _kmeans(vectors, dim, k, dist_fn, seed=seed)
        self._actual_k = len(self._centroids) // dim

        self._inverted_lists = {ci: [] for ci in range(self._actual_k)}
        for i in range(n_docs):
            vec = [vectors[i * dim + j] for j in range(dim)]
            best_cluster = 0
            best_dist = float("inf")
            for ci in range(self._actual_k):
                d = dist_fn(self._centroids, vec, dim, ci * dim)
                if d < best_dist:
                    best_dist = d
                    best_cluster = ci
            self._inverted_lists[best_cluster].append(i)

    def search(
        self,
        query: list[float],
        vectors: array.array[float],
        dim: int,
        dist_fn: Callable[[array.array[float], list[float], int, int], float],
        n_docs: int,
        top_k: int,
        sq8_dist: Callable[[array.array[float], list[float], int, int], float]
        | None = None,
    ) -> list[tuple[int, float]]:
        """Search using nprobe nearest clusters.

        Args:
            query: Query vector.
            vectors: Flat float32 array.
            dim: Dimensionality.
            dist_fn: Distance function.
            n_docs: Number of vectors.
            top_k: Number of results.
            sq8_dist: Optional distance function reading quantized storage;
                when given it replaces *dist_fn* for candidate distances.

        Returns:
            List of (vector_index, distance) sorted by ascending distance.
        """
        centroid_dists = []
        for ci in range(self._actual_k):
            d = dist_fn(self._centroids, query, dim, ci * dim)
            centroid_dists.append((ci, d))

        nprobe = min(self._nprobe, self._actual_k)
        nearest_clusters = heapq.nsmallest(nprobe, centroid_dists, key=lambda x: x[1])

        cand_dist = sq8_dist or dist_fn
        candidates: list[tuple[int, float]] = []
        for ci, _ in nearest_clusters:
            for idx in self._inverted_lists.get(ci, []):
                d = cand_dist(vectors, query, dim, idx * dim)
                candidates.append((idx, d))

        return heapq.nsmallest(top_k, candidates, key=lambda x: x[1])

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        centroids_b64 = base64.b64encode(self._centroids.tobytes()).decode("ascii")
        inv_lists = {str(k): v for k, v in self._inverted_lists.items()}
        return {
            "type": "ivf",
            "n_clusters": self._n_clusters,
            "nprobe": self._nprobe,
            "actual_k": self._actual_k,
            "centroids_b64": centroids_b64,
            "inverted_lists": inv_lists,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _IVFIndex:
        """Deserialize from dict."""
        obj = cls(n_clusters=data["n_clusters"], nprobe=data["nprobe"])
        obj._actual_k = data["actual_k"]
        raw = base64.b64decode(data["centroids_b64"])
        obj._centroids = array.array("f")
        obj._centroids.frombytes(raw)
        obj._inverted_lists = {int(k): v for k, v in data["inverted_lists"].items()}
        return obj


class _LSHIndex:
    """Locality-Sensitive Hashing for approximate nearest neighbor search."""

    def __init__(self, n_tables: int = 8, n_hash_funcs: int = 12) -> None:
        self._n_tables = n_tables
        self._n_hash_funcs = n_hash_funcs
        self._projections: array.array[float] = array.array("f")
        self._hash_tables: list[dict[int, list[int]]] = []
        self._dim: int = 0

    def build(
        self,
        vectors: array.array[float],
        dim: int,
        dist_fn: Callable[[array.array[float], list[float], int, int], float],
        n_docs: int,
        seed: int = 42,
    ) -> None:
        """Build LSH tables with random projections.

        Args:
            vectors: Flat float32 array of all vectors.
            dim: Dimensionality.
            dist_fn: Distance function (used for fallback, not hashing).
            n_docs: Number of vectors.
            seed: Random seed.
        """
        rng = random.Random(seed)
        self._dim = dim
        n_proj = self._n_tables * self._n_hash_funcs * dim
        self._projections = array.array(
            "f", [rng.gauss(0.0, 1.0) for _ in range(n_proj)]
        )

        self._hash_tables = [{} for _ in range(self._n_tables)]
        for i in range(n_docs):
            base = i * dim
            for t in range(self._n_tables):
                h = self._hash_vector(vectors, base, t)
                bucket = self._hash_tables[t].setdefault(h, [])
                bucket.append(i)

    def _hash_vector(
        self,
        vectors: array.array[float],
        base: int,
        table_idx: int,
    ) -> int:
        """Compute hash for a vector in a specific table.

        Args:
            vectors: Flat float32 array.
            base: Element offset of the vector in *vectors*.
            table_idx: Which hash table.

        Returns:
            Integer hash value.
        """
        dim = self._dim
        k = self._n_hash_funcs
        proj_base = table_idx * k * dim
        h = 0
        for ki in range(k):
            dot = 0.0
            p_offset = proj_base + ki * dim
            for j in range(dim):
                dot += vectors[base + j] * self._projections[p_offset + j]
            if dot >= 0.0:
                h |= 1 << ki
        return h

    def _hash_query(self, query: list[float], table_idx: int) -> int:
        """Compute hash for a query vector in a specific table.

        Args:
            query: Query vector as list.
            table_idx: Which hash table.

        Returns:
            Integer hash value.
        """
        dim = self._dim
        k = self._n_hash_funcs
        proj_base = table_idx * k * dim
        h = 0
        for ki in range(k):
            dot = 0.0
            p_offset = proj_base + ki * dim
            for j in range(dim):
                dot += query[j] * self._projections[p_offset + j]
            if dot >= 0.0:
                h |= 1 << ki
        return h

    def search(
        self,
        query: list[float],
        vectors: array.array[float],
        dim: int,
        dist_fn: Callable[[array.array[float], list[float], int, int], float],
        n_docs: int,
        top_k: int,
        sq8_dist: Callable[[array.array[float], list[float], int, int], float]
        | None = None,
    ) -> list[tuple[int, float]]:
        """Search by hashing query and collecting candidates.

        Falls back to flat search if too few candidates are found.

        Args:
            query: Query vector.
            vectors: Flat float32 array.
            dim: Dimensionality.
            dist_fn: Distance function.
            n_docs: Number of vectors.
            top_k: Number of results.
            sq8_dist: Optional distance function reading quantized storage;
                when given it replaces *dist_fn* for candidate distances.

        Returns:
            List of (vector_index, distance) sorted by ascending distance.
        """
        candidates: set[int] = set()
        for t in range(self._n_tables):
            h = self._hash_query(query, t)
            bucket = self._hash_tables[t].get(h)
            if bucket is not None:
                candidates.update(bucket)

        if len(candidates) < top_k:
            flat = _FlatIndex()
            return flat.search(
                query, vectors, dim, dist_fn, n_docs, top_k, sq8_dist=sq8_dist
            )

        cand_dist = sq8_dist or dist_fn
        dists = [(idx, cand_dist(vectors, query, dim, idx * dim)) for idx in candidates]
        return heapq.nsmallest(top_k, dists, key=lambda x: x[1])

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        proj_b64 = base64.b64encode(self._projections.tobytes()).decode("ascii")
        tables = []
        for table in self._hash_tables:
            tables.append({str(k): v for k, v in table.items()})
        return {
            "type": "lsh",
            "n_tables": self._n_tables,
            "n_hash_funcs": self._n_hash_funcs,
            "dim": self._dim,
            "projections_b64": proj_b64,
            "hash_tables": tables,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _LSHIndex:
        """Deserialize from dict."""
        obj = cls(n_tables=data["n_tables"], n_hash_funcs=data["n_hash_funcs"])
        obj._dim = data["dim"]
        raw = base64.b64decode(data["projections_b64"])
        obj._projections = array.array("f")
        obj._projections.frombytes(raw)
        obj._hash_tables = []
        for table_data in data["hash_tables"]:
            obj._hash_tables.append({int(k): v for k, v in table_data.items()})
        return obj


# ---------------------------------------------------------------------------
# SemanticIndex
# ---------------------------------------------------------------------------


class SemanticIndex:
    """Dense vector search index for semantic similarity.

    This class is not thread-safe.  External locking is required for
    concurrent access from multiple threads.

    Args:
        dim: Vector dimensionality.
        metric: Distance metric. One of ``"cosine"``, ``"euclidean"``,
            ``"inner_product"``.
        index_type: Index type. One of ``"flat"``, ``"ivf"``, ``"lsh"``.
        embed: Optional embedding function ``list[str] -> list[list[float]]``.
        quantize: If ``True``, :meth:`build_index` compresses the stored
            vectors to SQ8 (one byte per dimension) and releases the
            float32 copy; searches then approximate distances from the
            quantized data (typically ~1% recall loss) and the serialized
            index is about 4x smaller. Mutating the index
            (``add``/``update``/``delete``) reconstructs the float copy;
            the next build re-quantizes it.
        **index_params: Index-specific parameters. For ``"ivf"``:
            ``n_clusters`` (default 64), ``nprobe`` (default 4).
            For ``"lsh"``: ``n_tables`` (default 8), ``n_hash_funcs``
            (default 12).
    """

    _SQLITE_MAGIC = b"SQLite format 3\x00"

    def __init__(
        self,
        dim: int,
        metric: str = "cosine",
        index_type: str = "flat",
        embed: Callable[[list[str]], list[list[float]]] | None = None,
        quantize: bool = False,
        **index_params: Any,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if metric not in ("cosine", "euclidean", "inner_product"):
            raise ValueError(
                f"Unknown metric {metric!r}, expected 'cosine', 'euclidean', "
                f"or 'inner_product'"
            )
        if index_type not in ("flat", "ivf", "lsh"):
            raise ValueError(
                f"Unknown index_type {index_type!r}, expected 'flat', 'ivf', or 'lsh'"
            )

        self._dim = dim
        self._metric = metric
        self._index_type = index_type
        self._embed = embed
        self._quantize = quantize
        self._index_params = index_params
        self._dist_fn = _DISTANCE_FNS[metric]

        self._doc_ids: list[str] = []
        self._doc_id_to_idx: dict[str, int] = {}
        self._metadata: dict[str, dict[str, Any] | None] = {}
        self._vectors: array.array[float] = array.array("f")

        self._index_impl = self._create_index_impl()
        self._index_dirty: bool = True

        self._sq8_data: bytes | None = None
        self._sq8_min: array.array[float] | None = None
        self._sq8_step: array.array[float] | None = None
        self._sq8_norms: array.array[float] | None = None
        self._vectors_encoded: bool = False

    # -- properties ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._doc_ids)

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._doc_id_to_idx

    @property
    def doc_count(self) -> int:
        """Number of documents in the index."""
        return len(self._doc_ids)

    @property
    def dim(self) -> int:
        """Vector dimensionality."""
        return self._dim

    # -- document management -------------------------------------------------

    def add(
        self,
        doc_id: str,
        *,
        vector: list[float] | None = None,
        text: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a document to the index.

        Exactly one of *vector* or *text* must be provided. When *text* is
        given, the configured embedding function is used to produce the vector.

        Args:
            doc_id: Unique document identifier.
            vector: Pre-computed embedding vector.
            text: Text to embed via the configured embedding function.
            metadata: Arbitrary JSON-serializable metadata.

        Raises:
            ValueError: If ``doc_id`` already exists (use ``update`` instead),
                both or neither of *vector*/*text* are given, *text* is given
                without an embedding function, or vector dimension mismatches.
        """
        if doc_id in self._doc_id_to_idx:
            raise ValueError(
                f"Document {doc_id!r} already exists, use update() instead"
            )
        self._insert(doc_id, vector=vector, text=text, metadata=metadata)

    def add_many(
        self,
        doc_ids: list[str],
        *,
        vectors: list[list[float]] | None = None,
        texts: list[str] | None = None,
        metadatas: list[dict[str, Any] | None] | None = None,
    ) -> None:
        """Add multiple documents in one call.

        When *texts* is provided, they are embedded in a single batch call
        (one API round-trip instead of one per document).

        Args:
            doc_ids: Unique document identifiers.
            vectors: Pre-computed embedding vectors (one per doc).
            texts: Texts to embed via the configured embedding function.
            metadatas: Per-document metadata (``None`` entries are allowed).

        Raises:
            ValueError: If lengths mismatch, both or neither of
                *vectors*/*texts* are given, any ``doc_id`` already exists,
                or vector dimensions mismatch.
        """
        n = len(doc_ids)
        if vectors is not None and texts is not None:
            raise ValueError("Provide either vectors or texts, not both")
        if vectors is None and texts is None:
            raise ValueError("Provide either vectors or texts")

        if vectors is not None and len(vectors) != n:
            raise ValueError(f"Length mismatch: {n} doc_ids but {len(vectors)} vectors")
        if texts is not None and len(texts) != n:
            raise ValueError(f"Length mismatch: {n} doc_ids but {len(texts)} texts")
        if metadatas is not None and len(metadatas) != n:
            raise ValueError(
                f"Length mismatch: {n} doc_ids but {len(metadatas)} metadatas"
            )

        for doc_id in doc_ids:
            if doc_id in self._doc_id_to_idx:
                raise ValueError(
                    f"Document {doc_id!r} already exists, use update() instead"
                )

        if texts is not None:
            if self._embed is None:
                raise ValueError(
                    "texts requires an embedding function (pass embed= to constructor)"
                )
            vecs = self._embed(texts)
            vectors = vecs

        for i, doc_id in enumerate(doc_ids):
            vec = vectors[i]  # type: ignore[index]
            meta = metadatas[i] if metadatas is not None else None
            self._insert(doc_id, vector=vec, text=None, metadata=meta)

    def remove(self, doc_id: str) -> None:
        """Remove a document from the index.

        Args:
            doc_id: Document identifier to remove.

        Raises:
            KeyError: If ``doc_id`` does not exist.
        """
        if doc_id not in self._doc_id_to_idx:
            raise KeyError(f"Document {doc_id!r} not found")
        self._delete(doc_id)

    def update(
        self,
        doc_id: str,
        *,
        vector: list[float] | None = None,
        text: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update (replace) a document in the index.

        Args:
            doc_id: Document identifier to update.
            vector: New embedding vector.
            text: New text to embed.
            metadata: New metadata (replaces old metadata entirely).

        Raises:
            KeyError: If ``doc_id`` does not exist (use ``add`` instead).
        """
        if doc_id not in self._doc_id_to_idx:
            raise KeyError(f"Document {doc_id!r} not found, use add() instead")
        self._delete(doc_id)
        self._insert(doc_id, vector=vector, text=text, metadata=metadata)

    # -- search --------------------------------------------------------------

    def search(
        self,
        *,
        vector: list[float] | None = None,
        text: str | None = None,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[Result]:
        """Search for nearest neighbor documents.

        Exactly one of *vector* or *text* must be provided.

        Args:
            vector: Query vector.
            text: Query text (requires embedding function).
            top_k: Maximum number of results.
            filters: Metadata filters. Each key-value pair must match exactly
                in the document's metadata. Use a callable value for custom
                filter logic (e.g. ``{"year": lambda y: y > 2023}``).
                For approximate indices (IVF, LSH), filtering is applied
                post-retrieval — restrictive filters may return fewer than
                *top_k* results when matching documents are in non-probed
                clusters.

        Returns:
            List of :class:`Result` objects sorted by descending score.
        """
        if not self._doc_ids:
            return []

        query = self._resolve_vector(vector, text)

        if self._index_dirty and self._index_type != "flat":
            self.build_index()

        n_docs = len(self._doc_ids)
        fetch_k = min(top_k * 4, n_docs) if filters else top_k
        raw = self._index_impl.search(
            query,
            self._vectors,
            self._dim,
            self._dist_fn,
            n_docs,
            fetch_k,
            sq8_dist=self._quantized_distance(),
        )

        results: list[Result] = []
        for idx, dist in raw:
            did = self._doc_ids[idx]
            if filters and not self._match_filters(did, filters):
                continue

            if self._metric == "cosine":
                score = 1.0 - dist
            elif self._metric == "euclidean":
                score = 1.0 / (1.0 + dist)
            else:
                score = -dist

            results.append(
                Result(doc_id=did, score=score, metadata=self._metadata.get(did))
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def build_index(self) -> None:
        """Build or rebuild the internal search index.

        For ``"flat"`` indices this is a no-op. For ``"ivf"`` and ``"lsh"``
        indices this must be called after adding all documents and before
        searching (though :meth:`search` will auto-build if needed).

        When *quantize* is enabled this also computes the SQ8 data and
        releases the full-precision vector array; searches compare against
        the quantized vectors instead.
        """
        n_docs = len(self._doc_ids)
        if n_docs == 0:
            self._index_dirty = False
            return

        if self._vectors_encoded:
            self._materialize_vectors()

        if self._index_type == "ivf":
            impl = self._index_impl
            if isinstance(impl, _IVFIndex):
                impl.build(self._vectors, self._dim, self._dist_fn, n_docs, seed=42)
        elif self._index_type == "lsh":
            impl = self._index_impl
            if isinstance(impl, _LSHIndex):
                impl.build(self._vectors, self._dim, self._dist_fn, n_docs, seed=42)

        if self._quantize:
            self._sq8_data, self._sq8_min, self._sq8_step = _quantize_sq8(
                self._vectors, self._dim, n_docs
            )
            self._sq8_norms = _sq8_norms(
                self._sq8_data, self._sq8_min, self._sq8_step, self._dim, n_docs
            )
            self._vectors = array.array("f")
            self._vectors_encoded = True
        else:
            self._sq8_data = None
            self._sq8_min = None
            self._sq8_step = None
            self._sq8_norms = None
            self._vectors_encoded = False

        self._index_dirty = False

    # -- persistence ---------------------------------------------------------

    def save(self, path: str, fmt: str | None = None) -> None:
        """Save the index to disk.

        Args:
            path: File path to save to.
            fmt: ``"json"`` or ``"sqlite"``. If ``None``, inferred from
                file extension (``.db`` -> sqlite, otherwise json).
        """
        fmt = self._resolve_format(path, fmt)
        if fmt == "json":
            self._save_json(path)
        else:
            self._save_sqlite(path)

    @classmethod
    def load(cls, path: str) -> SemanticIndex:
        """Load an index from disk.

        Auto-detects format by inspecting the file header.

        Args:
            path: File path to load from.

        Returns:
            A new :class:`SemanticIndex` instance with restored state.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Index file not found: {path}")

        with open(p, "rb") as f:
            header = f.read(16)

        if header.startswith(cls._SQLITE_MAGIC):
            return cls._load_sqlite(path)
        return cls._load_json(path)

    # -- internal: vector resolution -----------------------------------------

    def _resolve_vector(
        self,
        vector: list[float] | None,
        text: str | None,
    ) -> list[float]:
        """Resolve a vector from explicit vector or text input.

        Args:
            vector: Pre-computed vector.
            text: Text to embed.

        Returns:
            Resolved vector (normalized if metric is cosine).

        Raises:
            ValueError: If both or neither are provided, text is given
                without an embedding function, or dimension mismatches.
        """
        if vector is not None and text is not None:
            raise ValueError("Provide either vector or text, not both")
        if vector is None and text is None:
            raise ValueError("Provide either vector or text")

        if text is not None:
            if self._embed is None:
                raise ValueError(
                    "text requires an embedding function (pass embed= to constructor)"
                )
            vecs = self._embed([text])
            vector = vecs[0]

        if len(vector) != self._dim:  # type: ignore[arg-type]
            raise ValueError(
                f"Expected vector of dimension {self._dim}, got {len(vector)}"  # type: ignore[arg-type]
            )

        if self._metric == "cosine":
            return _normalize(vector)  # type: ignore[arg-type]
        return list(vector)  # type: ignore[arg-type]

    # -- internal: insert / delete -------------------------------------------

    def _insert(
        self,
        doc_id: str,
        *,
        vector: list[float] | None,
        text: str | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        """Insert a document into internal storage."""
        vec = self._resolve_vector(vector, text)
        self._materialize_vectors()

        idx = len(self._doc_ids)
        self._doc_ids.append(doc_id)
        self._doc_id_to_idx[doc_id] = idx
        self._metadata[doc_id] = metadata
        self._vectors.extend(array.array("f", vec))
        self._index_dirty = True

    def _delete(self, doc_id: str) -> None:
        """Delete a document using swap-and-pop for O(1) removal."""
        self._materialize_vectors()
        idx = self._doc_id_to_idx[doc_id]
        last_idx = len(self._doc_ids) - 1

        if idx != last_idx:
            last_doc_id = self._doc_ids[last_idx]

            self._doc_ids[idx] = last_doc_id
            self._doc_id_to_idx[last_doc_id] = idx

            src_start = last_idx * self._dim
            dst_start = idx * self._dim
            for j in range(self._dim):
                self._vectors[dst_start + j] = self._vectors[src_start + j]

        self._doc_ids.pop()
        del self._doc_id_to_idx[doc_id]
        del self._metadata[doc_id]

        trim_start = len(self._doc_ids) * self._dim
        del self._vectors[trim_start:]

        self._index_dirty = True

    # -- internal: index construction ----------------------------------------

    def _quantized_distance(
        self,
    ) -> Callable[[array.array[float], list[float], int, int], float] | None:
        """Return a distance function over SQ8 storage, if it is active."""
        if (
            not self._vectors_encoded
            or self._sq8_data is None
            or self._sq8_min is None
            or self._sq8_step is None
        ):
            return None
        if self._sq8_norms is None:
            self._sq8_norms = _sq8_norms(
                self._sq8_data,
                self._sq8_min,
                self._sq8_step,
                self._dim,
                len(self._doc_ids),
            )
        return _sq8_distance_factory(
            self._metric,
            self._sq8_data,
            self._sq8_min,
            self._sq8_step,
            self._sq8_norms,
        )

    def _materialize_vectors(self) -> None:
        """Rebuild full-precision vectors from SQ8 storage.

        Adding or deleting a document mutates the float array in place, so a
        quantized index first reconstructs it.  The quantized copy is dropped;
        the next :meth:`build_index` call re-quantizes the updated data.
        """
        if not self._vectors_encoded:
            return
        if self._sq8_data is None or self._sq8_min is None or self._sq8_step is None:
            self._vectors_encoded = False
            return

        vectors = _dequantize_sq8_all(
            self._sq8_data,
            self._sq8_min,
            self._sq8_step,
            self._dim,
            len(self._doc_ids),
        )
        if self._metric == "cosine":
            for i in range(len(self._doc_ids)):
                base = i * self._dim
                norm = math.sqrt(sum(v * v for v in vectors[base : base + self._dim]))
                if norm > 0.0:
                    for j in range(self._dim):
                        vectors[base + j] /= norm

        self._vectors = vectors
        self._sq8_data = None
        self._sq8_min = None
        self._sq8_step = None
        self._sq8_norms = None
        self._vectors_encoded = False

    def _create_index_impl(self) -> _FlatIndex | _IVFIndex | _LSHIndex:
        """Create the internal index implementation from config."""
        if self._index_type == "ivf":
            return _IVFIndex(
                n_clusters=self._index_params.get("n_clusters", 64),
                nprobe=self._index_params.get("nprobe", 4),
            )
        if self._index_type == "lsh":
            return _LSHIndex(
                n_tables=self._index_params.get("n_tables", 8),
                n_hash_funcs=self._index_params.get("n_hash_funcs", 12),
            )
        return _FlatIndex()

    # -- internal: metadata filtering ----------------------------------------

    def _match_filters(self, doc_id: str, filters: dict[str, Any]) -> bool:
        """Check if a document's metadata matches all filters."""
        meta = self._metadata.get(doc_id)
        if meta is None:
            return False

        for key, expected in filters.items():
            actual = meta.get(key)
            if callable(expected):
                if not expected(actual):
                    return False
            elif actual != expected:
                return False

        return True

    # -- internal: persistence helpers ---------------------------------------

    @staticmethod
    def _resolve_format(path: str, fmt: str | None) -> str:
        """Determine save format from explicit arg or file extension."""
        if fmt is not None:
            if fmt not in ("json", "sqlite"):
                raise ValueError(f"Unknown format {fmt!r}, expected 'json' or 'sqlite'")
            return fmt
        if Path(path).suffix in (".db", ".sqlite", ".sqlite3"):
            return "sqlite"
        return "json"

    # -- internal: persistence (JSON) ----------------------------------------

    def _to_dict(self) -> dict[str, Any]:
        """Serialize index state to a JSON-compatible dict."""
        vectors_b64 = base64.b64encode(self._vectors.tobytes()).decode("ascii")

        index_state = None
        if not self._index_dirty:
            index_state = self._index_impl.to_dict()

        quantization = None
        if self._sq8_data is not None and self._sq8_min is not None:
            quantization = {
                "data_b64": base64.b64encode(self._sq8_data).decode("ascii"),
                "min_vals_b64": base64.b64encode(self._sq8_min.tobytes()).decode(
                    "ascii"
                ),
                "step_vals_b64": base64.b64encode(
                    self._sq8_step.tobytes()  # type: ignore[union-attr]
                ).decode("ascii"),
            }

        metadata: dict[str, dict[str, Any] | None] = {}
        for doc_id in self._doc_ids:
            metadata[doc_id] = self._metadata.get(doc_id)

        return {
            "version": _VERSION,
            "config": {
                "dim": self._dim,
                "metric": self._metric,
                "index_type": self._index_type,
                "quantize": self._quantize,
                "index_params": self._index_params,
            },
            "doc_ids": list(self._doc_ids),
            "metadata": metadata,
            "vectors_b64": vectors_b64,
            "index_state": index_state,
            "quantization": quantization,
        }

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> SemanticIndex:
        """Restore index state from a deserialized dict."""
        config = data["config"]
        instance = cls(
            dim=config["dim"],
            metric=config["metric"],
            index_type=config["index_type"],
            quantize=config.get("quantize", False),
            **config.get("index_params", {}),
        )

        instance._doc_ids = data["doc_ids"]
        instance._doc_id_to_idx = {
            doc_id: i for i, doc_id in enumerate(instance._doc_ids)
        }
        instance._metadata = data.get("metadata", {})

        raw = base64.b64decode(data["vectors_b64"])
        instance._vectors = array.array("f")
        instance._vectors.frombytes(raw)

        idx_state = data.get("index_state")
        if idx_state is not None:
            idx_type = idx_state.get("type", "flat")
            if idx_type == "ivf":
                instance._index_impl = _IVFIndex.from_dict(idx_state)
            elif idx_type == "lsh":
                instance._index_impl = _LSHIndex.from_dict(idx_state)
            else:
                instance._index_impl = _FlatIndex.from_dict(idx_state)
            instance._index_dirty = False
        else:
            instance._index_dirty = True

        quant = data.get("quantization")
        if quant is not None:
            instance._sq8_data = base64.b64decode(quant["data_b64"])
            instance._sq8_min = array.array("f")
            instance._sq8_min.frombytes(base64.b64decode(quant["min_vals_b64"]))
            instance._sq8_step = array.array("f")
            instance._sq8_step.frombytes(base64.b64decode(quant["step_vals_b64"]))
            if len(instance._vectors) == 0:
                instance._vectors_encoded = True

        return instance

    def _save_json(self, path: str) -> None:
        """Save index as JSON."""
        data = self._to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def _load_json(cls, path: str) -> SemanticIndex:
        """Load index from JSON."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls._from_dict(data)

    # -- internal: persistence (SQLite) --------------------------------------

    def _save_sqlite(self, path: str) -> None:
        """Save index as SQLite database."""
        p = Path(path)
        if p.exists():
            p.unlink()

        conn = sqlite3.connect(path)
        try:
            cur = conn.cursor()
            cur.executescript("""
                CREATE TABLE config (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE docs (
                    idx      INTEGER PRIMARY KEY,
                    doc_id   TEXT UNIQUE NOT NULL,
                    metadata TEXT
                );
                CREATE TABLE vectors (
                    idx  INTEGER PRIMARY KEY,
                    data BLOB NOT NULL
                );
                CREATE TABLE index_state (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
            """)

            config = {
                "version": _VERSION,
                "dim": self._dim,
                "metric": self._metric,
                "index_type": self._index_type,
                "quantize": self._quantize,
                "index_params": self._index_params,
            }
            for key, value in config.items():
                cur.execute(
                    "INSERT INTO config (key, value) VALUES (?, ?)",
                    (key, json.dumps(value)),
                )

            dim = self._dim
            all_bytes = self._vectors.tobytes()
            bpv = dim * 4
            for i, doc_id in enumerate(self._doc_ids):
                meta = self._metadata.get(doc_id)
                cur.execute(
                    "INSERT INTO docs (idx, doc_id, metadata) VALUES (?, ?, ?)",
                    (
                        i,
                        doc_id,
                        json.dumps(meta) if meta is not None else None,
                    ),
                )

                if not self._vectors_encoded:
                    start = i * bpv
                    vec_bytes = all_bytes[start : start + bpv]
                    cur.execute(
                        "INSERT INTO vectors (idx, data) VALUES (?, ?)",
                        (i, vec_bytes),
                    )

            if not self._index_dirty:
                idx_dict = self._index_impl.to_dict()
                for key, value in idx_dict.items():
                    cur.execute(
                        "INSERT INTO index_state (key, value) VALUES (?, ?)",
                        (key, json.dumps(value)),
                    )

            if self._sq8_data is not None and self._sq8_min is not None:
                cur.execute(
                    "INSERT INTO config (key, value) VALUES (?, ?)",
                    (
                        "sq8_data_b64",
                        json.dumps(base64.b64encode(self._sq8_data).decode("ascii")),
                    ),
                )
                cur.execute(
                    "INSERT INTO config (key, value) VALUES (?, ?)",
                    (
                        "sq8_min_b64",
                        json.dumps(
                            base64.b64encode(self._sq8_min.tobytes()).decode("ascii")
                        ),
                    ),
                )
                cur.execute(
                    "INSERT INTO config (key, value) VALUES (?, ?)",
                    (
                        "sq8_step_b64",
                        json.dumps(
                            base64.b64encode(
                                self._sq8_step.tobytes()  # type: ignore[union-attr]
                            ).decode("ascii")
                        ),
                    ),
                )

            conn.commit()
        finally:
            conn.close()

    @classmethod
    def _load_sqlite(cls, path: str) -> SemanticIndex:
        """Load index from SQLite database."""
        conn = sqlite3.connect(path)
        try:
            cur = conn.cursor()

            config: dict[str, Any] = {}
            for key, value in cur.execute("SELECT key, value FROM config"):
                config[key] = json.loads(value)

            index_params = config.get("index_params", {})
            if isinstance(index_params, str):
                index_params = json.loads(index_params)

            instance = cls(
                dim=config["dim"],
                metric=config["metric"],
                index_type=config["index_type"],
                quantize=config.get("quantize", False),
                **index_params,
            )

            rows = list(
                cur.execute("SELECT idx, doc_id, metadata FROM docs ORDER BY idx")
            )
            for idx, doc_id, metadata_json in rows:
                instance._doc_ids.append(doc_id)
                instance._doc_id_to_idx[doc_id] = idx
                instance._metadata[doc_id] = (
                    json.loads(metadata_json) if metadata_json is not None else None
                )

            chunks = [
                data for (data,) in cur.execute("SELECT data FROM vectors ORDER BY idx")
            ]
            all_bytes = b"".join(chunks)
            instance._vectors = array.array("f")
            if all_bytes:
                instance._vectors.frombytes(all_bytes)

            idx_state: dict[str, Any] = {}
            for key, value in cur.execute("SELECT key, value FROM index_state"):
                idx_state[key] = json.loads(value)

            if idx_state:
                idx_type = idx_state.get("type", "flat")
                if idx_type == "ivf":
                    instance._index_impl = _IVFIndex.from_dict(idx_state)
                elif idx_type == "lsh":
                    instance._index_impl = _LSHIndex.from_dict(idx_state)
                else:
                    instance._index_impl = _FlatIndex.from_dict(idx_state)
                instance._index_dirty = False
            else:
                instance._index_dirty = True

            sq8_data_b64 = config.get("sq8_data_b64")
            if sq8_data_b64 is not None:
                instance._sq8_data = base64.b64decode(sq8_data_b64)
                instance._sq8_min = array.array("f")
                instance._sq8_min.frombytes(base64.b64decode(config["sq8_min_b64"]))
                instance._sq8_step = array.array("f")
                instance._sq8_step.frombytes(base64.b64decode(config["sq8_step_b64"]))
                if len(instance._vectors) == 0:
                    instance._vectors_encoded = True

            return instance
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Similarity & Re-ranking Utilities
# ---------------------------------------------------------------------------


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    ``sim(a, b) = dot(a, b) / (||a|| * ||b||)``

    Returns 0.0 when either vector has zero norm.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Similarity in [-1.0, 1.0].
    """
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    denom = math.sqrt(norm_a) * math.sqrt(norm_b)
    if denom == 0.0:
        return 0.0
    return dot / denom


def rrf(
    *result_lists: list[Result],
    k: int = 60,
    top_k: int | None = None,
    weights: list[float] | None = None,
) -> list[Result]:
    """Reciprocal Rank Fusion of multiple ranked result lists.

    Combines results from different retrieval systems (e.g. BM25 sparse
    and dense vector search) using the formula::

        Score(d) = Sigma weight_i / (k + rank_i)

    where *rank_i* is the 1-based position of document *d* in the *i*-th
    list (Cormack et al., SIGIR 2009).

    Example::

        sparse = sparse_idx.search("quick fox", top_k=20)
        dense  = semantic_idx.search(vector=query_vec, top_k=20)
        fused  = rrf(sparse, dense, k=60, top_k=10)

    Args:
        *result_lists: One or more lists of :class:`Result` objects. Each
            list should be pre-sorted by descending relevance.
        k: RRF constant controlling rank sensitivity. Higher values flatten
            rank differences. Default 60 (per Cormack et al.).
        top_k: Maximum number of results to return. ``None`` returns all.
        weights: Per-list weights. When ``None``, all lists are weighted
            equally (1.0). Length must match the number of result lists.

    Returns:
        Fused list of :class:`Result` sorted by descending RRF score.
        Metadata is taken from the occurrence with the highest original
        score.

    Raises:
        ValueError: If no result lists are provided, ``weights`` length
            does not match, or ``k`` is not positive.
    """
    if not result_lists:
        raise ValueError("rrf() requires at least one result list")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if weights is not None and len(weights) != len(result_lists):
        raise ValueError(
            f"weights length ({len(weights)}) must match "
            f"number of result lists ({len(result_lists)})"
        )

    rrf_scores: dict[str, float] = {}
    best_meta: dict[str, dict[str, Any] | None] = {}
    best_original: dict[str, float] = {}

    for list_idx, results in enumerate(result_lists):
        w = weights[list_idx] if weights is not None else 1.0
        seen_in_list: set[str] = set()
        for rank_0, r in enumerate(results):
            if r.doc_id in seen_in_list:
                continue
            seen_in_list.add(r.doc_id)
            rrf_scores[r.doc_id] = rrf_scores.get(r.doc_id, 0.0) + w / (k + rank_0 + 1)
            if r.doc_id not in best_original or r.score > best_original[r.doc_id]:
                best_original[r.doc_id] = r.score
                best_meta[r.doc_id] = r.metadata

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    if top_k is not None:
        ranked = ranked[:top_k]

    return [
        Result(doc_id=did, score=score, metadata=best_meta[did])
        for did, score in ranked
    ]


def mmr(
    results: list[Result],
    similarity_fn: Callable[[Result, Result], float],
    *,
    lambda_: float = 0.5,
    top_k: int | None = None,
) -> list[Result]:
    """Maximal Marginal Relevance re-ranking for result diversification.

    Greedily selects results that balance relevance with diversity::

        MMR(d) = argmax[lambda * rel(d) - (1 - lambda) * max sim(d, d_j)]

    where *d_j* are already-selected results. Relevance scores are min-max
    normalized to [0, 1] internally so they are comparable to similarity.

    Example::

        vecs = {r.doc_id: stored_vectors[r.doc_id] for r in results}
        sim = lambda a, b: cosine_similarity(vecs[a.doc_id], vecs[b.doc_id])
        diverse = mmr(results, sim, lambda_=0.7, top_k=5)

    Args:
        results: Candidate results, typically from :meth:`SemanticIndex.search`
            or :func:`rrf`.
        similarity_fn: A callable ``(Result, Result) -> float`` returning a
            similarity score in [0, 1]. See :func:`cosine_similarity` for
            a vector-based helper.
        lambda_: Trade-off between relevance (1.0) and diversity (0.0).
            Default 0.5.
        top_k: Number of results to select. ``None`` selects all (full
            re-ranking).

    Returns:
        Re-ranked list of :class:`Result` with scores set to their MMR
        scores.

    Raises:
        ValueError: If ``lambda_`` is not in [0, 1].
    """
    if not (0.0 <= lambda_ <= 1.0):
        raise ValueError(f"lambda_ must be in [0, 1], got {lambda_}")
    if not results:
        return []

    n = len(results)
    select_k = n if top_k is None else min(top_k, n)

    max_score = max(r.score for r in results)
    min_score = min(r.score for r in results)
    score_range = max_score - min_score
    if score_range > 0:
        norm = {r.doc_id: (r.score - min_score) / score_range for r in results}
    else:
        norm = {r.doc_id: (1.0 if max_score > 0 else 0.0) for r in results}

    candidates = list(results)
    selected: list[Result] = []

    while len(selected) < select_k and candidates:
        best_mmr = -math.inf
        best_idx = 0

        for idx, cand in enumerate(candidates):
            rel = norm[cand.doc_id]
            if selected:
                max_sim = max(similarity_fn(cand, s) for s in selected)
            else:
                max_sim = 0.0
            mmr_score = lambda_ * rel - (1.0 - lambda_) * max_sim
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx

        chosen = candidates.pop(best_idx)
        selected.append(
            Result(doc_id=chosen.doc_id, score=best_mmr, metadata=chosen.metadata)
        )

    return selected
