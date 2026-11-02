# SearXNG Semantic Reranker

**English** | [中文](README_zh.md)

[![PyPI](https://img.shields.io/pypi/v/searxng-sem-reranker?color=green)](https://pypi.org/project/searxng-sem-reranker/)
[![TestPyPI](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Ftest.pypi.org%2Fpypi%2Fsearxng-sem-reranker%2Fjson&query=%24.info.version&label=TestPyPI&color=blue&logo=pypi&logoColor=white)](https://test.pypi.org/project/searxng-sem-reranker/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-green.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python](https://img.shields.io/badge/Python-≥3.10-blue.svg)](https://www.python.org/)
[![SearXNG](https://img.shields.io/badge/SearXNG-plugin-orange.svg)](https://github.com/searxng/searxng)
[![Docker](https://img.shields.io/docker/pulls/oaklight/searxng?label=Docker%20pulls)](https://hub.docker.com/r/oaklight/searxng)

An external SearXNG plugin that reranks search results with a remote embedding
model, a remote cross-encoder reranker, or both. Model hosting is separate from
the SearXNG process: the plugin only needs an API URL, a key, and model names.

- **Two independent stages** — embedding similarity and cross-encoder reranking;
  both are off by default.
- **Several API formats** — OpenAI, Cohere, Jina, and Voyage request/response
  shapes. Point each stage at its own base URL and key.
- **Zero runtime dependencies** — HTTP, cache, retry, cosine, and RRF come from
  vendored [zerodep](https://github.com/Oaklight/zerodep) modules.
- **Safe fallback** — invalid or empty stage config is skipped, and a provider
  error keeps the previous ranking.
- **Works alone or with `bm25_reranker`** — one coordinator fuses engine order,
  BM25, and neural scores.

## Install

```bash
pip install searxng-sem-reranker
```

Register the plugin and enable at least one stage:

```yaml
plugins:
  searxng_sem_reranker.SXNGPlugin:
    active: true

sem_reranker:
  rerank:
    enabled: true
    format: jina
    url: http://localhost:8000/v1/rerank
    model: BAAI/bge-reranker-v2-m3
    api_key_env: ""                # set when the local server uses --api-key
    top_k: 10
    timeout_ms: 3000
```

`plugins:` accepts only `active`; all other options belong in the top-level
`sem_reranker:` block. A user `plugins:` block replaces SearXNG's default plugin
list, so include every plugin you want.

## Configuration

Both stages are optional and off by default. Enable either one, or both.

```yaml
sem_reranker:
  embedding:
    enabled: false                 # optional; set true to enable this stage
    format: openai                 # openai | cohere | jina | voyage
    url: http://localhost:8000/v1/embeddings
    model: BAAI/bge-m3             # or Qwen/Qwen3-Embedding-0.6B (32k)
    api_key_env: ""                # set when the local server uses --api-key
    api_key: ""
    weight: 1.0
    max_chars: 4000
    max_tokens: 0                  # 0 = no token cap; 512 recommended
    query_prefix: ""               # e.g. "query: "
    document_prefix: ""            # e.g. "passage: "
    timeout_ms: 3000

  rerank:
    enabled: false                 # optional; set true to enable this stage
    format: jina                   # jina | cohere | voyage
    url: http://localhost:8000/v1/rerank
    model: BAAI/bge-reranker-v2-m3 # or Qwen/Qwen3-Reranker-0.6B (32k)
    api_key_env: ""                # set when the local server uses --api-key
    api_key: ""
    top_k: 10
    weight: 1.0
    max_chars: 4000
    timeout_ms: 3000

  fusion:
    rrf_k: 20
    engine_weight: 1.0
    bm25_weight: 1.0
    neural_weight: 1.0

  cache:
    ttl: 300
    max_entries: 512
```

`query_prefix` and `document_prefix` apply to the embedding stage only. Set
them for models that expect an instruction prefix. `max_tokens` bounds the
embedding input with a script-aware estimate; `0` disables it and leaves
`max_chars` as the only cap.

The key is read from `api_key_env` first, then from `api_key`. The key is never
logged. To set a key directly, leave `api_key_env: ""` and put it in `api_key`:

```yaml
  rerank:
    api_key_env: ""
    api_key: "your-key"
```

Environment variables are preferred when `settings.yml` is shared, because
anyone who can read the file can read a literal key.

There are no provider presets: every stage field is explicit.

### Fallback behavior

A stage is skipped when `enabled` is false, when `format`, `url`, or `model`
is missing or invalid, or when `api_key_env` names an unset variable. A 400,
401, 403, 404, 405, or 422 response, or a malformed provider response, disables
that stage for 5 minutes. Changing
its URL, model, or key resets the disable. Timeouts, 429, and 5xx responses are
treated as transient; the next request retries. With no usable stage,
`score_neural` returns only `None`, so the coordinator keeps the BM25 or engine
order.

The example uses a vLLM server on `localhost:8000`. llama.cpp's server exposes
the same paths on `localhost:8080`; start it with `--embeddings` for the
embedding stage and `--reranking` for the rerank stage.

`embedding` and `rerank` are independent. They may use different `format`,
`url`, `model`, `api_key_env`, and `api_key` values. `url` is the full endpoint.

For example, embedding and rerank can come from different hosts with different
keys:

```yaml
sem_reranker:
  embedding:
    enabled: true
    format: openai
    url: https://embed-gateway.example/v1/embeddings
    model: BAAI/bge-m3
    api_key_env: EMBED_API_KEY

  rerank:
    enabled: true
    format: jina
    url: https://rerank-gateway.example/v1/rerank
    model: BAAI/bge-reranker-v2-m3
    api_key_env: RERANK_API_KEY
```

### Example models

These models can be served by vLLM, llama.cpp, or a hosted provider. Any model
served by the configured endpoint can be used.

| Stage | Model | Context |
| --- | --- | --- |
| embedding | `BAAI/bge-m3` | 8k |
| embedding | `Qwen/Qwen3-Embedding-0.6B` | 32k |
| rerank | `BAAI/bge-reranker-v2-m3` | 8k |
| rerank | `Qwen/Qwen3-Reranker-0.6B` | 32k |

The plugin does not assume vector dimension or a model-specific tokenizer.
`max_chars` truncates an over-long query or document before the request; normal
SearXNG snippets are far below the 8k limit.

### Provider formats

| Format | Stage | Request | Response |
| --- | --- | --- | --- |
| openai | embed | `model`, `input`, `encoding_format` | `data[].embedding` |
| cohere | embed | `model`, `texts`, `input_type`, `embedding_types` | `embeddings.float[]` |
| jina | embed | `model`, `input`, `task` | `data[].embedding` |
| voyage | embed | `model`, `input`, `input_type` | `data[].embedding` |
| jina | rerank | `model`, `query`, `documents`, `top_n` | `results[].{index,relevance_score}` |
| cohere | rerank | `model`, `query`, `documents`, `top_n` | `results[].{index,relevance_score}` |
| voyage | rerank | `model`, `query`, `documents`, `top_k` | `data[].{index,relevance_score}` |

The same `format` works against a self-hosted gateway that speaks that shape.
Point `url` at the gateway endpoint.

## Coordination with `bm25_reranker`

SearXNG plugin order is not stable, so only one plugin may rewrite `positions`.
Set the coordinator explicitly:

```yaml
rerankers:
  coordinator: sem_reranker
```

The coordinator computes all active signals and rewrites positions once. The
other plugin returns immediately. If the configured coordinator is not active,
the active plugin coordinates. If the other package is not installed, the
coordinator uses its own signal.

When BM25 is the coordinator, `rerankers.escalate` decides whether this plugin
is called. See the BM25 README for the coverage, margin, and scored-result
indicators.

## Latency

Measured from a residential host against SiliconFlow, warm connection:

| Call | Documents | Time |
| --- | --- | --- |
| rerank | 10 | ~0.83–1.09 s |
| rerank | 30 | ~1.1 s |
| embedding | 10 | ~2.07–2.20 s |
| embedding | 30 | ~2.6 s |

The rerank stage is the recommended default. The embedding stage is opt-in.

## Development

```bash
make install-dev
make test
```

Run `pytest` from the repository root so the shared `conftest.py` stub loads.
Set `SILICONFLOW_API_KEY` to run the live test; it is skipped otherwise.

## License

[AGPL-3.0](LICENSE).
