# SearXNG Rerankers

**English** | [中文](README_zh.md)

[![PyPI bm25](https://img.shields.io/pypi/v/searxng-bm25-reranker?color=green&label=PyPI%20bm25)](https://pypi.org/project/searxng-bm25-reranker/)
[![PyPI sem](https://img.shields.io/pypi/v/searxng-sem-reranker?color=green&label=PyPI%20sem)](https://pypi.org/project/searxng-sem-reranker/)
[![TestPyPI bm25](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Ftest.pypi.org%2Fpypi%2Fsearxng-bm25-reranker%2Fjson&query=%24.info.version&label=TestPyPI%20bm25&color=blue&logo=pypi&logoColor=white)](https://test.pypi.org/project/searxng-bm25-reranker/)
[![TestPyPI sem](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Ftest.pypi.org%2Fpypi%2Fsearxng-sem-reranker%2Fjson&query=%24.info.version&label=TestPyPI%20sem&color=blue&logo=pypi&logoColor=white)](https://test.pypi.org/project/searxng-sem-reranker/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-green.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python](https://img.shields.io/badge/Python-≥3.10-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/docker/pulls/oaklight/searxng?label=Docker%20pulls)](https://hub.docker.com/r/oaklight/searxng)

Two independent SearXNG reranking plugins that share one repository and one
optional coordination protocol.

| Package | Distribution | Plugin id | Signal |
| --- | --- | --- | --- |
| [`packages/bm25_reranker`](packages/bm25_reranker) | `searxng-bm25-reranker` | `bm25_reranker` | BM25F text relevance |
| [`packages/sem_reranker`](packages/sem_reranker) | `searxng-sem-reranker` | `sem_reranker` | Embedding and cross-encoder scores |

Both are runtime-dependency-free. The only third-party code is copied into
`_vendor/` with the [zerodep](https://github.com/Oaklight/zerodep) CLI.

## Install

```bash
pip install searxng-bm25-reranker
pip install searxng-sem-reranker
```

For local development:

```bash
make install-dev
```

## Register the plugins

Add the plugin classes under `plugins:` and their options in top-level blocks
keyed by the plugin id:

```yaml
plugins:
  searxng_bm25_reranker.SXNGPlugin:
    active: true
  searxng_sem_reranker.SXNGPlugin:
    active: true

bm25_reranker:
  rrf_k: 20
  url_weight: 0.5

sem_reranker:
  embedding:
    enabled: false            # optional; enable for vector similarity
    format: openai
    url: http://localhost:8000/v1/embeddings
    model: BAAI/bge-m3
    api_key_env: ""           # set when the local server uses --api-key

  rerank:
    enabled: true             # optional; enable for cross-encoder reranking
    format: jina
    url: http://localhost:8000/v1/rerank
    model: BAAI/bge-reranker-v2-m3
    api_key_env: ""           # set when the local server uses --api-key
    top_k: 10
    timeout_ms: 3000

rerankers:
  coordinator: sem_reranker
```

Both stages are optional. The example enables rerank only; embedding is off
by default. Set `embedding.enabled: true` to add vector similarity. Embedding
also supports `query_prefix`, `document_prefix`, and `max_tokens`. Each stage
has its own `format`, `url`, `api_key_env`, and `api_key`. Embedding and rerank
can use different hosts and different keys. The full schema is in the
[sem_reranker README](packages/sem_reranker/README_en.md).

Two SearXNG rules matter:

- `plugins:` accepts only `active` for each entry. Any other key is passed to
  `PluginCfg(**settings)` and raises `TypeError`. Put plugin tunables in the
  top-level block instead.
- A user `plugins:` block replaces SearXNG's default plugin list. List every
  plugin you want, including the built-in ones, or keep the default list and
  merge only what you need.

## Coordination

SearXNG stores plugins in a set, so `post_search` order is not stable. Both
plugins write `positions`, so exactly one plugin may do the work.

Set `rerankers.coordinator` to `bm25_reranker` or `sem_reranker`. The named
plugin computes its own signal, optionally imports the other package's scoring
function, fuses engine order + BM25 + neural scores with RRF, and rewrites
positions once. The other plugin returns without a network call or a position
write.

If the coordinator is not active, the active plugin coordinates and logs a
warning. If both are active and no coordinator is set, `sem_reranker`
coordinates. If the other package is not installed, the coordinator uses its
own signal only.

When BM25 is the coordinator, the semantic call can be gated:

```yaml
rerankers:
  coordinator: bm25_reranker
  escalate:
    enabled: true
    min_coverage: 0.5
    min_margin: 0.1
    min_scored: 2
    top_k: 10
```

BM25 runs first. The gate opens when lexical coverage is low, the top-two score
margin is small, or too few results have a BM25 score. Only then is
`sem_reranker` called. With `enabled: false` (the default), sem is called on
every request when it is active.

## Development

```bash
make test
make stress
make lint
make build
make vendor
```

Run `pytest` from the repository root so the shared `conftest.py` stub loads for
both packages. `make stress` runs the large-input and failure-path tests; the
default run excludes them.

## License

[AGPL-3.0](LICENSE).
