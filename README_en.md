# SearXNG Rerankers

**English** | [中文](README_zh.md)

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

sem_reranker:
  rerank:
    enabled: true
    format: jina
    url: https://api.siliconflow.cn/v1/rerank
    model: BAAI/bge-reranker-v2-m3
    api_key_env: SILICONFLOW_API_KEY
    top_k: 10
    timeout_ms: 3000

rerankers:
  coordinator: sem_reranker
```

Each stage has its own `format`, `url`, `api_key_env`, and `api_key`.
Embedding and rerank can use different hosts and different keys.

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
make lint
make build
make vendor
```

Run `pytest` from the repository root so the shared `conftest.py` stub loads for
both packages.

## License

[AGPL-3.0](LICENSE).
