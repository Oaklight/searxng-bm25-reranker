[![GitHub release](https://img.shields.io/github/v/release/Oaklight/searxng-bm25-reranker?color=green)](https://github.com/Oaklight/searxng-bm25-reranker/releases/latest)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-green.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python](https://img.shields.io/badge/Python-≥3.10-blue.svg)](https://www.python.org/)
[![SearXNG](https://img.shields.io/badge/SearXNG-plugin-orange.svg)](https://github.com/searxng/searxng)
[![Docker](https://img.shields.io/docker/pulls/oaklight/searxng?label=Docker%20pulls)](https://hub.docker.com/r/oaklight/searxng)

**English** | [中文](README_zh.md)

# SearXNG BM25 Reranker

An external SearXNG plugin that reranks search results using BM25 text relevance scoring to improve search quality.

## Features

- **BM25F Multi-Field Weighting** — Title weight 2.0, content weight 1.0, prioritizing title-matching results
- **RRF Fusion Ranking** — Combines BM25 scores with original engine rankings via Reciprocal Rank Fusion, rather than replacing them
- **CJK Tokenization** — Built-in zero-dependency CJK tokenizer (unigram + bigram), no jieba required
- **Zero External Dependencies** — Core BM25 engine from [zerodep/sparse_search](https://github.com/Oaklight/zerodep), pure standard library
- **Plug and Play** — Standard SearXNG external plugin, deployable via Docker image or pip install

## How It Works

```
Search request → Engines return results → [post_search hook]
                                                ↓
                                   Build temporary BM25F index (title + content)
                                                ↓
                                   BM25 retrieval with original query
                                                ↓
                                   RRF fusion (engine ranking + BM25 ranking, k=60)
                                                ↓
                                   Rewrite positions to influence scoring → Reranked results
```

The plugin hooks into the `post_search` phase, after all engine results are collected but before final scores are calculated. By rewriting each result's `positions` list, it influences SearXNG's built-in `calculate_score()` formula (`weight / position`), achieving non-invasive reranking.

## Installation

### Option 1: Pre-built Docker Image (Recommended)

Use the pre-built image from Docker Hub, which tracks `searxng/searxng:latest` with the plugin pre-installed:

```yaml
services:
  searxng:
    image: oaklight/searxng:latest
    # ... rest of your config
```

### Option 2: Inline Build in Docker Compose

Add the plugin to your existing `compose.yaml` without a separate Dockerfile:

```yaml
services:
  searxng:
    build:
      dockerfile_inline: |
        FROM searxng/searxng:latest
        RUN /usr/local/searxng/.venv/bin/python3 -m ensurepip && \
            /usr/local/searxng/.venv/bin/python3 -m pip install --no-cache-dir searxng-bm25-reranker && \
            /usr/local/searxng/.venv/bin/python3 -m pip uninstall -y pip
    # ... rest of your config
```

### Option 3: Custom Dockerfile

```dockerfile
FROM searxng/searxng:latest
RUN /usr/local/searxng/.venv/bin/python3 -m ensurepip && \
    /usr/local/searxng/.venv/bin/python3 -m pip install --no-cache-dir searxng-bm25-reranker && \
    /usr/local/searxng/.venv/bin/python3 -m pip uninstall -y pip
```

### Plugin Registration

For all options, register the plugin in `settings.yml`:

```yaml
plugins:
  searxng_bm25_reranker.SXNGPlugin:
    active: true
```

Then start or restart the container:

```bash
docker compose up -d
```

## Configuration

The plugin works out of the box with no additional configuration. Default parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| BM25 variant | `bm25` | Standard Okapi BM25 |
| k1 | `1.5` | Term frequency saturation |
| b | `0.75` | Document length normalization |
| title weight | `2.0` | BM25F weight for title field |
| content weight | `1.0` | BM25F weight for content field |
| RRF k | `60` | RRF fusion constant |

## Project Structure

```
src/searxng_bm25_reranker/
├── __init__.py          # SXNGPlugin class, post_search reranking logic
├── _tokenizer.py        # CJK-aware tokenizer (unigram + bigram)
└── _vendor/
    └── sparse_search.py # BM25 engine from zerodep (vendored)
```

## Acknowledgements

- [SearXNG](https://github.com/searxng/searxng) — Privacy-respecting metasearch engine
- [zerodep/sparse_search](https://github.com/Oaklight/zerodep) — Zero-dependency BM25 full-text search engine

## License

[AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0)
