[![GitHub release](https://img.shields.io/github/v/release/Oaklight/searxng-bm25-reranker?color=green)](https://github.com/Oaklight/searxng-bm25-reranker/releases/latest)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-green.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python](https://img.shields.io/badge/Python-≥3.11-blue.svg)](https://www.python.org/)
[![SearXNG](https://img.shields.io/badge/SearXNG-plugin-orange.svg)](https://github.com/searxng/searxng)

**English** | [中文](README.md)

# SearXNG BM25 Reranker

An external SearXNG plugin that reranks search results using BM25 text relevance scoring to improve search quality.

## Features

- **BM25F Multi-Field Weighting** — Title weight 2.0, content weight 1.0, prioritizing title-matching results
- **RRF Fusion Ranking** — Combines BM25 scores with original engine rankings via Reciprocal Rank Fusion, rather than replacing them
- **CJK Tokenization** — Built-in zero-dependency CJK tokenizer (unigram + bigram), no jieba required
- **Zero External Dependencies** — Core BM25 engine from [zerodep/sparse_search](https://github.com/Oaklight/zerodep), pure standard library
- **Plug and Play** — Standard SearXNG external plugin, deployable via volume mount or pip install

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

### Option 1: Volume Mount (Recommended for Quick Deployment)

1. Clone and copy the plugin code to your server:

```bash
git clone https://github.com/Oaklight/searxng-bm25-reranker.git
cp -r searxng-bm25-reranker/src/searxng_bm25_reranker /path/to/plugins/
```

2. Update `compose.yaml`:

```yaml
services:
  searxng:
    volumes:
      - /path/to/plugins:/usr/local/searxng/plugins:ro
    environment:
      - PYTHONPATH=/usr/local/searxng/plugins
```

3. Register the plugin in `settings.yml`:

```yaml
plugins:
  searxng_bm25_reranker.SXNGPlugin:
    active: true
```

4. Restart the container:

```bash
docker compose restart searxng
```

### Option 2: pip Install (For Custom Images)

```dockerfile
FROM searxng/searxng:latest
RUN pip install --no-cache-dir searxng-bm25-reranker
```

You still need to register the plugin in `settings.yml`.

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
