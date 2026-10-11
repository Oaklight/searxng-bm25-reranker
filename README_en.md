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
- **CJK Tokenization** — Built-in zero-dependency CJK tokenizer (unigram + bigram) covering Chinese ideographs and Japanese kana, no jieba required
- **HTML-Aware Scoring** — Strips markup and decodes entities before tokenizing, so tag noise (`div`, `href`, `www`) never dilutes relevance
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
                                   RRF fusion (engine ranking + BM25 ranking, k=20)
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
    restart: always
    ports:
      - "127.0.0.1:8080:8080"
    volumes:
      - ./searxng:/etc/searxng:rw
    environment:
      - SEARXNG_BASE_URL=https://search.example.com
```

### Option 2: Inline Build in Docker Compose

Build from the official SearXNG image with the plugin installed at compose time:

```yaml
services:
  searxng:
    build:
      dockerfile_inline: |
        FROM searxng/searxng:latest
        RUN /usr/local/searxng/.venv/bin/python3 -m ensurepip && \
            /usr/local/searxng/.venv/bin/python3 -m pip install --no-cache-dir searxng-bm25-reranker && \
            /usr/local/searxng/.venv/bin/python3 -m pip uninstall -y pip
    restart: always
    ports:
      - "127.0.0.1:8080:8080"
    volumes:
      - ./searxng:/etc/searxng:rw
    environment:
      - SEARXNG_BASE_URL=https://search.example.com
```

### Plugin Registration

For both options, register the plugin in `settings.yml`:

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

The plugin works out of the box with no additional configuration. Every value below can be
overridden per instance in `settings.yml` in a top-level block keyed by the plugin id; all
must be numbers `>= 1`, and `rrf_k` must be a whole number.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `title_weight` | `2.0` | BM25F weight for the title field |
| `content_weight` | `1.0` | BM25F weight for the content field |
| `rrf_k` | `20` | RRF fusion constant |
| `engine_weight` | `1.0` | Weight of the original engine ranking in the fusion |
| `bm25_weight` | `1.0` | Weight of the BM25 ranking in the fusion |

Fixed and not exposed: Okapi BM25 with `k1=1.5`, `b=0.75`, `delta=0`.

`rrf_k` defaults to 20 rather than the Cormack et al. value of 60 because that constant was
tuned on retrieval runs with thousands of candidates, while a SearXNG page holds roughly 30
results. At 60, the gap between rank 1 and rank 30 collapses to 1.48x and mutes the relevance
signal. A value far above 20 flattens the differences further and eventually reduces the
rerank to engine order.

```yaml
bm25_reranker:
  rrf_k: 10
  bm25_weight: 2.0
```

The block is top-level and keyed by the plugin id -- SearXNG's convention for
plugin-specific options (the in-tree hostnames plugin works the same way).
`PluginCfg` only carries `active`, so putting these keys under `plugins:`
makes SearXNG fail to start rather than fall back.

Raising `bm25_weight` lets text relevance outweigh the original engine order more aggressively;
raising `engine_weight` does the opposite.

### URL priority

`url_priority` multiplies the fused score of every result whose URL matches a
regular expression, so a domain can be pushed up or down without touching the
ranking code. A multiplier below `1.0` demotes, above `1.0` promotes, and `0.0`
sends the result to the bottom. Every matching pattern applies, so overlapping
entries compound.

The block goes at the **top level** of `settings.yml`, keyed by the plugin id.
It does not go under `plugins:`, which only accepts `active`:

```yaml
bm25_reranker:
  url_priority:
    '(.*\.)?arxiv\.org$': 1.5
    '(.*\.)?github\.com$': 1.2
    '(.*\.)?pinterest\.com$': 0.1
    '(.*\.)?facebook\.com$': 0.0

plugins:
  searxng_bm25_reranker.SXNGPlugin:
    active: true
```

An invalid regular expression, or a multiplier that is not a number or is
negative, is skipped with a warning in the log. A YAML boolean counts as "not a
number", so writing `false` does not switch a rule off; it is skipped too. A
typo therefore degrades to "no multiplier" instead of breaking search.

Patterns are matched with `re.search`, so an unanchored pattern matches
anywhere in the URL, query string included. Anchor with `$` as the examples do
to mean the whole domain. The setting is read when the plugin loads, so changing
it needs a restart.

## Project Structure

```
src/searxng_bm25_reranker/
├── __init__.py          # SXNGPlugin class, post_search reranking logic
├── _tokenizer.py        # CJK-aware tokenizer (unigram + bigram)
└── _vendor/
    └── sparse_search.py # BM25 engine from zerodep (vendored)

tests/
├── conftest.py          # SearXNG API stub (it is not a dependency)
├── conftest.py          # SearXNG API stub (it is not a dependency)
├── test_rerank.py
├── test_tokenizer.py
└── test_url_priority.py
```

Run the tests with `pytest`.

## Acknowledgements

- [SearXNG](https://github.com/searxng/searxng) — Privacy-respecting metasearch engine
- [zerodep/sparse_search](https://github.com/Oaklight/zerodep) — Zero-dependency BM25 full-text search engine

## License

[AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0)
