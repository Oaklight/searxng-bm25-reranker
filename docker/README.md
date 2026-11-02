# Docker

The image installs both distributions:

- `searxng-bm25-reranker`
- `searxng-sem-reranker`

The example `settings.yml` loads both plugins. BM25 does the work by default.
The semantic stages are off until you point their URLs at a local vLLM or
llama.cpp server and set `enabled: true`.

## Run

From the repository root:

```bash
SEARXNG_SECRET="$(openssl rand -hex 32)" \
  docker compose -f docker/docker-compose.yml up --build
```

Open <http://localhost:8080>. Set `SEARXNG_PORT` to use a different host port.

`SEARXNG_SECRET` must be set for a real deployment. The default `change-me`
is only for a local test.

## TestPyPI build

`BM25_VERSION` and `SEM_VERSION` take a PEP 440 specifier. Point
`PIP_INDEX_URL` at TestPyPI to test a pre-release without touching PyPI.

```bash
PIP_INDEX_URL=https://test.pypi.org/simple/ \
  BM25_VERSION="==0.4.0" \
  SEM_VERSION="==0.2.0" \
  SEARXNG_PORT=8088 \
  SEARXNG_SECRET="$(openssl rand -hex 32)" \
  docker compose -p searxng-test -f docker/docker-compose.yml up --build -d
```

## Local model server

vLLM:

```bash
vllm serve BAAI/bge-m3 --task embed --port 8000
vllm serve BAAI/bge-reranker-v2-m3 --task score --port 8001
```

llama.cpp:

```bash
llama-server -m bge-m3.gguf --embeddings --port 8080
llama-server -m bge-reranker-v2-m3.gguf --reranking --port 8081
```

The plugin `url` is the full endpoint. For vLLM on the host, use
`http://host.docker.internal:8000/v1/embeddings` and
`http://host.docker.internal:8001/v1/rerank`. The compose file maps
`host.docker.internal` on Linux.

Both servers run as separate processes on different ports. The embedding and
rerank stages have independent `format`, `url`, `model`, and key settings.

## Plugin list

SearXNG replaces its default plugin list when `settings.yml` contains a
`plugins:` block. The example keeps the built-ins and adds the two rerankers.
After upgrading the base image, compare the list with
`/usr/local/searxng/searx/settings.yml` inside the image.
