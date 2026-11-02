FROM searxng/searxng:latest

# Point PIP_INDEX_URL at https://test.pypi.org/simple/ to test a TestPyPI
# build. BM25_VERSION and SEM_VERSION take a PEP 440 specifier, for example
# "==0.4.0.dev1". Leave them empty to install the latest release.
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG BM25_VERSION=
ARG SEM_VERSION=

RUN /usr/local/searxng/.venv/bin/python3 -m ensurepip && \
    /usr/local/searxng/.venv/bin/python3 -m pip install --no-cache-dir \
      --index-url "${PIP_INDEX_URL}" \
      "searxng-bm25-reranker${BM25_VERSION}" \
      "searxng-sem-reranker${SEM_VERSION}" && \
    /usr/local/searxng/.venv/bin/python3 -m pip uninstall -y pip
