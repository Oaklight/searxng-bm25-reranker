FROM searxng/searxng:latest
RUN /usr/local/searxng/.venv/bin/python3 -m ensurepip && \
    /usr/local/searxng/.venv/bin/python3 -m pip install --no-cache-dir searxng-bm25-reranker && \
    /usr/local/searxng/.venv/bin/python3 -m pip uninstall -y pip
