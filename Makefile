.PHONY: install-dev test live lint build vendor

install-dev:
	python -m pip install -e "packages/bm25_reranker[dev]" -e packages/sem_reranker

test:
	python -m pytest packages -q

live:
	python -m pytest packages -q -m live

lint:
	ruff check packages
	complexipy
	ty check packages/bm25_reranker/src packages/sem_reranker/src

build:
	python -m build packages/bm25_reranker
	python -m build packages/sem_reranker

vendor:
	cd packages/bm25_reranker && zerodep update --all
	cd packages/sem_reranker && zerodep update --all
