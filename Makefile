.PHONY: dev worker test test-cov lint format \
        docker-build docker-up docker-down docker-logs docker-shell \
        docker-up-infra docker-down-infra \
        init-db download-models install clean

# ============================================================
# Development
# ============================================================

dev:
	uvicorn app.main:app --reload --port 8100

worker:
	celery -A app.tasks.celery_app worker --pool=prefork --concurrency=2 --loglevel=info

# ============================================================
# Testing
# ============================================================

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=app --cov-report=term-missing --cov-report=html

# ============================================================
# Code Quality
# ============================================================

lint:
	ruff check app/ tests/

format:
	ruff format app/ tests/
	ruff check --fix app/ tests/

# ============================================================
# Docker — Infrastructure
# ============================================================

docker-up-infra:
	docker compose -f docker/docker-compose.infra.yml up -d

docker-down-infra:
	docker compose -f docker/docker-compose.infra.yml down

docker-logs-infra:
	docker compose -f docker/docker-compose.infra.yml logs -f

# ============================================================
# Docker — Application
# ============================================================

docker-build:
	docker build -f docker/Dockerfile -t uni-ai-python:latest .

docker-up:
	docker compose -f docker/docker-compose.infra.yml \
	               -f docker/docker-compose.yml up -d

docker-down:
	docker compose -f docker/docker-compose.infra.yml \
	               -f docker/docker-compose.yml down

docker-restart:
	docker compose -f docker/docker-compose.yml restart uni-ai-python

docker-logs:
	docker logs -f uni-ai-python

docker-logs-worker:
	docker logs -f uni-ai-celery

docker-shell:
	docker exec -it uni-ai-python /bin/bash

# Scale Celery workers (usage: make docker-scale-worker N=4)
docker-scale-worker:
	docker compose -f docker/docker-compose.infra.yml \
	               -f docker/docker-compose.yml up -d --scale uni-ai-celery=$(N)

# ============================================================
# Database / Index Initialization
# ============================================================

init-db:
	python scripts/init_milvus.py
	python scripts/init_es.py

# ============================================================
# Models
# ============================================================

download-models:
	python scripts/download_models.py

# ============================================================
# Installation
# ============================================================

install:
	pip install -e ".[dev]"

# ============================================================
# Cleanup
# ============================================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
