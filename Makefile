# ============================================================
# ACD Framework — Makefile
# ============================================================

.PHONY: help install dev test lint format train eval api docker clean

# ── Default ──────────────────────────────────────────────────
help:
	@echo "ACD Framework — Available targets:"
	@echo ""
	@echo "  Setup:"
	@echo "    install       Install production dependencies"
	@echo "    dev           Install dev dependencies + pre-commit hooks"
	@echo "    setup-data    Create data directories"
	@echo "    setup-neo4j   Start Neo4j via Docker"
	@echo ""
	@echo "  Development:"
	@echo "    api           Start FastAPI dev server (hot reload)"
	@echo "    worker        Start Celery worker"
	@echo "    flower        Start Celery monitoring UI"
	@echo ""
	@echo "  Training:"
	@echo "    train         Start full CVaR-PPO + EWC training run"
	@echo "    train-adv     Start adversarial training run"
	@echo "    eval          Run evaluation suite (50 episodes)"
	@echo "    benchmark     Run full benchmark table (all variants)"
	@echo ""
	@echo "  Knowledge Graph:"
	@echo "    kg-build      Full KG rebuild (NVD + ATT&CK + Neo4j)"
	@echo "    kg-update     Incremental CVE update (last 7 days)"
	@echo "    kg-embeddings Build RAG FAISS index"
	@echo ""
	@echo "  Quality:"
	@echo "    test          Run test suite with coverage"
	@echo "    lint          Run ruff linter"
	@echo "    format        Auto-format with ruff + black"
	@echo "    typecheck     Run mypy type checker"
	@echo ""
	@echo "  Docker:"
	@echo "    docker-build  Build production Docker image"
	@echo "    docker-up     Start full stack (API + Neo4j + Redis)"
	@echo "    docker-down   Stop all containers"
	@echo ""
	@echo "    clean         Remove __pycache__ and .pyc files"

# ── Setup ────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

dev:
	pip install -r requirements-dev.txt
	pre-commit install
	@echo "Dev environment ready."

setup-data:
	mkdir -p data/checkpoints data/logs data/experiences \
	         data/kg_cache data/embeddings data/incidents
	@echo "Data directories created."

setup-neo4j:
	docker run -d \
	  --name acd-neo4j \
	  -p 7474:7474 -p 7687:7687 \
	  -e NEO4J_AUTH=neo4j/password \
	  -v $(PWD)/data/neo4j:/data \
	  neo4j:5.24
	@echo "Neo4j started at bolt://localhost:7687 (neo4j/password)"

# ── Development ──────────────────────────────────────────────
api:
	ACD_AUTH_DISABLED=true \
	DEBUG=true \
	uvicorn api.app:app \
	  --host 0.0.0.0 --port 8000 \
	  --reload --log-level debug

worker:
	celery -A tasks.celery_app worker \
	  --loglevel=info \
	  -Q default,training,kg,reports \
	  -c 4

flower:
	celery -A tasks.celery_app flower --port=5555

# ── Training ─────────────────────────────────────────────────
train:
	python train_full.py \
	  --config config.yaml \
	  --timesteps 2000000 \
	  --eval

train-adv:
	python train_full.py \
	  --config config.yaml \
	  --timesteps 2000000 \
	  --adversarial

eval:
	python evaluate.py \
	  --config config.yaml \
	  --checkpoint data/checkpoints/cvar_ppo_final.zip \
	  --n-episodes 50 \
	  --alpha-sweep

benchmark:
	python evaluate.py \
	  --config config.yaml \
	  --benchmark \
	  --n-episodes 50 \
	  --export data/paper_results_benchmark.csv

# ── Knowledge Graph ──────────────────────────────────────────
kg-build:
	python scripts/download_nvd.py
	python scripts/download_attck.py
	python scripts/build_embeddings.py

kg-update:
	python -c "from tasks.kg_tasks import update_cves; update_cves(days=7)"

kg-embeddings:
	python scripts/build_embeddings.py

# ── Quality ──────────────────────────────────────────────────
test:
	pytest tests/ -v \
	  --cov=. \
	  --cov-report=term-missing \
	  --cov-report=html:htmlcov \
	  -x

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-math:
	python tests/verify_cvar_math.py
	python tests/verify_ewc_math.py

lint:
	ruff check .

format:
	ruff check --fix .
	black .

typecheck:
	mypy . --ignore-missing-imports

# ── Docker ───────────────────────────────────────────────────
docker-build:
	docker build -t acd-backend:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f api

# ── Cleanup ──────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned."