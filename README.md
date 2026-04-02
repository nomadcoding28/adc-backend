# ACD Framework — Autonomous Cyber Defence
Reinforcement learning system for autonomous cyber defence.

## Research Novelties

| # | Novelty | Description |
|---|---------|-------------|
| 1 | **EWC + Continual Learning** | Prevents catastrophic forgetting when attack patterns shift (881× reduction) |
| 2 | **Adversarial Min-Max Training** | Robust to FGSM/PGD observation perturbations and reward poisoning |
| 3 | **CVaR-PPO** | Risk-sensitive RL optimising worst-5% outcomes (75% fewer catastrophic failures) |
| 4 | **Game-Theoretic Attacker Model** | Bayesian belief over attacker type + Nash equilibrium defender strategy |

## Prerequisites

Before getting started, make sure you have the following installed:
- **Python 3.10+**
- **Docker & Docker Compose** (for Neo4j and Redis)
- **Git**

**Note on CybORG:** This project depends on the legacy CybORG v2.1 (CAGE Challenge 2 architecture) which will be automatically installed via `requirements.txt` / `pyproject.toml`.

## Getting Started

Follow these steps to set up the backend locally:

### 1. Clone and Virtual Environment
```bash
git clone https://github.com/your-org/acd-framework
cd acd_backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies
```bash
# Production dependencies
make install
# Or for local development, pre-commit hooks, and testing
make dev
```

### 3. Application Setup
```bash
# Create required data directories
make setup-data

# Copy environment variables and fill with your secrets
cp .env.example .env
```
*Make sure to fill in your `OPENAI_API_KEY`, `NEO4J_PASSWORD`, and `ACD_JWT_SECRET` in the `.env` file.*

## Running the Application

### 1. Start Infrastructure (Redis & Neo4j)
```bash
# Start via Docker Compose
make docker-up
# Or just Neo4j standalone
make setup-neo4j
```

### 2. Start the API Server
```bash
# Starts the FastAPI server with hot-reloading (auth disabled for dev)
make api
# The API will be available at: http://localhost:8000
# Interactive Docs (Swagger): http://localhost:8000/docs
```

### 3. Start Background Workers
In a new terminal process (with the virtual environment activated), start the Celery worker to manage training and background tasks:
```bash
make worker
```
You can also monitor Celery queues via Flower (available on port 5555):
```bash
make flower
```

## Running the RL Training & Evaluations

```bash
# Train the CVaR-PPO + EWC agent (2M steps)
make train

# Train an adversarial model
make train-adv

# Run evaluation suite (50 episodes + risk metrics)
make eval

# Run full baseline benchmarking
make benchmark
```

## Testing Suite

The project includes a robust test suite covering mathematical verification, unit testing, and full CybORG environment integration tests.

```bash
# Run all tests with coverage
make test

# Run specific suites
make test-unit
make test-integration
make test-math
```

## Knowledge Graph Management

Manages pulling from MITRE ATT&CK and NVD to a Neo4j database and building the RAG FAISS index.

```bash
# Initial full KG rebuild (downloads NVD + ATT&CK + builds Neo4j)
make kg-build

# Update with newer CVEs (last 7 days)
make kg-update

# Rebuild RAG Vector Embeddings
make kg-embeddings
```

## Architecture

```
acd_backend/
├── agents/         RL agents: CVaR-PPO, EWC, adversarial trainer, registry
├── envs/           CybORG wrappers: obs space, action space, reward shaper
├── knowledge/      KG pipeline: Neo4j, NVD fetcher, MITRE parser, BERT mapper
├── explainability/ LLM explanations: RAG, ReAct loop, report generator
├── game/           Stochastic game: attacker model, Bayesian belief, Nash solver
├── drift/          Drift detection: Wasserstein, KS, MMD, window manager
├── api/            FastAPI: routers, schemas, WebSocket, middleware
├── auth/           JWT auth, bcrypt passwords, role guards
├── db/             SQLAlchemy ORM, Alembic migrations
├── cache/          Redis client, cache keys, @cached decorator
├── tasks/          Celery: training, evaluation, KG rebuild, reports
├── monitoring/     Prometheus metrics, Sentry, structured logging
└── utils/          Logger, config loader, device, seed, timer, metrics
```

## API Endpoint Reference

| Endpoint | Description |
|----------|-------------|
| `GET  /docs` | Interactive Swagger UI |
| `POST /training/start` | Start CVaR-PPO training |
| `GET  /training/status` | Live training metrics |
| `GET  /network/topology` | Live CybORG network state |
| `GET  /cvar/metrics` | CVaR risk metrics |
| `GET  /drift/current` | Current Wasserstein distance |
| `GET  /kg/graph` | Knowledge graph for D3 viewer |
| `POST /explain/action` | LLM explanation for defender action |
| `GET  /game/belief` | Bayesian attacker type belief |
| `GET  /evaluation/benchmark` | Benchmark table (Paper Table 1) |
| `WS   /ws/training` | Live training metrics stream |
| `WS   /ws/alerts` | Live alert stream |

## License

MIT License — See `LICENSE` file for more details.