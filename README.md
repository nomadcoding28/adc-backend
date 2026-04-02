# ACD Framework — Autonomous Cyber Defence

Production-grade reinforcement learning system for autonomous cyber defence,
built for MSRIT Major Project 2026.

## Research Novelties

| # | Novelty | Description |
|---|---------|-------------|
| 1 | **EWC + Continual Learning** | Prevents catastrophic forgetting when attack patterns shift (881× reduction) |
| 2 | **Adversarial Min-Max Training** | Robust to FGSM/PGD observation perturbations and reward poisoning |
| 3 | **CVaR-PPO** | Risk-sensitive RL optimising worst-5% outcomes (75% fewer catastrophic failures) |
| 4 | **Game-Theoretic Attacker Model** | Bayesian belief over attacker type + Nash equilibrium defender strategy |

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/your-org/acd-framework
cd acd_backend
cp .env.example .env          # Fill in secrets
make dev                       # Install dev dependencies
make setup-data                # Create data directories

# 2. Start infrastructure
make setup-neo4j               # Start Neo4j
# Or with Docker Compose:
docker-compose up -d redis neo4j

# 3. Start API server (hot reload)
make api
# → http://localhost:8000/docs

# 4. Train the agent
make train                     # 2M steps, CVaR-PPO + EWC

# 5. Run evaluation
make eval                      # 50 episodes + α-sensitivity table
make benchmark                 # Full Table 1 comparison
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
├── api/            FastAPI: 14 routers, schemas, WebSocket, middleware
├── auth/           JWT auth, bcrypt passwords, role guards
├── db/             SQLAlchemy ORM, Alembic migrations
├── cache/          Redis client, cache keys, @cached decorator
├── tasks/          Celery: training, evaluation, KG rebuild, reports
├── monitoring/     Prometheus metrics, Sentry, structured logging
└── utils/          Logger, config loader, device, seed, timer, metrics
```

## API Reference

| Endpoint | Description |
|----------|-------------|
| `GET  /docs` | Interactive Swagger UI |
| `POST /training/start` | Start CVaR-PPO training |
| `GET  /training/status` | Live training metrics |
| `GET  /network/topology` | Live CybORG network state |
| `GET  /cvar/metrics` | CVaR risk metrics |
| `GET  /cvar/alpha` | α-sensitivity table (Paper Table 2) |
| `GET  /drift/current` | Current Wasserstein distance |
| `GET  /kg/graph` | Knowledge graph for D3 viewer |
| `POST /explain/action` | LLM explanation for defender action |
| `GET  /game/belief` | Bayesian attacker type belief |
| `GET  /game/nash` | Nash equilibrium recommendations |
| `GET  /evaluation/benchmark` | Benchmark table (Paper Table 1) |
| `WS   /ws/training` | Live training metrics stream |
| `WS   /ws/alerts` | Live alert stream |

## Paper Results

### Table 1 — Benchmark Comparison (50 episodes)

| Agent | Mean Reward | CVaR α=0.05 | Success | Catastrophic |
|-------|-------------|-------------|---------|--------------|
| **CVaR-PPO + EWC (Ours)** | **+8.74** | **-2.14** | **87.3%** | **2.1%** |
| Standard PPO | +9.12 | -6.97 | 85.1% | 8.4% |
| PPO + CVaR (no EWC) | +8.21 | -2.89 | 84.7% | 3.8% |
| PPO + EWC (no CVaR) | +8.90 | -5.20 | 84.2% | 6.2% |
| Random Agent | -14.20 | -48.30 | 18.2% | 44.1% |

### EWC Forgetting Reduction

| Scenario | Without EWC | With EWC | Reduction |
|----------|-------------|----------|-----------|
| General | 4644.90 | 5.27 | **881×** |
| Ransomware→APT | 3980.06 | 9.95 | **400×** |

## Environment Variables

See `.env.example` for all required variables.
Key variables: `OPENAI_API_KEY`, `NEO4J_PASSWORD`, `ACD_JWT_SECRET`.

## License

MIT License — See LICENSE file.