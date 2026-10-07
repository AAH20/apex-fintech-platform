# Getting Started

Quick start guide for the Apex Fintech Platform.

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10+ | 3.11+ recommended |
| Redis | 7.0+ | Optional — in-memory fallback included |
| PostgreSQL | 15+ | Optional — SQLite for local dev |
| Docker | 24+ | For containerized deployment |
| Node.js | 18+ | Only for frontend development |

**Core dependencies (always installed):**
- numpy, networkx, pydantic, fastapi, uvicorn
- asyncpg, redis, aiokafka (optional)

**Full dependencies (optional extras):**
- scipy, scikit-learn, xgboost, torch, torch-geometric
- pandas, pyarrow, prophet, web3.py

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/AAH20/apex-fintech-platform.git
cd apex-fintech-platform
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

### 3. Install Core Package (Minimal)

```bash
# Core only — numpy, networkx, fastapi, pydantic
# No scipy, torch, sklearn, pandas
pip install -e ".[core]"
```

### 4. Install Full Stack (All Engines)

```bash
# Full stack — includes scipy, sklearn, torch, xgboost, pandas, web3
pip install -e ".[full]"

# Or with development tools
pip install -e ".[full,dev]"
```

### 5. Verify Installation

```bash
# Run core tests (no optional deps needed)
pytest tests/test_portfolio.py tests/test_risk.py tests/test_routing.py -v

# Run all tests (requires full deps)
pytest tests/ -v --tb=short
```

## Configuration

### Environment Variables

```bash
# API
export APEX_API_HOST=0.0.0.0
export APEX_API_PORT=8000
export APEX_API_WORKERS=4
export APEX_JWT_SECRET=your-secret-key  # Required in production

# Redis (optional — falls back to in-memory LRU)
export APEX_REDIS_URL=redis://localhost:6379/0

# PostgreSQL (optional — SQLite used if not set)
export APEX_POSTGRES_DSN=postgresql://user:pass@localhost/apex

# Observability
export APEX_METRICS_PORT=9090
export APEX_JAEGER_ENDPOINT=http://localhost:4318
export APEX_LOG_LEVEL=INFO

# Governance
export APEX_ISO_42001_ENABLED=true
export APEX_AUDIT_TRAIL_ENABLED=true
```

### Configuration File (config/platform.yaml)

```yaml
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  rate_limit: "1000/minute"
  auth:
    type: "jwt"
    algorithm: "RS256"
    public_key_path: "/etc/apex/jwt-public.pem"

redis:
  url: "redis://localhost:6379/0"
  max_connections: 50
  fallback: "memory"  # "memory" | "none"

postgres:
  dsn: "postgresql://user:pass@localhost/apex"
  pool_size: 20
  max_overflow: 10

governance:
  iso_42001: true
  audit_trail: true
  model_cards: true
  evidence_chain: true
  evidence_retention_days: 2555  # 7 years

observability:
  metrics_port: 9090
  traces_endpoint: "http://jaeger:4318"
  log_level: "INFO"
  log_format: "json"

engines:
  portfolio:
    default_k: 50
    default_risk_aversion: 1.0
    max_assets: 2000
  risk:
    default_confidence: 0.95
    default_horizon: 1
  marketmaking:
    default_gamma: 0.1
    default_sigma: 0.02
    default_k: 1.5
    default_A: 0.1
  simulation:
    default_dt: 0.01
    default_paths: 10000
```

## First Run

### Start API Server

```bash
# Development (auto-reload)
uvicorn src.api.gateway:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn src.api.gateway:app --host 0.0.0.0 --port 8000 --workers 4
```

### Test Portfolio Optimization

```python
import numpy as np
from portfolio.optimizer import CardinalityConstrainedOptimizer

# Generate synthetic returns
np.random.seed(42)
n_assets = 500
n_days = 252
returns = np.random.normal(0.0005, 0.02, (n_assets, n_days))
cov = np.cov(returns) + np.eye(n_assets) * 0.001

# Optimize with cardinality constraint K=50
opt = CardinalityConstrainedOptimizer(
    mu=returns.mean(axis=1),
    Sigma=cov,
    k=50,
    risk_aversion=1.0
)
opt.optimize()
weights = opt.get_weights()

print(f"Non-zero positions: {(weights > 1e-6).sum()}")
print(f"Portfolio variance: {weights @ cov @ weights:.6f}")
print(f"Expected return: {weights @ returns.mean(axis=1):.6f}")
```

### Test via API

```bash
# Health check
curl http://localhost:8000/health

# Portfolio optimization
curl -X POST http://localhost:8000/api/v1/portfolio/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "mu": [0.001, 0.002, 0.0015],
    "Sigma": [[0.04, 0.01, 0.005], [0.01, 0.03, 0.008], [0.005, 0.008, 0.02]],
    "k": 2,
    "risk_aversion": 1.0
  }'

# Risk aggregation
curl -X POST http://localhost:8000/api/v1/risk/var-bounds \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [[-0.02, -0.01, 0.0, 0.01, 0.02], [-0.03, -0.015, 0.0, 0.015, 0.03]],
    "confidence_level": 0.95
  }'

# Payment routing
curl -X POST http://localhost:8000/api/v1/payments/route \
  -H "Content-Type: application/json" \
  -d '{
    "source": "0",
    "target": "999",
    "max_fee": 0.001,
    "min_liquidity": 1000
  }'
```

### OpenAPI Documentation

Visit http://localhost:8000/docs for interactive Swagger UI.

## Common Workflows

### Run Full Test Suite

```bash
# Core tests only (fast, no optional deps)
pytest tests/test_portfolio.py tests/test_risk.py tests/test_routing.py \
       tests/test_marketmaking.py tests/test_network.py \
       tests/test_gametheory.py tests/test_simulation.py -v

# All tests (requires full install)
pytest tests/ -v --tb=short -x

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Run Benchmarks

```bash
# Agent eval harness benchmarks
python benchmark.py

# Load testing (requires locust)
locust -f locustfile.py --host=http://localhost:8000
```

### Docker Deployment

```bash
# Build image
docker build -t apex-fintech-platform:latest .

# Run container
docker run -d \
  -p 8000:8000 \
  -e APEX_REDIS_URL=redis://host.docker.internal:6379/0 \
  -e APEX_POSTGRES_DSN=postgresql://user:pass@host.docker.internal/apex \
  apex-fintech-platform:latest

# Or use docker-compose
docker-compose up -d
```

### Kubernetes Deployment

```bash
# Apply manifests
kubectl apply -f k8s/

# Or use Helm
helm install apex ./helm/apex-fintech-platform \
  --set redis.url=redis://redis:6379/0 \
  --set postgres.dsn=postgresql://user:pass@postgres/apex
```

## Project Structure

```
apex-fintech-platform/
├── src/
│   ├── api/              # FastAPI gateway
│   ├── portfolio/        # Portfolio optimization
│   ├── risk/             # Risk management
│   ├── execution/        # Optimal execution
│   ├── marketmaking/     # Market making
│   ├── derivatives/      # Derivatives pricing
│   ├── simulation/       # Monte Carlo
│   ├── network/          # Network analysis
│   ├── gametheory/       # Game theory
│   ├── roboadvisory/     # Robo advisory
│   ├── esg/              # ESG analytics
│   ├── behavioral/       # Behavioral finance
│   ├── wealth/           # Wealth management
│   ├── pe/               # Private equity
│   ├── payments/         # Payment routing
│   ├── credit/           # Credit underwriting
│   ├── fraud/            # Fraud detection
│   ├── blockchain/       # Blockchain analytics
│   ├── tokenization/     # RWA tokenization
│   ├── regtech/          # RegTech compliance
│   ├── ethics/           # Ethics compliance
│   ├── ml/               # ML pipeline
│   ├── nlp/              # NLP analytics
│   ├── data/             # Data engineering
│   ├── caching/          # Caching engine
│   ├── observability/    # Observability
│   ├── config/           # Configuration
│   ├── docs/             # Auto-docs
│   ├── testing/          # Testing engine
│   ├── security/         # Security engine
│   ├── devops/           # DevOps engine
│   └── ...               # 15+ more engines
├── tests/                # 1433 tests
├── docs/                 # Documentation
│   ├── architecture.md
│   ├── diagrams/
│   └── workflows/
├── benchmark.py          # Agent eval harness
├── pyproject.toml        # Package config
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Development Workflow

### Adding a New Engine

1. **Create engine module** under `src/<category>/`
2. **Write tests FIRST** in `tests/test_<engine>.py` (TDD)
3. **Implement engine** with numpy-only critical path
4. **Add to API gateway** in `src/api/gateway.py`
5. **Run tests**: `pytest tests/test_<engine>.py -v`
6. **Benchmark**: Add to `benchmark.py`
7. **Document**: Update `docs/architecture.md` and `README.md`

### Code Standards

- **Type hints required**: All public functions
- **Docstrings**: NumPy style for all classes/methods
- **Line length**: 100 chars (ruff enforced)
- **Anti-loop constraints** (for agent-developed code):
  - Max 200 lines per file
  - Single file implementation
  - Max 3 file writes
  - TDD mandatory

### Git Workflow

```bash
# Feature branch
git checkout -b feat/new-engine

# Develop with TDD
# ... write tests, implement, test ...

# Commit with conventional commits
git commit -m "feat(engine): add new engine with TDD"

# Push and PR
git push origin feat/new-engine
# Open PR on GitHub
```

## Troubleshooting

### Import Errors (Missing Optional Dependencies)

```bash
# Error: ModuleNotFoundError: scipy
pip install scipy

# Error: ModuleNotFoundError: torch
pip install torch

# Error: ModuleNotFoundError: sklearn
pip install scikit-learn

# Or install all at once
pip install -e ".[full]"
```

### Redis Connection Failed

The platform automatically falls back to in-memory LRU cache. Check logs:
```
WARNING: Redis unavailable, using in-memory LRU fallback
```

### Database Migration Issues

```bash
# Run migrations (if using alembic)
alembic upgrade head

# Or reset local DB
rm -f apex.db  # SQLite
```

### Performance Issues

```bash
# Profile with py-spy
py-spy record -o profile.svg -- python benchmark.py

# Check engine latencies
python benchmark.py  # Shows avg/p99 per engine
```

## Next Steps

1. **Explore engines**: Read module docs in `docs/modules/`
2. **Understand architecture**: See `docs/architecture.md`
3. **Review sequences**: See `docs/diagrams/sequences.md`
4. **Run benchmarks**: `python benchmark.py`
5. **Deploy to staging**: Use `docker-compose.yml` or Kubernetes manifests
6. **Join community**: GitHub Discussions for questions and ideas

## Support

- **Issues**: [GitHub Issues](https://github.com/AAH20/apex-fintech-platform/issues)
- **Discussions**: [GitHub Discussions](https://github.com/AAH20/apex-fintech-platform/discussions)
- **Security**: Email aah@a2zsoc.com (GPG key on GitHub profile)
- **Commercial**: Contact for enterprise support, SLAs, custom engines

---

*Built with TDD discipline, governance-first architecture, and MENA market focus.*