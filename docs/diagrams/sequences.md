# Sequence Diagrams

Key workflows in the Apex Fintech Platform. Each diagram traces the actual call path through the codebase.

---

## Workflow 1: Portfolio Optimization Request

**Description**: Client requests cardinality-constrained portfolio optimization via the API gateway.

```mermaid
sequenceDiagram
    participant Client
    participant GW as API Gateway
    participant Auth as Auth Middleware
    participant PO as Portfolio Optimizer
    participant Cache as Caching Engine
    participant Obs as Observability

    Client->>GW: POST /api/v1/portfolio/optimize\n{mu, Sigma, k=50, risk_aversion=1.0}
    GW->>Auth: validate JWT + tenant scope
    Auth-->>GW: claims {tenant_id, roles, quotas}
    GW->>Obs: record request metric\ncounter.requests_total++
    GW->>Cache: check cache key\n"portfolio:{tenant}:{hash}"
    alt Cache Hit
        Cache-->>GW: cached weights
        GW-->>Client: 200 OK {weights, cached: true}
    else Cache Miss
        GW->>PO: optimize(mu, Sigma, k, risk_aversion)
        Note over PO: Greedy heuristic:\n1. Solve unconstrained QP\n2. Select top-K by weight\n3. Re-solve with active-set
        PO-->>GW: optimal weights (ndarray)
        GW->>Cache: set cache with TTL=300s
        GW->>Obs: record latency histogram\nportfolio.optimize.latency_ms
        GW-->>Client: 200 OK {weights, objective, cached: false}
    end
```

**Walkthrough:**
1. **Request validation** — [`src/api/gateway.py:optimize_portfolio`](src/api/gateway.py) validates input schema via Pydantic
2. **Auth** — [`src/security/engine.py:verify_token`](src/security/engine.py) checks JWT signature, expiry, tenant claims
3. **Cache check** — [`src/caching/engine.py:get`](src/caching/engine.py) tries Redis first, falls back to in-memory LRU
4. **Optimization** — [`src/portfolio/optimizer.py:optimize`](src/portfolio/optimizer.py) runs 3-step greedy heuristic
5. **Response** — Weights serialized to JSON, metadata includes cache status and objective value

---

## Workflow 2: Risk Aggregation with VaR Bounds

**Description**: Compute VaR/CVaR bounds for a portfolio using the rearrangement algorithm.

```mermaid
sequenceDiagram
    participant Client
    participant GW as API Gateway
    participant RA as Risk Aggregation
    participant RM as Risk Management
    participant Obs as Observability

    Client->>GW: POST /api/v1/risk/var-bounds\n{samples: [[...]], confidence=0.95}
    GW->>Obs: record request
    GW->>RA: compute_var_bounds(samples, confidence)
    Note over RA: Rearrangement Algorithm:\n1. Sort samples per risk\n2. Independent sum (lower bound)\n3. Countermonotonic (upper, n=2 only)\n4. For n>2: use independent sum\n   as conservative lower bound
    RA-->>GW: (var_lower, var_upper)
    GW->>Obs: record latency + result
    GW-->>Client: 200 OK {var_lower, var_upper, cvar_lower, cvar_upper}
```

**Walkthrough:**
1. **Input validation** — Samples shape checked: `(n_risks, n_scenarios)`
2. **Rearrangement** — [`src/risk/aggregation.py:compute_var_bounds`](src/risk/aggregation.py) implements Puccetti & Rüschendorf (2012)
3. **Conservative bounds** — For n>2 risks, countermonotonic coupling is invalid; independent sum provides valid lower bound
4. **CVaR bounds** — Same rearrangement applied to tail expectations

---

## Workflow 3: Payment Routing with Constraints

**Description**: Find optimal payment path in a network with fee and liquidity constraints.

```mermaid
sequenceDiagram
    participant Client
    participant GW as API Gateway
    participant PR as Payment Router
    participant NX as NetworkX Graph
    participant Obs as Observability

    Client->>GW: POST /api/v1/payments/route\n{source, target, max_fee=0.001, min_liquidity=1000}
    GW->>Obs: record request
    GW->>PR: find_optimal_path(source, target, max_fee, min_liquidity)
    Note over PR: Constrained Dijkstra:\n1. Filter edges: fee <= max_fee AND liquidity >= min_liquidity\n2. If negative fees: Bellman-Ford\n3. Else: Dijkstra (nx.shortest_path)\n4. Raise NetworkXNoPath if unreachable
    PR->>NX: shortest_path(filtered_graph, source, target, weight="fee")
    NX-->>PR: path [0, 42, 87, 999]
    PR-->>GW: optimal path
    GW->>Obs: record latency + path length
    GW-->>Client: 200 OK {path, total_fee, total_liquidity, hops: 3}
```

**Walkthrough:**
1. **Graph construction** — [`src/payments/routing.py:__init__`](src/payments/routing.py) builds NetworkX graph from edge list
2. **Constraint filtering** — Subgraph created with only viable edges
3. **Algorithm selection** — Dijkstra for non-negative fees, Bellman-Ford if negative fees detected
4. **Result** — Path, cumulative fee, minimum liquidity along path, hop count

---

## Workflow 4: Market Making Quote Generation

**Description**: Generate Avellaneda-Stoikov optimal bid/ask quotes for a given inventory and time horizon.

```mermaid
sequenceDiagram
    participant Client
    participant GW as API Gateway
    participant MM as Market Making Engine
    participant Obs as Observability

    Client->>GW: POST /api/v1/marketmaking/quotes\n{mid_price=100, inventory=-10, time_remaining=1.0}
    GW->>Obs: record request
    GW->>MM: compute_quotes(mid_price, inventory, time_remaining)
    Note over MM: Avellaneda-Stoikov:\nreservation = mid - q·γ·σ²·(T-t)\nspread = γ·σ²·(T-t) + (2/γ)·ln(1 + γ/(k·A))\nbid = reservation - spread/2\nask = reservation + spread/2
    MM-->>GW: Quote{bid, ask, mid, spread, reservation_price}
    GW->>Obs: record latency + spread metric
    GW-->>Client: 200 OK {bid: 99.42, ask: 100.58, spread: 1.16, reservation: 100.0}
```

**Walkthrough:**
1. **Parameter validation** — Mid price > 0, time_remaining >= 0, inventory clamped to max_inventory
2. **Reservation price** — Shifts mid by inventory risk: `q * γ * σ² * (T-t)`
3. **Optimal spread** — Balances adverse selection vs. order arrival: `γσ²τ + (2/γ)ln(1+γ/kA)`
4. **Adverse selection** — Optional spread widening proportional to `|inventory|`

---

## Workflow 5: Monte Carlo Option Pricing (European + American)

**Description**: Price European and American options using Monte Carlo with Longstaff-Schwartz for early exercise.

```mermaid
sequenceDiagram
    participant Client
    participant GW as API Gateway
    participant MC as Monte Carlo Engine
    participant LSM as Longstaff-Schwartz
    participant Obs as Observability

    Client->>GW: POST /api/v1/simulation/price\n{s0=100, k=100, r=0.05, sigma=0.2, T=1, n_paths=10000, type="american"}
    GW->>Obs: record request
    GW->>MC: price_american_option(s0, k, r, sigma, T, n_paths, "call")
    Note over MC: GBM Path Generation:\n1. n_paths × n_steps lognormal paths\n2. dt = T / n_steps (default 0.01)
    MC->>LSM: backward_induction(paths, payoff, r, dt)
    Note over LSM: LSM Algorithm:\n1. At each step t from T-1 to 0:\n   - Regress continuation value on basis functions\n   - Compare immediate exercise vs. continuation\n   - Optimal exercise boundary learned
    LSM-->>MC: option_price, std_error, CI
    MC-->>GW: SimulationResult{price, std_error, ci, n_paths}
    GW->>Obs: record latency + price distribution
    GW-->>Client: 200 OK {price: 10.50, std_error: 0.03, ci: [10.44, 10.56], n_paths: 10000}
```

**Walkthrough:**
1. **Path generation** — [`src/simulation/engine.py:simulate_gbm`](src/simulation/engine.py) uses vectorized numpy
2. **LSM regression** — Polynomial basis (1, S, S²) for continuation value estimation
3. **Early exercise** — Compares intrinsic value vs. discounted continuation at each step
4. **Statistics** — Price, standard error, 95% confidence interval from path distribution

---

## Workflow 6: RWA Token Deployment (ERC-3643)

**Description**: Deploy a compliant security token for a real-world asset.

```mermaid
sequenceDiagram
    participant Client
    participant GW as API Gateway
    participant RT as RWA Tokenizer
    participant W3 as Web3.py
    participant Chain as Blockchain
    participant Obs as Observability

    Client->>GW: POST /api/v1/tokenization/deploy\n{name, symbol, asset_type, asset_value, jurisdiction, supply}
    GW->>Obs: record request
    GW->>RT: deploy(asset_params)
    Note over RT: ERC-3643 Deployment:\n1. Validate jurisdiction (not sanctioned)\n2. Check asset_type in allowed list\n3. Deploy T-REX implementation\n4. Initialize compliance contract\n5. Set token metadata\n6. Mint initial supply to owner
    RT->>W3: deploy_contract(bytecode, abi, constructor_args)
    W3->>Chain: eth_sendRawTransaction
    Chain-->>W3: tx_hash
    W3-->>RT: contract_address
    RT->>RT: initialize_compliance(identity_registry, compliance)
    RT-->>GW: {address, tx_hash, token_info}
    GW->>Obs: record deployment event
    GW-->>Client: 201 Created {contract_address, tx_hash, explorer_url}
```

**Walkthrough:**
1. **Compliance checks** — [`src/tokenization/rwa.py:__init__`](src/tokenization/rwa.py) validates jurisdiction against sanctions list
2. **Contract deployment** — Uses Web3.py to deploy ERC-3643 (T-REX) implementation
3. **Identity registry** — Links to on-chain identity verification (KYC/AML providers)
4. **Compliance rules** — Transfer restrictions, holding limits, freeze functions configured

---

## Workflow 7: AI Credit Underwriting with Explainability

**Description**: Train credit scoring model and generate SHAP explanations for decisions.

```mermaid
sequenceDiagram
    participant Client
    participant GW as API Gateway
    participant CU as Credit Underwriting
    participant SKL as scikit-learn
    participant SHAP as SHAP Explainer
    participant Obs as Observability

    Client->>GW: POST /api/v1/credit/train\n{X: [[...]], y: [...], feature_names: [...]}
    GW->>Obs: record request
    GW->>CU: train(X, y)
    Note over CU: RandomForest Training:\n1. SimpleImputer for missing values\n2. RandomForestClassifier(n=200, depth=8)\n3. Cross-validation for robustness
    CU->>SKL: fit(X_imputed, y)
    SKL-->>CU: trained_model
    CU-->>GW: {model_id, cv_score, feature_importance}
    GW-->>Client: 200 OK {model_id, accuracy: 0.96}

    Client->>GW: POST /api/v1/credit/explain\n{model_id, X_new: [[...]]}
    GW->>CU: explain(X_new)
    CU->>SHAP: shap_values(model, X_new)
    Note over SHAP: TreeSHAP:\n1. Exact algorithm for trees\n2. Feature importance × normalized value\n3. Per-sample, per-feature contributions
    SHAP-->>CU: explanations[{feature, value, contribution}]
    CU-->>GW: explainability report
    GW-->>Client: 200 OK {predictions, explanations, feature_importance}
```

**Walkthrough:**
1. **Training** — [`src/credit/underwriting.py:train`](src/credit/underwriting.py) uses RF with imputation
2. **Explainability** — TreeSHAP provides exact feature contributions per prediction
3. **Output** — Prediction + per-feature SHAP values + global feature importance ranking

---

## Workflow 8: Compliance Check with Audit Trail

**Description**: Run regulatory compliance check and generate auditable evidence chain.

```mermaid
sequenceDiagram
    participant Client
    participant GW as API Gateway
    participant RC as RegTech Engine
    participant EC as Ethics Compliance
    participant Audit as Audit Trail
    participant Obs as Observability

    Client->>GW: POST /api/v1/compliance/check\n{entity, jurisdiction, product}
    GW->>Obs: record request
    GW->>RC: check_compliance(entity, jurisdiction)
    Note over RC: Rule Engine:\n1. Load rules for jurisdiction\n2. Evaluate conditions against entity\n3. Collect evidence per rule\n4. Aggregate violations/warnings
    RC->>EC: check_suitability(entity, product)
    EC-->>RC: SuitabilityResult
    RC->>Audit: record_evidence(entity, rules, results, timestamp)
    Note over Audit: ISO 42001 Evidence Chain:\n- Rule ID + version\n- Input data hash\n- Evaluation result\n- Timestamp + actor\n- Immutable append-only log
    Audit-->>RC: evidence_ref
    RC-->>GW: ComplianceResult{compliant, violations, warnings, evidence_ref}
    GW->>Obs: record compliance outcome
    GW-->>Client: 200 OK {compliant: false, violations: [...], evidence_ref: "audit:abc123"}
```

**Walkthrough:**
1. **Rule evaluation** — [`src/regtech/compliance.py:check_compliance`](src/regtech/compliance.py) loads jurisdiction-specific rules
2. **Ethics check** — [`src/ethics/compliance.py:check_suitability`](src/ethics/compliance.py) validates fiduciary duty
3. **Evidence chain** — Every check creates immutable audit record with input hashes
4. **Traceability** — Evidence reference allows full reconstruction of decision

---

## Workflow 9: Multi-Engine Advisory Pipeline

**Description**: End-to-end robo-advisory workflow combining risk profiling, ESG, behavioral, and portfolio optimization.

```mermaid
sequenceDiagram
    participant Client
    participant GW as API Gateway
    participant RO as Robo Advisory
    participant ES as ESG Analytics
    participant BF as Behavioral Finance
    participant PO as Portfolio Optimizer
    participant Cache as Caching Engine
    participant Obs as Observability

    Client->>GW: POST /api/v1/advisory/plan\n{age, income, risk_tolerance, esg_preferences, goals}
    GW->>Obs: record request
    GW->>RO: generate_plan(profile)
    RO->>RO: risk_profile(questionnaire)
    RO->>ES: portfolio_esg_score(holdings, universe)
    ES-->>RO: ESGScore{overall, carbon_intensity}
    RO->>BF: detect_biases(transactions, survey)
    BF-->>RO: biases{loss_aversion, overconfidence, ...}
    RO->>PO: optimize(mu, Sigma, k=20, risk_aversion)
    Note over PO: Cardinality constraint\naligned with advisor\nrecommendation count
    PO-->>RO: optimal weights
    RO->>Cache: cache plan with TTL=3600s
    RO-->>GW: comprehensive_plan{weights, esg, biases, tax_alpha, projections}
    GW->>Obs: record end-to-end latency
    GW-->>Client: 200 OK {plan, metadata}
```

**Walkthrough:**
1. **Risk profiling** — Questionnaire maps to risk aversion parameter
2. **ESG integration** — Universe filtered/scored by ESG preferences
3. **Behavioral adjustments** — Biases detected → portfolio nudges (e.g., reduce loss-aversion-driven concentration)
4. **Optimization** — Portfolio optimizer constrained by advisor's recommended position count
5. **Tax alpha** — Tax-loss harvesting opportunities identified
6. **Caching** — Full plan cached for 1 hour for repeat requests

---

*See also: [Architecture](architecture.md), [Class Diagram](diagrams/class-diagram.md), [Workflows](workflows/)*