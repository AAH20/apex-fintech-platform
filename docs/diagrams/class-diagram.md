# Class Diagram

Core type relationships across the Apex Fintech Platform. Only the most critical domain types are shown.

## Core Quant Engines

```mermaid
classDiagram
    %% Portfolio Optimization
    class CardinalityConstrainedOptimizer {
        +mu: ndarray
        +Sigma: ndarray
        +k: int
        +risk_aversion: float
        +sectors: Optional[ndarray]
        +sector_limits: Optional[dict]
        +_weights: Optional[ndarray]
        +_objective: Optional[float]
        +__init__(mu, Sigma, k, risk_aversion, sectors, sector_limits)
        +optimize() None
        +get_weights() ndarray
        +get_objective_value() float
        -_solve_eq(idx) ndarray
        -_solve_nonneg(idx) ndarray
        -_select_top_k(w) ndarray
        -_apply_sector_limits(w) ndarray
    }

    class AssetAllocationEngine {
        +risk_free_rate: float
        +__init__(risk_free_rate)
        +strategic_allocation(mu, Sigma, risk_aversion) ndarray
        +tactical_allocation(strategic, tilts, mu, Sigma, max_deviation, risk_aversion) ndarray
        +factor_allocation(B, f) ndarray
        +black_litterman(Sigma, w_mkt, views, view_assets, tau, omega) ndarray
        +risk_parity(Sigma) ndarray
        -_project_simplex(v, z) ndarray
    }

    %% Risk Management
    class RiskAggregationEngine {
        +confidence_level: float
        +__init__(confidence_level)
        +compute_var_bounds(samples) Tuple[float, float]
        +compute_cvar_bounds(samples) Tuple[float, float]
        +rearrangement_algorithm(samples) Tuple[float, float]
    }

    class RiskManagementEngine {
        +confidence_level: float
        +time_horizon: int
        +__init__(confidence_level, time_horizon)
        +historical_var(returns) float
        +parametric_var(returns) float
        +monte_carlo_var(returns, n_sims) float
        +historical_cvar(returns) float
        +parametric_cvar(returns) float
        +stress_test(portfolio, shock, correlation) dict
        +scenario_analysis(portfolio, scenarios) list
        +capital_adequacy_ratio(capital, rwa) float
    }

    %% Execution & Market Making
    class OptimalExecutionEngine {
        +__init__()
        +compute_trajectory(total_shares, time_horizon, volatility, risk_aversion, permanent_impact, temporary_impact) ndarray
    }

    class MarketMakingEngine {
        +gamma: float
        +sigma: float
        +k: float
        +A: float
        +dt: float
        +max_inventory: int
        +adverse_selection: float
        +__init__(gamma, sigma, k, A, dt, max_inventory, adverse_selection)
        +compute_quotes(mid_price, inventory, time_remaining) Quote
        +compute_quotes_batch(mid_prices, inventories, times_remaining) list[Quote]
        +expected_profit(quote, inventory) float
    }

    class Quote {
        +bid: float
        +ask: float
        +mid: float
        +spread: float
        +reservation_price: float
    }

    %% Derivatives & Monte Carlo
    class DerivativesPricingEngine {
        +__init__()
        +price_european_call(S, K, T, r, sigma) float
        +price_european_put(S, K, T, r, sigma) float
        +compute_greeks(S, K, T, r, sigma) dict
        +implied_volatility(price, S, K, T, r, option_type) float
    }

    class MonteCarloEngine {
        +dt: float
        +seed: Optional[int]
        +_rng: Generator
        +__init__(dt, seed)
        +_reset_rng() None
        +simulate_gbm(s0, mu, sigma, T, n_paths) ndarray
        +price_european_option(s0, k, r, sigma, T, n_paths, option_type) SimulationResult
        +price_american_option(s0, k, r, sigma, T, n_paths, option_type) SimulationResult
    }

    class SimulationResult {
        +price: float
        +std_error: float
        +confidence_interval: Tuple[float, float]
        +n_paths: int
    }

    %% Network Analysis
    class NetworkAnalysisEngine {
        +graph: nx.Graph
        +__init__()
        +from_edge_list(edges, directed) NetworkAnalysisEngine
        +from_adjacency_matrix(matrix, node_names) NetworkAnalysisEngine
        +num_nodes: int
        +num_edges: int
        +degree_centrality(weighted) dict[str, float]
        +betweenness_centrality() dict[str, float]
        +eigenvector_centrality() dict[str, float]
        +pagerank(alpha) dict[str, float]
        +community_detection_louvain() dict[str, int]
        +community_detection_greedy() dict[str, int]
        +modularity(communities) float
        +debtrank(shocked_nodes, iterations, equity) dict[str, float]
        +debtrank_centrality() dict[str, float]
        +systemic_risk_score() float
        +density() float
        +connected_components() list[set[str]]
        +clustering_coefficient() float
        +average_shortest_path_length() float
        +summary() dict
    }

    %% Game Theory
    class GameTheoryEngine {
        +__init__()
        +find_pure_nash(payoff) list[Tuple[int, int]]
        +find_mixed_nash(payoff) list[ndarray]
        +correlated_equilibrium(payoff) ndarray
        +shapley_value(characteristic_function) dict
        +core(characteristic_function) list[ndarray]
    }

    %% Relationships
    CardinalityConstrainedOptimizer --> AssetAllocationEngine : uses
    RiskAggregationEngine --> RiskManagementEngine : extends
    OptimalExecutionEngine --> MarketMakingEngine : related
    DerivativesPricingEngine --> MonteCarloEngine : uses
    MonteCarloEngine --> SimulationResult : returns
    NetworkAnalysisEngine --> GameTheoryEngine : related
```

## Advisory & Wealth Engines

```mermaid
classDiagram
    class RoboAdvisoryEngine {
        +__init__()
        +generate_plan(age, income, net_worth, risk_tolerance, goal, horizon, target_amount) dict
        +risk_profile(questionnaire_responses) dict
        +goal_based_allocation(goals, current_portfolio) dict
        +tax_loss_harvesting(portfolio, tax_rate) dict
        +rebalance(portfolio, target_weights, transaction_cost) dict
    }

    class ESGAnalyticsEngine {
        +ENV_WEIGHT: float = 0.4
        +SOC_WEIGHT: float = 0.3
        +GOV_WEIGHT: float = 0.3
        +DEFAULT_SCORE: float = 50.0
        +__init__()
        +score_company(data) ESGScore
        +portfolio_esg_score(holdings, companies) ESGScore
        +carbon_intensity(holdings, companies) float
        +greenwashing_detection(companies) list[dict]
    }

    class ESGDataInput {
        +company_id: str
        +company_name: str
        +environmental_score: Optional[float]
        +social_score: Optional[float]
        +governance_score: Optional[float]
        +carbon_emissions_tons: Optional[float]
        +revenue_millions: Optional[float]
    }

    class ESGScore {
        +company_id: str
        +company_name: str
        +environmental: float
        +social: float
        +governance: float
        +overall: float
        +carbon_intensity: Optional[float]
    }

    class BehavioralFinanceEngine {
        +__init__()
        +detect_biases(transactions, survey) dict
        +prospect_theory_value(x, alpha, beta, lambda_) float
        +loss_aversion_coefficient(choices) float
        +nudge_recommendation(biases, context) list[str]
    }

    class WealthManagementEngine {
        +__init__()
        +comprehensive_plan(profile, goals, assets, liabilities) dict
        +tax_optimization(portfolio, jurisdiction) dict
        +estate_planning(assets, beneficiaries, jurisdiction) dict
        +retirement_projection(current_age, retirement_age, savings, returns) dict
    }

    class PEAnalyticsEngine {
        +__init__()
        +waterfall_distribution(contributions, distributions, pref_return, catchup, carry) dict
        +compute_irr(cashflows) float
        +compute_moic(contributions, distributions) float
        +compute_dpi(distributions, contributions) float
        +compute_tvpi(contributions, distributions, nav) float
        +pme_analysis(fund_cashflows, benchmark_returns) dict
    }

    %% Relationships
    RoboAdvisoryEngine --> ESGAnalyticsEngine : uses
    RoboAdvisoryEngine --> BehavioralFinanceEngine : uses
    WealthManagementEngine --> RoboAdvisoryEngine : extends
    WealthManagementEngine --> PEAnalyticsEngine : uses
    ESGAnalyticsEngine --> ESGDataInput : consumes
    ESGAnalyticsEngine --> ESGScore : produces
```

## Payments & Credit Engines

```mermaid
classDiagram
    class PaymentRouter {
        +graph: nx.Graph
        +__init__(graph)
        +find_optimal_path(source, target, max_fee, min_liquidity) list
        +get_path_cost(path) float
        +get_all_paths(source, target, max_fee, min_liquidity) list[list]
    }

    class CreditUnderwritingEngine {
        +model: RandomForestClassifier
        +imputer: SimpleImputer
        +feature_names: list[str]
        +__init__()
        +train(X, y) None
        +predict(X) ndarray
        +predict_proba(X) ndarray
        +explain(X) list[dict]
        +get_feature_importance() dict[str, float]
    }

    class FraudDetector {
        +gnn_model: nn.Module
        +concept_drift_detector: DriftDetector
        +__init__()
        +train_graph(edges, features, labels) None
        +score_transactions(features, edges) ndarray
        +detect_rings(edges, scores, threshold) list[set]
        +update_drift(new_data) None
    }

    class BlockchainAnalyticsEngine {
        +__init__()
        +compute_block_metrics(blocks) dict
        +detect_whales(transactions, threshold) list[dict]
        +mev_detection(transactions) list[dict]
        +trace_funds(address, depth) dict
    }

    class RWATokenizer {
        +w3: Web3
        +owner_address: str
        +token_name: str
        +token_symbol: str
        +initial_supply: int
        +asset_type: str
        +asset_value: int
        +jurisdiction: str
        +__init__(w3, owner_address, token_name, token_symbol, initial_supply, asset_type, asset_value, jurisdiction)
        +deploy() dict
        +mint(to, amount) dict
        +burn(from, amount) dict
        +transfer(from, to, amount) dict
        +distribute_dividends(amount_per_token) dict
        +freeze(address) dict
        +unfreeze(address) dict
        +add_compliance_rule(rule) dict
    }

    %% Relationships
    PaymentRouter --> CreditUnderwritingEngine : related
    CreditUnderwritingEngine --> FraudDetector : related
    BlockchainAnalyticsEngine --> RWATokenizer : uses
    RWATokenizer --> PaymentRouter : related
```

## Governance & Compliance Engines

```mermaid
classDiagram
    class RegTechEngine {
        +rules: list[ComplianceRule]
        +__init__()
        +add_rule(rule) None
        +check_compliance(entity, jurisdiction) ComplianceResult
        +generate_report(entity, period) dict
        +monitor_changes(feed) None
    }

    class ComplianceRule {
        +id: str
        +jurisdiction: str
        +regulation: str
        +condition: Callable
        +severity: str
    }

    class ComplianceResult {
        +compliant: bool
        +violations: list[Violation]
        +warnings: list[Warning]
        +evidence: list[Evidence]
    }

    class EthicsComplianceEngine {
        +__init__()
        +check_suitability(client, product) SuitabilityResult
        +fiduciary_duty_check(advisor, client, recommendation) bool
        +conflict_of_interest_detection(advisor, client, product) list[Conflict]
        +best_execution_check(orders, venues) dict
    }

    class SuitabilityResult {
        +suitable: bool
        +risk_match: bool
        +knowledge_match: bool
        +experience_match: bool
        +rationale: str
    }

    class SecurityEngine {
        +__init__()
        +encrypt(data, key) bytes
        +decrypt(ciphertext, key) bytes
        +rotate_keys() None
        +audit_access(user, resource, action) bool
    }

    %% Relationships
    RegTechEngine --> ComplianceRule : manages
    RegTechEngine --> ComplianceResult : produces
    EthicsComplianceEngine --> SuitabilityResult : produces
    SecurityEngine --> RegTechEngine : secures
```

## Infrastructure Engines

```mermaid
classDiagram
    class MLPPipelineEngine {
        +feature_engineer: FeatureEngineer
        +model_selector: ModelSelector
        +hyperparameter_tuner: HyperparameterTuner
        +__init__()
        +fit(X, y) None
        +predict(X) ndarray
        +evaluate(X, y) dict
        +export_model(path) None
    }

    class FeatureEngineer {
        +steps: list[Transformer]
        +__init__()
        +add_step(transformer) None
        +fit_transform(X, y) ndarray
        +transform(X) ndarray
    }

    class CachingEngine {
        +redis_client: Optional[Redis]
        +memory_cache: LRUCache
        +__init__(redis_url, max_memory_size, default_ttl)
        +get(key) Any
        +set(key, value, ttl) None
        +delete(key) None
        +clear() None
        +stats() dict
    }

    class LRUCache {
        +max_size: int
        +default_ttl: int
        +_cache: OrderedDict
        +_timestamps: dict
        +__init__(max_size, default_ttl)
        +get(key) Any
        +set(key, value, ttl) None
        +_evict_expired() None
        +_evict_lru() None
    }

    class ObservabilityEngine {
        +metrics: PrometheusMetrics
        +tracer: Tracer
        +logger: Logger
        +__init__(service_name)
        +record_metric(name, value, labels) None
        +start_span(name, attributes) Span
        +log(level, message, fields) None
    }

    class ConfigurationEngine {
        +settings: BaseSettings
        +__init__()
        +get(key) Any
        +set(key, value) None
        +validate() bool
        +reload() None
    }

    %% Relationships
    MLPPipelineEngine --> FeatureEngineer : uses
    MLPPipelineEngine --> CachingEngine : caches
    ObservabilityEngine --> MLPPipelineEngine : monitors
    ConfigurationEngine --> CachingEngine : configures
    ConfigurationEngine --> ObservabilityEngine : configures
```

## Notes

- **Type annotations**: All classes use Python 3.10+ type hints (`list`, `dict`, `Optional`, `Tuple`, `ndarray`)
- **Pydantic models**: `ESGDataInput`, `ESGScore`, `SuitabilityResult`, `ComplianceResult` use Pydantic v2 for validation
- **Dataclasses**: `Quote`, `SimulationResult`, `StressScenario`, `RiskReport` use `@dataclass`
- **Protocols**: Engine interfaces follow implicit protocols (duck typing) for testability
- **Dependency injection**: Engines accept dependencies via `__init__` for easy mocking in tests
- **Async support**: I/O-bound engines (blockchain, data engineering) use `async`/`await`; CPU-bound engines are synchronous

---

*Generated from source code. See individual module docs in `modules/` for detailed API reference.*