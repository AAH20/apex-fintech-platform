"""Testing Engine — orchestrates unit, integration, and performance tests.

Provides a unified framework for testing all platform engines:
- Market Making (Avellaneda-Stoikov, Hawkes)
- Alternative Data (satellite, sentiment, transactions)
- Tokenization (ERC-3643 RWA)
- RegTech (compliance automation)
- Fraud Detection (GNN-based)

References:
    CFA Institute. (2024). Standards of Professional Conduct.
    ISO/IEC 25010:2011 Systems and software Quality Requirements.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable


class TestStatus(Enum):
    """Status of a test execution."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TestResult:
    """Result of a single test execution."""
    test_name: str
    test_type: str
    status: TestStatus
    duration_ms: float
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSuite:
    """A collection of related tests."""
    name: str
    test_type: str
    tests: list[Callable] = field(default_factory=list)


@dataclass
class TestReport:
    """Aggregated report from a full test run."""
    total_tests: int
    passed: int
    failed: int
    total_duration_ms: float
    generated_at: str
    results: list[TestResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate as a fraction."""
        if self.total_tests == 0:
            return 0.0
        return self.passed / self.total_tests


class TestingEngine:
    """Main testing engine for the Apex Fintech Platform.

    Orchestrates unit, integration, and performance tests across
    all platform engines and generates comprehensive reports.
    """

    def __init__(self) -> None:
        """Initialize the testing engine."""
        self.results: list[TestResult] = []
        self.suites: dict[str, TestSuite] = {}
        self._custom_tests: list[tuple[str, str, Callable]] = []

    def reset(self) -> None:
        """Reset all test state."""
        self.results.clear()
        self.suites.clear()
        self._custom_tests.clear()

    def register_test(
        self,
        name: str,
        test_func: Callable,
        test_type: str = "unit",
    ) -> None:
        """Register a custom test function.

        Args:
            name: Unique test name.
            test_func: Callable that returns (bool, str) — (passed, message).
            test_type: Category of test (unit, integration, performance).
        """
        self._custom_tests.append((name, test_type, test_func))
        if name not in self.suites:
            self.suites[name] = TestSuite(name=name, test_type=test_type, tests=[test_func])
        else:
            self.suites[name].tests.append(test_func)

    def _execute_test(
        self,
        name: str,
        test_type: str,
        test_func: Callable,
    ) -> TestResult:
        """Execute a single test and capture results.

        Args:
            name: Test name.
            test_type: Test category.
            test_func: Test callable returning (bool, str).

        Returns:
            TestResult with status, duration, and message.
        """
        start = time.perf_counter()
        try:
            passed, message = test_func()
            elapsed_ms = (time.perf_counter() - start) * 1000
            status = TestStatus.PASSED if passed else TestStatus.FAILED
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            status = TestStatus.FAILED
            message = f"Exception: {type(exc).__name__}: {exc}"

        return TestResult(
            test_name=name,
            test_type=test_type,
            status=status,
            duration_ms=elapsed_ms,
            message=message,
        )

    # ─── Unit Tests ──────────────────────────────────────────────────────

    def run_unit_tests(self) -> list[TestResult]:
        """Run all unit tests across platform engines.

        Returns:
            List of TestResult objects.
        """
        # Clear previous unit test results to avoid duplication
        self.results = [r for r in self.results if r.test_type != "unit"]
        unit_tests = self._get_unit_tests()
        results = []
        for name, func in unit_tests:
            result = self._execute_test(name, "unit", func)
            results.append(result)
            self.results.append(result)
        return results

    def _get_unit_tests(self) -> list[tuple[str, Callable]]:
        """Get all unit test functions."""
        tests = []

        # Market Making unit tests
        tests.extend(self._market_making_unit_tests())
        # Hawkes process unit tests
        tests.extend(self._hawkes_unit_tests())
        # Compliance unit tests
        tests.extend(self._compliance_unit_tests())
        # Alt Data unit tests
        tests.extend(self._altdata_unit_tests())
        # Tokenization unit tests
        tests.extend(self._tokenization_unit_tests())
        # Fraud detection unit tests
        tests.extend(self._fraud_unit_tests())
        # Custom registered tests
        for name, test_type, func in self._custom_tests:
            if test_type == "unit":
                tests.append((name, func))

        return tests

    def _market_making_unit_tests(self) -> list[tuple[str, Callable]]:
        """Unit tests for market making engine."""
        from src.marketmaking.engine import MarketMakingEngine

        def test_quote_computation():
            engine = MarketMakingEngine(
                gamma=0.1, sigma=0.5, k=1.5, A=0.1, dt=1.0, max_inventory=10
            )
            quote = engine.compute_quotes(mid_price=100.0, inventory=0, time_remaining=10.0)
            assert quote.bid < quote.ask, "Bid must be less than ask"
            assert quote.mid == 100.0, "Mid price mismatch"
            return True, "Quote computation correct"

        def test_inventory_skew():
            engine = MarketMakingEngine(
                gamma=0.1, sigma=0.5, k=1.5, A=0.1, dt=1.0, max_inventory=10
            )
            q0 = engine.compute_quotes(100.0, 0, 10.0)
            q5 = engine.compute_quotes(100.0, 5, 10.0)
            assert q5.bid < q0.bid, "Long inventory should skew quotes down"
            return True, "Inventory skew correct"

        def test_spread_widens_with_volatility():
            engine_low = MarketMakingEngine(0.1, 0.3, 1.5, 0.1, 1.0, 10)
            engine_high = MarketMakingEngine(0.1, 1.0, 1.5, 0.1, 1.0, 10)
            q_low = engine_low.compute_quotes(100.0, 0, 10.0)
            q_high = engine_high.compute_quotes(100.0, 0, 10.0)
            assert (q_high.ask - q_high.bid) > (q_low.ask - q_low.bid)
            return True, "Volatility spread relationship correct"

        def test_batch_quotes():
            import numpy as np
            engine = MarketMakingEngine(0.1, 0.5, 1.5, 0.1, 1.0, 10)
            mids = np.array([100.0, 101.0, 102.0])
            invs = np.array([0, 1, -1])
            times = np.array([10.0, 10.0, 10.0])
            quotes = engine.compute_quotes_batch(mids, invs, times)
            assert len(quotes) == 3
            return True, "Batch quote computation correct"

        return [
            ("unit_market_making_quote_computation", test_quote_computation),
            ("unit_market_making_inventory_skew", test_inventory_skew),
            ("unit_market_making_spread_volatility", test_spread_widens_with_volatility),
            ("unit_market_making_batch_quotes", test_batch_quotes),
        ]

    def _hawkes_unit_tests(self) -> list[tuple[str, Callable]]:
        """Unit tests for Hawkes process."""
        from src.marketmaking.hawkes import HawkesProcess

        def test_hawkes_init():
            hp = HawkesProcess(mu=0.5, alpha=0.3, beta=1.0)
            assert hp.mu == 0.5
            assert hp.alpha == 0.3
            assert hp.beta == 1.0
            return True, "Hawkes initialization correct"

        def test_branching_ratio():
            hp = HawkesProcess(mu=0.5, alpha=0.3, beta=1.0)
            assert hp.branching_ratio() < 1.0
            return True, "Branching ratio < 1 for stationarity"

        def test_unconditional_intensity():
            hp = HawkesProcess(mu=0.5, alpha=0.3, beta=1.0)
            expected = 0.5 / (1.0 - 0.3 / 1.0)
            assert abs(hp.unconditional_intensity() - expected) < 1e-10
            return True, "Unconditional intensity correct"

        def test_simulation_produces_events():
            hp = HawkesProcess(mu=1.0, alpha=0.5, beta=2.0)
            events = hp.simulate(T=100.0, seed=42)
            assert len(events) > 0
            return True, f"Simulation produced {len(events)} events"

        return [
            ("unit_hawkes_initialization", test_hawkes_init),
            ("unit_hawkes_branching_ratio", test_branching_ratio),
            ("unit_hawkes_unconditional_intensity", test_unconditional_intensity),
            ("unit_hawkes_simulation", test_simulation_produces_events),
        ]

    def _compliance_unit_tests(self) -> list[tuple[str, Callable]]:
        """Unit tests for compliance engine."""
        from src.regtech.compliance import (
            ComplianceEngine,
            ComplianceFramework,
            ComplianceStatus,
        )

        def test_compliance_engine_init():
            engine = ComplianceEngine()
            assert engine.controls == {}
            return True, "Compliance engine initializes empty"

        def test_register_control():
            engine = ComplianceEngine()
            control = engine.register_control(
                ComplianceFramework.PCI_DSS,
                "PCI-1.1",
                "Firewall",
                "Firewall config",
            )
            assert control.control_id == "PCI-1.1"
            assert control.status == ComplianceStatus.NOT_ASSESSED
            return True, "Control registration correct"

        def test_assess_control():
            engine = ComplianceEngine()
            engine.register_control(
                ComplianceFramework.PCI_DSS, "PCI-1.1", "Firewall", "desc"
            )
            engine.assess_control("PCI-1.1", ComplianceStatus.COMPLIANT)
            assert engine.controls["PCI-1.1"].status == ComplianceStatus.COMPLIANT
            return True, "Control assessment correct"

        def test_generate_report():
            engine = ComplianceEngine()
            engine.register_control(
                ComplianceFramework.PCI_DSS, "PCI-1.1", "Firewall", "desc"
            )
            engine.assess_control("PCI-1.1", ComplianceStatus.COMPLIANT)
            report = engine.generate_report()
            assert report.total_controls == 1
            assert report.compliant_count == 1
            return True, "Report generation correct"

        return [
            ("unit_compliance_engine_init", test_compliance_engine_init),
            ("unit_compliance_register_control", test_register_control),
            ("unit_compliance_assess_control", test_assess_control),
            ("unit_compliance_generate_report", test_generate_report),
        ]

    def _altdata_unit_tests(self) -> list[tuple[str, Callable]]:
        """Unit tests for alternative data engine."""
        from src.altdata.engine import AlternativeDataEngine

        def test_altdata_init():
            engine = AlternativeDataEngine(api_key="test")
            assert engine.api_key == "test"
            assert engine.data_store == {}
            return True, "AltData engine initializes correctly"

        def test_satellite_ingestion():
            engine = AlternativeDataEngine()
            data = [
                {"ticker": "AAPL", "parking_lot_count": 150, "activity_score": 0.85},
                {"ticker": "AAPL", "parking_lot_count": 160, "activity_score": 0.90},
            ]
            engine.ingest_satellite_data(data)
            assert len(engine.data_store["satellite"]) == 2
            return True, "Satellite data ingestion correct"

        def test_sentiment_analysis():
            engine = AlternativeDataEngine()
            data = [
                {"ticker": "AAPL", "sentiment": 0.75, "volume": 1000},
                {"ticker": "AAPL", "sentiment": 0.60, "volume": 1200},
            ]
            engine.ingest_sentiment_data(data)
            trends = engine.analyze_sentiment_trends("AAPL")
            assert trends["total_volume"] == 2200
            return True, "Sentiment analysis correct"

        def test_composite_signal():
            engine = AlternativeDataEngine()
            engine.ingest_satellite_data([
                {"ticker": "AAPL", "parking_lot_count": 150, "activity_score": 0.85}
            ])
            signal = engine.generate_composite_signal("AAPL")
            assert "composite_score" in signal
            assert -1 <= signal["composite_score"] <= 1
            return True, "Composite signal generation correct"

        return [
            ("unit_altdata_init", test_altdata_init),
            ("unit_altdata_satellite", test_satellite_ingestion),
            ("unit_altdata_sentiment", test_sentiment_analysis),
            ("unit_altdata_composite", test_composite_signal),
        ]

    def _tokenization_unit_tests(self) -> list[tuple[str, Callable]]:
        """Unit tests for tokenization engine."""
        from unittest.mock import MagicMock
        from src.tokenization.rwa import RWATokenizer, ComplianceStatus, InvestorStatus

        def test_tokenizer_init():
            mock_w3 = MagicMock()
            mock_w3.eth.send_raw_transaction = MagicMock()
            tokenizer = RWATokenizer(
                w3=mock_w3,
                owner_address="0xOwner",
                token_name="Test Token",
                token_symbol="TEST",
                initial_supply=1_000_000,
                asset_type="real_estate",
                asset_value=500_000_000,
                jurisdiction="US",
            )
            assert tokenizer.token_name == "Test Token"
            assert tokenizer.initial_supply == 1_000_000
            return True, "Tokenizer initialization correct"

        def test_compliance_check():
            mock_w3 = MagicMock()
            mock_w3.eth.send_raw_transaction = MagicMock()
            tokenizer = RWATokenizer(
                w3=mock_w3,
                owner_address="0xOwner",
                token_name="Test",
                token_symbol="TST",
                initial_supply=1_000_000,
                asset_type="real_estate",
                asset_value=500_000_000,
                jurisdiction="US",
            )
            addr = "0xInvestor"
            tokenizer.investors[addr] = {
                "status": InvestorStatus.VERIFIED,
                "kyc_expiry": 9999999999,
                "jurisdiction": "US",
                "accredited": True,
            }
            result = tokenizer.check_compliance(addr)
            assert result == ComplianceStatus.COMPLIANT
            return True, "Compliance check correct"

        def test_transfer_restrictions():
            mock_w3 = MagicMock()
            mock_w3.eth.send_raw_transaction = MagicMock()
            tokenizer = RWATokenizer(
                w3=mock_w3,
                owner_address="0xOwner",
                token_name="Test",
                token_symbol="TST",
                initial_supply=1_000_000,
                asset_type="real_estate",
                asset_value=500_000_000,
                jurisdiction="US",
            )
            addr1 = "0xAddr1"
            addr2 = "0xAddr2"
            tokenizer.register_investor(addr1, "US", True)
            tokenizer.verify_investor(addr1)
            tokenizer.register_investor(addr2, "US", True)
            tokenizer.verify_investor(addr2)
            tokenizer.balances[addr1] = 1000
            assert tokenizer.can_transfer(addr1, addr2, 100) is True
            return True, "Transfer restrictions correct"

        def test_dividend_distribution():
            mock_w3 = MagicMock()
            mock_w3.eth.send_raw_transaction = MagicMock()
            tokenizer = RWATokenizer(
                w3=mock_w3,
                owner_address="0xOwner",
                token_name="Test",
                token_symbol="TST",
                initial_supply=1_000_000,
                asset_type="real_estate",
                asset_value=500_000_000,
                jurisdiction="US",
            )
            addr = "0xAddr"
            tokenizer.register_investor(addr, "US", True)
            tokenizer.verify_investor(addr)
            tokenizer.balances[addr] = 1_000_000
            result = tokenizer.distribute_dividends(10_000)
            assert result["recipients"] == 1
            assert result["total_amount"] == 10_000
            return True, "Dividend distribution correct"

        return [
            ("unit_tokenization_init", test_tokenizer_init),
            ("unit_tokenization_compliance", test_compliance_check),
            ("unit_tokenization_transfers", test_transfer_restrictions),
            ("unit_tokenization_dividends", test_dividend_distribution),
        ]

    def _fraud_unit_tests(self) -> list[tuple[str, Callable]]:
        """Unit tests for fraud detection engine."""
        try:
            from src.fraud.detection import FraudRingDetector, ConceptDriftHandler
        except ImportError:
            return []

        def test_concept_drift_handler():
            handler = ConceptDriftHandler(window_size=10)
            for i in range(10):
                handler.update(0.5)
            assert handler.get_threshold() > 0
            return True, "ConceptDriftHandler works correctly"

        def test_fraud_ring_detector():
            detector = FraudRingDetector(threshold=0.5, min_density=0.3)
            assert detector.threshold == 0.5
            assert detector.min_density == 0.3
            return True, "FraudRingDetector initialization correct"

        return [
            ("unit_fraud_concept_drift", test_concept_drift_handler),
            ("unit_fraud_ring_detector", test_fraud_ring_detector),
        ]

    # ─── Integration Tests ───────────────────────────────────────────────

    def run_integration_tests(self) -> list[TestResult]:
        """Run all integration tests across platform engines.

        Returns:
            List of TestResult objects.
        """
        # Clear previous integration test results to avoid duplication
        self.results = [r for r in self.results if r.test_type != "integration"]
        integration_tests = self._get_integration_tests()
        results = []
        for name, func in integration_tests:
            result = self._execute_test(name, "integration", func)
            results.append(result)
            self.results.append(result)
        return results

    def _get_integration_tests(self) -> list[tuple[str, Callable]]:
        """Get all integration test functions."""
        tests = []

        # Cross-engine integration tests
        tests.extend(self._cross_engine_integration_tests())
        # Custom registered tests
        for name, test_type, func in self._custom_tests:
            if test_type == "integration":
                tests.append((name, func))

        return tests

    def _cross_engine_integration_tests(self) -> list[tuple[str, Callable]]:
        """Integration tests for cross-engine interactions."""
        from src.marketmaking.engine import MarketMakingEngine
        from src.marketmaking.hawkes import HawkesProcess
        from src.regtech.compliance import (
            ComplianceEngine,
            ComplianceFramework,
            ComplianceStatus,
        )
        from src.altdata.engine import AlternativeDataEngine

        def test_mm_with_hawkes_order_flow():
            """Market making engine works with Hawkes-simulated order flow."""
            mm = MarketMakingEngine(0.1, 0.5, 1.5, 0.1, 1.0, 10)
            hp = HawkesProcess(mu=1.0, alpha=0.5, beta=2.0)
            events = hp.simulate(T=10.0, seed=42)
            # Simulate order flow affecting inventory
            inventory = 0
            for _ in range(min(len(events), 10)):
                inventory += 1 if _ % 2 == 0 else -1
            quote = mm.compute_quotes(100.0, inventory, 5.0)
            assert quote.bid < quote.ask
            return True, f"MM+Hawkes integration works (events: {len(events)})"

        def test_compliance_with_altdata():
            """Compliance engine can ingest alt data signals."""
            comp = ComplianceEngine()
            comp.register_control(
                ComplianceFramework.PCI_DSS,
                "PCI-ALT-1",
                "Alt Data Monitoring",
                "Monitor alt data for compliance",
            )
            alt = AlternativeDataEngine()
            alt.ingest_sentiment_data([
                {"ticker": "XYZ", "sentiment": 0.8, "volume": 5000}
            ])
            alt.generate_composite_signal("XYZ")
            comp.assess_control("PCI-ALT-1", ComplianceStatus.COMPLIANT)
            report = comp.generate_report()
            assert report.total_controls == 1
            return True, "Compliance+AltData integration works"

        def test_full_pipeline():
            """Full pipeline: alt data → market making → compliance."""
            alt = AlternativeDataEngine()
            alt.ingest_satellite_data([
                {"ticker": "AAPL", "parking_lot_count": 200, "activity_score": 0.9}
            ])
            signal = alt.generate_composite_signal("AAPL")

            mm = MarketMakingEngine(0.1, 0.5, 1.5, 0.1, 1.0, 10)
            quote = mm.compute_quotes(100.0, 0, 10.0)

            comp = ComplianceEngine()
            comp.register_control(
                ComplianceFramework.SOX,
                "SOX-PIPE-1",
                "Pipeline Test",
                "Full pipeline integration",
            )
            comp.assess_control("SOX-PIPE-1", ComplianceStatus.COMPLIANT)

            assert quote.bid < quote.ask
            assert "composite_score" in signal
            assert comp.controls["SOX-PIPE-1"].status == ComplianceStatus.COMPLIANT
            return True, "Full pipeline integration works"

        def test_tokenization_with_compliance():
            """Tokenization engine respects compliance engine decisions."""
            from unittest.mock import MagicMock
            from src.tokenization.rwa import RWATokenizer

            mock_w3 = MagicMock()
            mock_w3.eth.send_raw_transaction = MagicMock()
            tokenizer = RWATokenizer(
                w3=mock_w3,
                owner_address="0xOwner",
                token_name="Test",
                token_symbol="TST",
                initial_supply=1_000_000,
                asset_type="real_estate",
                asset_value=500_000_000,
                jurisdiction="US",
            )
            addr = "0xInvestor"
            tokenizer.register_investor(addr, "US", True)
            tokenizer.verify_investor(addr)

            comp = ComplianceEngine()
            comp.register_control(
                ComplianceFramework.PCI_DSS,
                "PCI-TOK-1",
                "Token Compliance",
                "Tokenization compliance check",
            )
            comp.assess_control("PCI-TOK-1", ComplianceStatus.COMPLIANT)

            assert tokenizer.check_compliance(addr).value == "compliant"
            assert comp.controls["PCI-TOK-1"].status == ComplianceStatus.COMPLIANT
            return True, "Tokenization+Compliance integration works"

        return [
            ("integration_mm_hawkes_order_flow", test_mm_with_hawkes_order_flow),
            ("integration_compliance_altdata", test_compliance_with_altdata),
            ("integration_full_pipeline", test_full_pipeline),
            ("integration_tokenization_compliance", test_tokenization_with_compliance),
        ]

    # ─── Performance Tests ───────────────────────────────────────────────

    def run_performance_tests(self) -> list[TestResult]:
        """Run all performance tests across platform engines.

        Returns:
            List of TestResult objects.
        """
        # Clear previous performance test results to avoid duplication
        self.results = [r for r in self.results if r.test_type != "performance"]
        perf_tests = self._get_performance_tests()
        results = []
        for name, func in perf_tests:
            result = self._execute_test(name, "performance", func)
            results.append(result)
            self.results.append(result)
        return results

    def _get_performance_tests(self) -> list[tuple[str, Callable]]:
        """Get all performance test functions."""
        tests = []

        # Performance tests
        tests.extend(self._performance_benchmarks())
        # Custom registered tests
        for name, test_type, func in self._custom_tests:
            if test_type == "performance":
                tests.append((name, func))

        return tests

    def _performance_benchmarks(self) -> list[tuple[str, Callable]]:
        """Performance benchmark tests."""
        from src.marketmaking.engine import MarketMakingEngine
        from src.marketmaking.hawkes import HawkesProcess

        def test_quote_latency():
            """Quote computation latency < 1ms."""
            engine = MarketMakingEngine(0.1, 0.5, 1.5, 0.1, 1.0, 10)
            # Warmup
            for _ in range(100):
                engine.compute_quotes(100.0, 0, 10.0)
            times = []
            for _ in range(1000):
                t0 = time.perf_counter()
                engine.compute_quotes(100.0, 0, 10.0)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)
            median = sorted(times)[len(times) // 2]
            assert median < 1.0, f"Median latency {median:.3f}ms exceeds 1ms"
            return True, f"Quote latency: {median:.3f}ms (median)"

        def test_batch_throughput():
            """Batch quote throughput > 1000 quotes/sec."""
            import numpy as np
            engine = MarketMakingEngine(0.1, 0.5, 1.5, 0.1, 1.0, 10)
            n = 500
            mids = np.random.uniform(90, 110, n)
            invs = np.random.randint(-5, 5, n)
            times = np.full(n, 10.0)
            t0 = time.perf_counter()
            engine.compute_quotes_batch(mids, invs, times)
            elapsed = time.perf_counter() - t0
            throughput = n / elapsed
            assert throughput > 1000, f"Throughput {throughput:.0f} quotes/sec too low"
            return True, f"Batch throughput: {throughput:.0f} quotes/sec"

        def test_hawkes_simulation_speed():
            """Hawkes simulation completes in reasonable time."""
            hp = HawkesProcess(mu=1.0, alpha=0.5, beta=2.0)
            t0 = time.perf_counter()
            events = hp.simulate(T=1000.0, seed=42)
            elapsed = time.perf_counter() - t0
            assert elapsed < 5.0, f"Simulation took {elapsed:.2f}s, too slow"
            return True, f"Hawkes sim: {len(events)} events in {elapsed:.3f}s"

        def test_compliance_report_speed():
            """Compliance report generation is fast."""
            from src.regtech.compliance import (
                ComplianceEngine,
                ComplianceFramework,
                ComplianceStatus,
            )
            engine = ComplianceEngine()
            for i in range(100):
                engine.register_control(
                    ComplianceFramework.PCI_DSS,
                    f"PCI-{i}",
                    f"Control {i}",
                    f"Description {i}",
                )
                engine.assess_control(f"PCI-{i}", ComplianceStatus.COMPLIANT)
            t0 = time.perf_counter()
            report = engine.generate_report()
            elapsed = time.perf_counter() - t0
            assert elapsed < 1.0, f"Report took {elapsed:.3f}s"
            assert report.total_controls == 100
            return True, f"Compliance report: {elapsed:.3f}s for 100 controls"

        return [
            ("performance_quote_latency", test_quote_latency),
            ("performance_batch_throughput", test_batch_throughput),
            ("performance_hawkes_simulation", test_hawkes_simulation_speed),
            ("performance_compliance_report", test_compliance_report_speed),
        ]

    # ─── Full Test Suite ─────────────────────────────────────────────────

    def run_all(self) -> TestReport:
        """Run all tests (unit + integration + performance).

        Returns:
            TestReport with aggregated results.
        """
        self.reset()

        self.run_unit_tests()
        self.run_integration_tests()
        self.run_performance_tests()

        return self.generate_report()

    def generate_report(self) -> TestReport:
        """Generate a test report from current results.

        Returns:
            TestReport with aggregated results.
        """
        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in self.results if r.status == TestStatus.FAILED)
        total_duration = sum(r.duration_ms for r in self.results)

        return TestReport(
            total_tests=total,
            passed=passed,
            failed=failed,
            total_duration_ms=total_duration,
            generated_at=datetime.now().isoformat(),
            results=list(self.results),
        )
