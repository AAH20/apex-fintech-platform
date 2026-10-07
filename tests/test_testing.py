"""TDD tests for the TestingEngine meta-framework.

The TestingEngine orchestrates unit, integration, and performance tests
across all platform engines (market making, alt data, tokenization, regtech).
"""
import pytest

from src.testing.engine import (
    TestingEngine,
    TestResult,
    TestReport,
    TestStatus,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def engine():
    """Create a fresh TestingEngine instance."""
    return TestingEngine()


@pytest.fixture
def mm_engine():
    """Create a market making engine for testing."""
    from src.marketmaking.engine import MarketMakingEngine
    return MarketMakingEngine(
        gamma=0.1, sigma=0.5, k=1.5, A=0.1, dt=1.0, max_inventory=10
    )


@pytest.fixture
def hawkes():
    """Create a Hawkes process for testing."""
    from src.marketmaking.hawkes import HawkesProcess
    return HawkesProcess(mu=0.5, alpha=0.3, beta=1.0)


@pytest.fixture
def alt_data_engine():
    """Create an alternative data engine for testing."""
    from src.altdata.engine import AlternativeDataEngine
    return AlternativeDataEngine(api_key="test_key")


@pytest.fixture
def compliance_engine():
    """Create a compliance engine for testing."""
    from src.regtech.compliance import ComplianceEngine, ComplianceFramework
    return ComplianceEngine(frameworks=[ComplianceFramework.PCI_DSS])


# ─── Test 1: Engine Initialization ───────────────────────────────────────────


class TestTestingEngineInit:
    """Test TestingEngine initialization and basic properties."""

    def test_engine_initializes_with_default_config(self, engine):
        """Engine initializes with sensible defaults."""
        assert engine is not None
        assert hasattr(engine, 'results')
        assert hasattr(engine, 'suites')
        assert isinstance(engine.results, list)
        assert isinstance(engine.suites, dict)

    def test_engine_initializes_empty(self, engine):
        """Engine starts with no test results."""
        assert len(engine.results) == 0
        assert len(engine.suites) == 0

    def test_engine_has_test_runner_methods(self, engine):
        """Engine has methods for each test type."""
        assert hasattr(engine, 'run_unit_tests')
        assert hasattr(engine, 'run_integration_tests')
        assert hasattr(engine, 'run_performance_tests')
        assert hasattr(engine, 'run_all')
        assert hasattr(engine, 'generate_report')

    def test_engine_has_register_method(self, engine):
        """Engine can register custom tests."""
        assert hasattr(engine, 'register_test')
        assert callable(engine.register_test)


# ─── Test 2: Unit Testing ────────────────────────────────────────────────────


class TestUnitTesting:
    """Test unit-level testing capabilities."""

    def test_run_unit_tests_returns_results(self, engine):
        """Running unit tests returns test results."""
        results = engine.run_unit_tests()
        assert isinstance(results, list)
        assert len(results) > 0

    def test_unit_tests_include_market_making(self, engine):
        """Unit tests cover market making engine."""
        results = engine.run_unit_tests()
        test_names = [r.test_name for r in results]
        assert any("market_making" in name.lower() or "quote" in name.lower()
                    for name in test_names)

    def test_unit_tests_include_hawkes(self, engine):
        """Unit tests cover Hawkes process."""
        results = engine.run_unit_tests()
        test_names = [r.test_name for r in results]
        assert any("hawkes" in name.lower() for name in test_names)

    def test_unit_tests_include_compliance(self, engine):
        """Unit tests cover compliance engine."""
        results = engine.run_unit_tests()
        test_names = [r.test_name for r in results]
        assert any("compliance" in name.lower() for name in test_names)

    def test_unit_tests_include_altdata(self, engine):
        """Unit tests cover alternative data engine."""
        results = engine.run_unit_tests()
        test_names = [r.test_name for r in results]
        assert any("altdata" in name.lower() or "alt_data" in name.lower()
                    for name in test_names)

    def test_unit_test_results_have_status(self, engine):
        """Each unit test result has a pass/fail status."""
        results = engine.run_unit_tests()
        for result in results:
            assert hasattr(result, 'status')
            assert result.status in (TestStatus.PASSED, TestStatus.FAILED)

    def test_unit_tests_all_pass(self, engine):
        """All unit tests should pass."""
        results = engine.run_unit_tests()
        passed = sum(1 for r in results if r.status == TestStatus.PASSED)
        assert passed == len(results), f"{len(results) - passed} unit tests failed"

    def test_unit_tests_record_duration(self, engine):
        """Unit tests record execution time."""
        results = engine.run_unit_tests()
        for result in results:
            assert hasattr(result, 'duration_ms')
            assert result.duration_ms >= 0


# ─── Test 3: Integration Testing ─────────────────────────────────────────────


class TestIntegrationTesting:
    """Test integration-level testing capabilities."""

    def test_run_integration_tests_returns_results(self, engine):
        """Running integration tests returns test results."""
        results = engine.run_integration_tests()
        assert isinstance(results, list)
        assert len(results) > 0

    def test_integration_tests_cover_cross_engine(self, engine):
        """Integration tests verify cross-engine interactions."""
        results = engine.run_integration_tests()
        test_names = [r.test_name for r in results]
        assert any("integration" in name.lower() or "cross" in name.lower()
                    for name in test_names)

    def test_integration_tests_all_pass(self, engine):
        """All integration tests should pass."""
        results = engine.run_integration_tests()
        passed = sum(1 for r in results if r.status == TestStatus.PASSED)
        assert passed == len(results), f"{len(results) - passed} integration tests failed"

    def test_integration_tests_have_duration(self, engine):
        """Integration tests record execution time."""
        results = engine.run_integration_tests()
        for result in results:
            assert hasattr(result, 'duration_ms')
            assert result.duration_ms >= 0


# ─── Test 4: Performance Testing ─────────────────────────────────────────────


class TestPerformanceTesting:
    """Test performance-level testing capabilities."""

    def test_run_performance_tests_returns_results(self, engine):
        """Running performance tests returns test results."""
        results = engine.run_performance_tests()
        assert isinstance(results, list)
        assert len(results) > 0

    def test_performance_tests_include_latency(self, engine):
        """Performance tests measure latency."""
        results = engine.run_performance_tests()
        test_names = [r.test_name for r in results]
        assert any("latency" in name.lower() or "speed" in name.lower()
                    for name in test_names)

    def test_performance_tests_include_throughput(self, engine):
        """Performance tests measure throughput."""
        results = engine.run_performance_tests()
        test_names = [r.test_name for r in results]
        assert any("throughput" in name.lower() or "batch" in name.lower()
                    for name in test_names)

    def test_performance_tests_all_pass(self, engine):
        """All performance tests should pass."""
        results = engine.run_performance_tests()
        passed = sum(1 for r in results if r.status == TestStatus.PASSED)
        assert passed == len(results), f"{len(results) - passed} performance tests failed"

    def test_performance_tests_have_metrics(self, engine):
        """Performance tests include timing metrics."""
        results = engine.run_performance_tests()
        for result in results:
            assert hasattr(result, 'duration_ms')
            assert result.duration_ms > 0


# ─── Test 5: Full Test Suite ─────────────────────────────────────────────────


class TestFullTestSuite:
    """Test running the complete test suite."""

    def test_run_all_returns_report(self, engine):
        """Running all tests returns a TestReport."""
        report = engine.run_all()
        assert isinstance(report, TestReport)

    def test_run_all_report_has_counts(self, engine):
        """Report includes pass/fail counts."""
        report = engine.run_all()
        assert hasattr(report, 'total_tests')
        assert hasattr(report, 'passed')
        assert hasattr(report, 'failed')
        assert report.total_tests == report.passed + report.failed

    def test_run_all_report_has_duration(self, engine):
        """Report includes total duration."""
        report = engine.run_all()
        assert hasattr(report, 'total_duration_ms')
        assert report.total_duration_ms > 0

    def test_run_all_report_has_timestamp(self, engine):
        """Report includes generation timestamp."""
        report = engine.run_all()
        assert hasattr(report, 'generated_at')
        assert report.generated_at is not None

    def test_run_all_includes_all_test_types(self, engine):
        """Full suite includes unit, integration, and performance tests."""
        report = engine.run_all()
        test_types = set()
        for result in report.results:
            test_types.add(result.test_type)
        assert "unit" in test_types
        assert "integration" in test_types
        assert "performance" in test_types

    def test_run_all_all_pass(self, engine):
        """All tests in full suite should pass."""
        report = engine.run_all()
        assert report.failed == 0, f"{report.failed} tests failed"


# ─── Test 6: Custom Test Registration ────────────────────────────────────────


class TestCustomTestRegistration:
    """Test registering and running custom tests."""

    def test_register_custom_test(self, engine):
        """Can register a custom test function."""
        def custom_test():
            return True, "Custom test passed"

        engine.register_test("custom_check", custom_test, test_type="unit")
        assert "custom_check" in engine.suites

    def test_custom_test_appears_in_results(self, engine):
        """Registered custom test appears in results."""
        def custom_test():
            return True, "Custom test passed"

        engine.register_test("my_custom_test", custom_test, test_type="unit")
        results = engine.run_unit_tests()
        test_names = [r.test_name for r in results]
        assert "my_custom_test" in test_names

    def test_custom_test_failure_recorded(self, engine):
        """Failed custom test is recorded correctly."""
        def failing_test():
            return False, "Intentional failure"

        engine.register_test("failing_test", failing_test, test_type="unit")
        results = engine.run_unit_tests()
        failing_results = [r for r in results if r.test_name == "failing_test"]
        assert len(failing_results) == 1
        assert failing_results[0].status == TestStatus.FAILED


# ─── Test 7: TestResult Data Class ───────────────────────────────────────────


class TestTestResult:
    """Test TestResult data class."""

    def test_result_has_required_fields(self):
        """TestResult has all required fields."""
        result = TestResult(
            test_name="test_example",
            test_type="unit",
            status=TestStatus.PASSED,
            duration_ms=1.5,
            message="OK",
        )
        assert result.test_name == "test_example"
        assert result.test_type == "unit"
        assert result.status == TestStatus.PASSED
        assert result.duration_ms == 1.5
        assert result.message == "OK"

    def test_result_status_enum(self):
        """TestStatus has PASSED and FAILED."""
        assert TestStatus.PASSED is not None
        assert TestStatus.FAILED is not None
        assert TestStatus.PASSED != TestStatus.FAILED


# ─── Test 8: TestReport Data Class ───────────────────────────────────────────


class TestTestReport:
    """Test TestReport data class."""

    def test_report_has_required_fields(self):
        """TestReport has all required fields."""
        report = TestReport(
            total_tests=10,
            passed=10,
            failed=0,
            total_duration_ms=100.0,
            generated_at="2024-01-01T00:00:00",
            results=[],
        )
        assert report.total_tests == 10
        assert report.passed == 10
        assert report.failed == 0
        assert report.total_duration_ms == 100.0
        assert report.generated_at == "2024-01-01T00:00:00"
        assert report.results == []

    def test_report_pass_rate(self):
        """Report calculates pass rate correctly."""
        report = TestReport(
            total_tests=10,
            passed=8,
            failed=2,
            total_duration_ms=50.0,
            generated_at="2024-01-01T00:00:00",
            results=[],
        )
        assert report.pass_rate == 0.8


# ─── Test 9: Engine Resets ───────────────────────────────────────────────────


class TestEngineReset:
    """Test engine state management."""

    def test_reset_clears_results(self, engine):
        """Reset clears all test results."""
        engine.run_unit_tests()
        assert len(engine.results) > 0
        engine.reset()
        assert len(engine.results) == 0

    def test_reset_clears_suites(self, engine):
        """Reset clears custom test suites."""
        def dummy():
            return True, "ok"
        engine.register_test("dummy", dummy)
        assert len(engine.suites) > 0
        engine.reset()
        assert len(engine.suites) == 0


# ─── Test 10: Edge Cases ─────────────────────────────────────────────────────


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_run_unit_tests_twice_no_duplication(self, engine):
        """Running unit tests twice doesn't duplicate results."""
        engine.run_unit_tests()
        count_after_first = len(engine.results)
        engine.run_unit_tests()
        count_after_second = len(engine.results)
        assert count_after_second == count_after_first

    def test_generate_report_without_running_tests(self, engine):
        """Generating report without tests produces empty report."""
        report = engine.generate_report()
        assert isinstance(report, TestReport)
        assert report.total_tests == 0

    def test_engine_handles_import_errors_gracefully(self, engine):
        """Engine handles missing modules gracefully."""
        # Should not raise even if some engines have issues
        results = engine.run_unit_tests()
        assert isinstance(results, list)
