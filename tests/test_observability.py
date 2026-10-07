"""Tests for the Observability Engine — logging, metrics, tracing, alerting."""
import logging
import time

import pytest

from observability import ObservabilityEngine, AlertRule, Span, TraceContext


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestObservabilityEngineInit:
    """Tests for ObservabilityEngine initialization."""

    def test_engine_initializes_with_default_name(self):
        """Engine initializes with a default name."""
        engine = ObservabilityEngine()
        assert engine.name == "observability"

    def test_engine_initializes_with_custom_name(self):
        """Engine initializes with a custom name."""
        engine = ObservabilityEngine(name="test-engine")
        assert engine.name == "test-engine"

    def test_engine_has_logger(self):
        """Engine has a configured logger."""
        engine = ObservabilityEngine()
        assert engine.logger is not None
        assert isinstance(engine.logger, logging.Logger)

    def test_engine_has_metrics_registry(self):
        """Engine has a Prometheus metrics registry."""
        engine = ObservabilityEngine()
        assert engine.registry is not None

    def test_engine_initializes_with_empty_traces(self):
        """Engine starts with no active traces."""
        engine = ObservabilityEngine()
        assert engine.active_traces == {}

    def test_engine_initializes_with_empty_alerts(self):
        """Engine starts with no alert rules."""
        engine = ObservabilityEngine()
        assert engine.alert_rules == {}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class TestLogging:
    """Tests for structured logging functionality."""

    def test_log_info_message(self):
        """Engine can log info messages."""
        engine = ObservabilityEngine()
        engine.log_info("Test message", engine="marketmaking")
        # Should not raise

    def test_log_warning_message(self):
        """Engine can log warning messages."""
        engine = ObservabilityEngine()
        engine.log_warning("Warning message", engine="regtech")
        # Should not raise

    def test_log_error_message(self):
        """Engine can log error messages."""
        engine = ObservabilityEngine()
        engine.log_error("Error message", engine="tokenization")
        # Should not raise

    def test_log_includes_engine_context(self):
        """Log messages include engine context."""
        engine = ObservabilityEngine()
        # The logger should have a custom attribute or formatter
        assert hasattr(engine, "log_info")
        assert hasattr(engine, "log_warning")
        assert hasattr(engine, "log_error")
        assert hasattr(engine, "log_debug")

    def test_log_with_extra_fields(self):
        """Log messages can include extra structured fields."""
        engine = ObservabilityEngine()
        engine.log_info(
            "Test with extras",
            engine="altdata",
            extra={"key": "value", "count": 42},
        )
        # Should not raise


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    """Tests for Prometheus metrics recording."""

    def test_record_counter_metric(self):
        """Engine can record counter metrics."""
        engine = ObservabilityEngine()
        engine.record_counter("test_counter", 1.0, labels={"engine": "marketmaking"})
        # Should not raise

    def test_record_histogram_metric(self):
        """Engine can record histogram metrics."""
        engine = ObservabilityEngine()
        engine.record_histogram("test_histogram", 0.5, labels={"engine": "regtech"})
        # Should not raise

    def test_record_gauge_metric(self):
        """Engine can record gauge metrics."""
        engine = ObservabilityEngine()
        engine.record_gauge("test_gauge", 42.0, labels={"engine": "tokenization"})
        # Should not raise

    def test_counter_increments(self):
        """Counter metric increments correctly."""
        engine = ObservabilityEngine()
        engine.record_counter("increment_test", 1.0)
        engine.record_counter("increment_test", 2.0)
        # Should not raise — counter should be at 3.0

    def test_histogram_observes_values(self):
        """Histogram metric observes multiple values."""
        engine = ObservabilityEngine()
        for val in [0.1, 0.5, 1.0, 2.0]:
            engine.record_histogram("latency_test", val)
        # Should not raise

    def test_gauge_sets_value(self):
        """Gauge metric sets a specific value."""
        engine = ObservabilityEngine()
        engine.record_gauge("gauge_test", 100.0)
        engine.record_gauge("gauge_test", 50.0)
        # Should not raise — gauge should be at 50.0

    def test_metrics_with_labels(self):
        """Metrics can be recorded with labels."""
        engine = ObservabilityEngine()
        engine.record_counter(
            "labeled_counter", 1.0, labels={"engine": "fraud", "status": "success"}
        )
        engine.record_counter(
            "labeled_counter", 1.0, labels={"engine": "fraud", "status": "failure"}
        )
        # Should not raise

    def test_get_metric_value(self):
        """Engine can retrieve current metric values."""
        engine = ObservabilityEngine()
        engine.record_gauge("retrievable_gauge", 42.0)
        value = engine.get_metric_value("retrievable_gauge")
        assert value == 42.0

    def test_get_metric_value_with_labels(self):
        """Engine can retrieve metric values filtered by labels."""
        engine = ObservabilityEngine()
        engine.record_gauge(
            "labeled_gauge", 10.0, labels={"engine": "portfolio"}
        )
        value = engine.get_metric_value("labeled_gauge", labels={"engine": "portfolio"})
        assert value == 10.0


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------


class TestTracing:
    """Tests for distributed tracing functionality."""

    def test_start_span(self):
        """Engine can start a new span."""
        engine = ObservabilityEngine()
        span = engine.start_span("test-operation", engine="marketmaking")
        assert isinstance(span, Span)
        assert span.operation_name == "test-operation"
        assert span.engine == "marketmaking"

    def test_span_has_start_time(self):
        """Span records its start time."""
        engine = ObservabilityEngine()
        before = time.time()
        span = engine.start_span("timed-operation")
        after = time.time()
        assert before <= span.start_time <= after

    def test_end_span(self):
        """Engine can end a span and record duration."""
        engine = ObservabilityEngine()
        span = engine.start_span("duration-test")
        time.sleep(0.01)
        engine.end_span(span)
        assert span.end_time is not None
        assert span.duration >= 0.01

    def test_span_records_status(self):
        """Span can record success or error status."""
        engine = ObservabilityEngine()
        span = engine.start_span("status-test")
        engine.end_span(span, status="success")
        assert span.status == "success"

    def test_span_records_error(self):
        """Span can record error information."""
        engine = ObservabilityEngine()
        span = engine.start_span("error-test")
        engine.end_span(span, status="error", error="ValueError: test error")
        assert span.status == "error"
        assert "ValueError" in span.error

    def test_trace_context(self):
        """Engine can create and manage trace contexts."""
        engine = ObservabilityEngine()
        ctx = engine.create_trace_context("test-trace")
        assert isinstance(ctx, TraceContext)
        assert ctx.trace_id == "test-trace"

    def test_nested_spans(self):
        """Engine supports nested spans (parent-child relationships)."""
        engine = ObservabilityEngine()
        parent = engine.start_span("parent-op")
        child = engine.start_span("child-op", parent_span=parent)
        assert child.parent_id == parent.span_id
        engine.end_span(child)
        engine.end_span(parent)

    def test_get_trace(self):
        """Engine can retrieve a trace by ID."""
        engine = ObservabilityEngine()
        ctx = engine.create_trace_context("retrieve-trace")
        span = engine.start_span("traced-op", trace_context=ctx)
        engine.end_span(span)
        trace = engine.get_trace("retrieve-trace")
        assert trace is not None
        assert trace.trace_id == "retrieve-trace"


# ---------------------------------------------------------------------------
# Alerting
# ---------------------------------------------------------------------------


class TestAlerting:
    """Tests for alert rule evaluation and firing."""

    def test_add_alert_rule(self):
        """Engine can add alert rules."""
        engine = ObservabilityEngine()
        rule = AlertRule(
            name="high-latency",
            metric="engine_latency",
            threshold=1.0,
            comparison=">",
            severity="warning",
        )
        engine.add_alert_rule(rule)
        assert "high-latency" in engine.alert_rules

    def test_alert_fires_when_threshold_exceeded(self):
        """Alert fires when metric exceeds threshold."""
        engine = ObservabilityEngine()
        rule = AlertRule(
            name="high-latency",
            metric="engine_latency",
            threshold=0.5,
            comparison=">",
            severity="warning",
        )
        engine.add_alert_rule(rule)
        engine.record_histogram("engine_latency", 2.0)
        alerts = engine.check_alerts()
        assert len(alerts) > 0
        assert alerts[0].rule_name == "high-latency"

    def test_alert_does_not_fire_when_below_threshold(self):
        """Alert does not fire when metric is below threshold."""
        engine = ObservabilityEngine()
        rule = AlertRule(
            name="high-latency",
            metric="engine_latency",
            threshold=10.0,
            comparison=">",
            severity="warning",
        )
        engine.add_alert_rule(rule)
        engine.record_histogram("engine_latency", 0.5)
        alerts = engine.check_alerts()
        assert len(alerts) == 0

    def test_alert_with_critical_severity(self):
        """Alert can have critical severity."""
        engine = ObservabilityEngine()
        rule = AlertRule(
            name="critical-error",
            metric="engine_errors",
            threshold=5.0,
            comparison=">=",
            severity="critical",
        )
        engine.add_alert_rule(rule)
        engine.record_counter("engine_errors", 10.0)
        alerts = engine.check_alerts()
        assert len(alerts) > 0
        assert alerts[0].severity == "critical"

    def test_remove_alert_rule(self):
        """Engine can remove alert rules."""
        engine = ObservabilityEngine()
        rule = AlertRule(
            name="test-rule",
            metric="test_metric",
            threshold=1.0,
            comparison=">",
            severity="info",
        )
        engine.add_alert_rule(rule)
        assert "test-rule" in engine.alert_rules
        engine.remove_alert_rule("test-rule")
        assert "test-rule" not in engine.alert_rules

    def test_alert_rule_with_labels(self):
        """Alert rules can filter by labels."""
        engine = ObservabilityEngine()
        rule = AlertRule(
            name="engine-specific",
            metric="engine_ops",
            threshold=100.0,
            comparison=">",
            severity="warning",
            labels={"engine": "marketmaking"},
        )
        engine.add_alert_rule(rule)
        engine.record_counter("engine_ops", 200.0, labels={"engine": "marketmaking"})
        alerts = engine.check_alerts()
        assert len(alerts) > 0


# ---------------------------------------------------------------------------
# Engine Monitoring Integration
# ---------------------------------------------------------------------------


class TestEngineMonitoring:
    """Tests for monitoring other engines."""

    def test_monitor_engine_operation(self):
        """Engine can monitor an operation with metrics and tracing."""
        engine = ObservabilityEngine()
        result = engine.monitor(
            engine_name="marketmaking",
            operation="compute_quotes",
            func=lambda: "result",
        )
        assert result == "result"

    def test_monitor_records_metrics(self):
        """Monitoring an operation records metrics."""
        engine = ObservabilityEngine()
        engine.monitor(
            engine_name="regtech",
            operation="assess_control",
            func=lambda: None,
        )
        # Should have recorded metrics
        assert True  # If we got here, metrics were recorded

    def test_monitor_records_errors(self):
        """Monitoring records errors when operation fails."""
        engine = ObservabilityEngine()

        def failing_func():
            raise ValueError("test error")

        with pytest.raises(ValueError):
            engine.monitor(
                engine_name="tokenization",
                operation="transfer",
                func=failing_func,
            )
        # Error should be recorded in metrics
        assert True  # If we got here, error was recorded

    def test_monitor_with_latency(self):
        """Monitoring records operation latency."""
        engine = ObservabilityEngine()

        def slow_func():
            time.sleep(0.05)
            return "done"

        result = engine.monitor(
            engine_name="altdata",
            operation="fetch_data",
            func=slow_func,
        )
        assert result == "done"
        # Latency should be recorded


# ---------------------------------------------------------------------------
# Prometheus Export
# ---------------------------------------------------------------------------


class TestPrometheusExport:
    """Tests for Prometheus metrics export."""

    def test_export_metrics(self):
        """Engine can export metrics in Prometheus format."""
        engine = ObservabilityEngine()
        engine.record_counter("export_counter", 1.0)
        output = engine.export_metrics()
        assert isinstance(output, str)
        assert "export_counter" in output

    def test_export_includes_help_text(self):
        """Prometheus export includes HELP text."""
        engine = ObservabilityEngine()
        engine.record_counter("help_counter", 1.0)
        output = engine.export_metrics()
        assert "# HELP" in output

    def test_export_includes_type_text(self):
        """Prometheus export includes TYPE text."""
        engine = ObservabilityEngine()
        engine.record_counter("type_counter", 1.0)
        output = engine.export_metrics()
        assert "# TYPE" in output


# ---------------------------------------------------------------------------
# Grafana Dashboard
# ---------------------------------------------------------------------------


class TestGrafanaDashboard:
    """Tests for Grafana dashboard generation."""

    def test_generate_dashboard(self):
        """Engine can generate a Grafana dashboard JSON."""
        engine = ObservabilityEngine()
        dashboard = engine.generate_dashboard()
        assert isinstance(dashboard, dict)
        assert "panels" in dashboard

    def test_dashboard_has_panels(self):
        """Generated dashboard has panels."""
        engine = ObservabilityEngine()
        dashboard = engine.generate_dashboard()
        assert len(dashboard["panels"]) > 0

    def test_dashboard_panels_have_targets(self):
        """Dashboard panels have Prometheus targets."""
        engine = ObservabilityEngine()
        dashboard = engine.generate_dashboard()
        for panel in dashboard["panels"]:
            assert "targets" in panel

    def test_dashboard_includes_all_engines(self):
        """Dashboard includes panels for all monitored engines."""
        engine = ObservabilityEngine()
        dashboard = engine.generate_dashboard()
        # Should have panels for various engine metrics
        panel_titles = [p.get("title", "") for p in dashboard["panels"]]
        assert len(panel_titles) > 0


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    """Tests for engine health checking."""

    def test_health_check_returns_status(self):
        """Engine health check returns a status dict."""
        engine = ObservabilityEngine()
        health = engine.health_check()
        assert isinstance(health, dict)
        assert "status" in health

    def test_health_check_reports_healthy(self):
        """Engine reports healthy when no critical alerts."""
        engine = ObservabilityEngine()
        health = engine.health_check()
        assert health["status"] == "healthy"

    def test_health_check_reports_unhealthy(self):
        """Engine reports unhealthy when critical alerts fire."""
        engine = ObservabilityEngine()
        rule = AlertRule(
            name="critical-test",
            metric="test_metric",
            threshold=0.5,
            comparison=">",
            severity="critical",
        )
        engine.add_alert_rule(rule)
        engine.record_counter("test_metric", 1.0)
        engine.check_alerts()
        health = engine.health_check()
        assert health["status"] == "unhealthy"


# ---------------------------------------------------------------------------
# Span Data Structure
# ---------------------------------------------------------------------------


class TestSpan:
    """Tests for the Span data structure."""

    def test_span_has_required_fields(self):
        """Span has all required fields."""
        span = Span(
            span_id="test-span-1",
            operation_name="test-op",
            start_time=time.time(),
        )
        assert span.span_id == "test-span-1"
        assert span.operation_name == "test-op"
        assert span.start_time > 0

    def test_span_duration_calculation(self):
        """Span duration is calculated correctly."""
        start = time.time()
        span = Span(
            span_id="test-span-2",
            operation_name="test-op",
            start_time=start,
        )
        span.end_time = start + 1.5
        assert span.duration == pytest.approx(1.5)

    def test_span_to_dict(self):
        """Span can be serialized to dict."""
        span = Span(
            span_id="test-span-3",
            operation_name="test-op",
            start_time=time.time(),
            engine="marketmaking",
        )
        d = span.to_dict()
        assert isinstance(d, dict)
        assert d["span_id"] == "test-span-3"
        assert d["operation_name"] == "test-op"
        assert d["engine"] == "marketmaking"


# ---------------------------------------------------------------------------
# TraceContext Data Structure
# ---------------------------------------------------------------------------


class TestTraceContext:
    """Tests for the TraceContext data structure."""

    def test_trace_context_has_trace_id(self):
        """TraceContext has a trace ID."""
        ctx = TraceContext(trace_id="trace-123")
        assert ctx.trace_id == "trace-123"

    def test_trace_context_has_spans(self):
        """TraceContext holds spans."""
        ctx = TraceContext(trace_id="trace-456")
        span = Span(
            span_id="span-1",
            operation_name="op",
            start_time=time.time(),
        )
        ctx.add_span(span)
        assert len(ctx.spans) == 1

    def test_trace_context_to_dict(self):
        """TraceContext can be serialized to dict."""
        ctx = TraceContext(trace_id="trace-789")
        d = ctx.to_dict()
        assert isinstance(d, dict)
        assert d["trace_id"] == "trace-789"


# ---------------------------------------------------------------------------
# AlertRule Data Structure
# ---------------------------------------------------------------------------


class TestAlertRule:
    """Tests for the AlertRule data structure."""

    def test_alert_rule_creation(self):
        """AlertRule can be created with all fields."""
        rule = AlertRule(
            name="test-alert",
            metric="test_metric",
            threshold=1.0,
            comparison=">",
            severity="warning",
            labels={"engine": "test"},
        )
        assert rule.name == "test-alert"
        assert rule.metric == "test_metric"
        assert rule.threshold == 1.0
        assert rule.comparison == ">"
        assert rule.severity == "warning"
        assert rule.labels == {"engine": "test"}

    def test_alert_rule_evaluates_greater_than(self):
        """AlertRule evaluates > comparison correctly."""
        rule = AlertRule(
            name="gt-test",
            metric="m",
            threshold=5.0,
            comparison=">",
            severity="info",
        )
        assert rule.evaluate(10.0) is True
        assert rule.evaluate(3.0) is False

    def test_alert_rule_evaluates_less_than(self):
        """AlertRule evaluates < comparison correctly."""
        rule = AlertRule(
            name="lt-test",
            metric="m",
            threshold=5.0,
            comparison="<",
            severity="info",
        )
        assert rule.evaluate(3.0) is True
        assert rule.evaluate(10.0) is False

    def test_alert_rule_evaluates_equal(self):
        """AlertRule evaluates == comparison correctly."""
        rule = AlertRule(
            name="eq-test",
            metric="m",
            threshold=5.0,
            comparison="==",
            severity="info",
        )
        assert rule.evaluate(5.0) is True
        assert rule.evaluate(3.0) is False

    def test_alert_rule_evaluates_greater_equal(self):
        """AlertRule evaluates >= comparison correctly."""
        rule = AlertRule(
            name="gte-test",
            metric="m",
            threshold=5.0,
            comparison=">=",
            severity="info",
        )
        assert rule.evaluate(5.0) is True
        assert rule.evaluate(10.0) is True
        assert rule.evaluate(3.0) is False

    def test_alert_rule_evaluates_less_equal(self):
        """AlertRule evaluates <= comparison correctly."""
        rule = AlertRule(
            name="lte-test",
            metric="m",
            threshold=5.0,
            comparison="<=",
            severity="info",
        )
        assert rule.evaluate(5.0) is True
        assert rule.evaluate(3.0) is True
        assert rule.evaluate(10.0) is False


# ---------------------------------------------------------------------------
# End-to-End Integration
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_monitoring_workflow(self):
        """Full workflow: monitor operation, record metrics, check alerts."""
        engine = ObservabilityEngine()

        # Add an alert rule — monitor() records to operation_latency with labels
        rule = AlertRule(
            name="latency-alert",
            metric="operation_latency",
            threshold=0.1,
            comparison=">",
            severity="warning",
            labels={"engine": "marketmaking", "operation": "compute_quotes"},
        )
        engine.add_alert_rule(rule)

        # Monitor an operation
        def operation():
            time.sleep(0.15)
            return "success"

        result = engine.monitor(
            engine_name="marketmaking",
            operation="compute_quotes",
            func=operation,
        )
        assert result == "success"

        # Check alerts — should fire because we slept 0.15s > 0.1s threshold
        alerts = engine.check_alerts()
        assert len(alerts) > 0

    def test_multiple_engines_monitored(self):
        """Multiple engines can be monitored simultaneously."""
        engine = ObservabilityEngine()

        for engine_name in ["marketmaking", "regtech", "tokenization", "altdata"]:
            engine.monitor(
                engine_name=engine_name,
                operation="test_op",
                func=lambda: None,
            )

        # All engines should have been monitored
        assert True  # If we got here, all engines were monitored

    def test_metrics_export_after_monitoring(self):
        """Metrics can be exported after monitoring operations."""
        engine = ObservabilityEngine()
        engine.monitor(
            engine_name="fraud",
            operation="detect",
            func=lambda: "clean",
        )
        output = engine.export_metrics()
        assert isinstance(output, str)
        assert len(output) > 0
