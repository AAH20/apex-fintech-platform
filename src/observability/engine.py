"""Observability Engine — logging, metrics, tracing, and alerting.

Provides unified observability for all platform engines using:
- Structured logging with engine context
- Prometheus metrics (counters, histograms, gauges)
- Distributed tracing with spans and trace contexts
- Alert rules with threshold-based firing
- Grafana dashboard generation
- Health checks

References:
    CFA Institute. (2020). "Monitoring and Observability in Financial Systems."
    Prometheus. (2024). "Prometheus Monitoring System."
    Grafana. (2024). "Grafana Dashboard Documentation."
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class Span:
    """A single operation span in a trace."""

    span_id: str
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    parent_id: Optional[str] = None
    engine: Optional[str] = None
    status: Optional[str] = None
    error: Optional[str] = None
    labels: dict[str, str] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Calculate span duration in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    def to_dict(self) -> dict[str, Any]:
        """Serialize span to dictionary."""
        return {
            "span_id": self.span_id,
            "operation_name": self.operation_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "parent_id": self.parent_id,
            "engine": self.engine,
            "status": self.status,
            "error": self.error,
            "labels": self.labels,
        }


@dataclass
class TraceContext:
    """Context for a distributed trace."""

    trace_id: str
    spans: list[Span] = field(default_factory=list)

    def add_span(self, span: Span) -> None:
        """Add a span to this trace context."""
        self.spans.append(span)

    def to_dict(self) -> dict[str, Any]:
        """Serialize trace context to dictionary."""
        return {
            "trace_id": self.trace_id,
            "spans": [s.to_dict() for s in self.spans],
        }


@dataclass
class AlertRule:
    """A rule for firing alerts based on metric thresholds."""

    name: str
    metric: str
    threshold: float
    comparison: str
    severity: str
    labels: dict[str, str] = field(default_factory=dict)

    def evaluate(self, value: float) -> bool:
        """Evaluate if the given value triggers this alert rule."""
        if self.comparison == ">":
            return value > self.threshold
        elif self.comparison == "<":
            return value < self.threshold
        elif self.comparison == "==":
            return value == self.threshold
        elif self.comparison == ">=":
            return value >= self.threshold
        elif self.comparison == "<=":
            return value <= self.threshold
        return False

    def to_dict(self) -> dict[str, Any]:
        """Serialize alert rule to dictionary."""
        return {
            "name": self.name,
            "metric": self.metric,
            "threshold": self.threshold,
            "comparison": self.comparison,
            "severity": self.severity,
            "labels": self.labels,
        }


@dataclass
class Alert:
    """A fired alert instance."""

    rule_name: str
    metric: str
    value: float
    threshold: float
    severity: str
    timestamp: float = field(default_factory=time.time)
    labels: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize alert to dictionary."""
        return {
            "rule_name": self.rule_name,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "severity": self.severity,
            "timestamp": self.timestamp,
            "labels": self.labels,
        }


# ---------------------------------------------------------------------------
# Observability Engine
# ---------------------------------------------------------------------------


class ObservabilityEngine:
    """Unified observability engine for monitoring all platform engines.

    Provides:
    - Structured logging with engine context
    - Prometheus metrics (counters, histograms, gauges)
    - Distributed tracing with spans
    - Alert rule evaluation
    - Grafana dashboard generation
    - Health checks
    """

    def __init__(self, name: str = "observability") -> None:
        """Initialize the observability engine.

        Args:
            name: Name identifier for this engine instance.
        """
        self.name = name
        self.logger = self._setup_logger()
        self.registry = CollectorRegistry()
        self.active_traces: dict[str, TraceContext] = {}
        self.alert_rules: dict[str, AlertRule] = {}
        self._metrics: dict[str, Any] = {}
        self._metric_values: dict[str, float] = {}
        self._spans: dict[str, Span] = {}

    def _setup_logger(self) -> logging.Logger:
        """Set up structured logger with engine context."""
        logger = logging.getLogger(f"apex.{self.name}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------

    def log_info(self, message: str, engine: Optional[str] = None, **kwargs: Any) -> None:
        """Log an info message with optional engine context."""
        self._log(logging.INFO, message, engine, **kwargs)

    def log_warning(self, message: str, engine: Optional[str] = None, **kwargs: Any) -> None:
        """Log a warning message with optional engine context."""
        self._log(logging.WARNING, message, engine, **kwargs)

    def log_error(self, message: str, engine: Optional[str] = None, **kwargs: Any) -> None:
        """Log an error message with optional engine context."""
        self._log(logging.ERROR, message, engine, **kwargs)

    def log_debug(self, message: str, engine: Optional[str] = None, **kwargs: Any) -> None:
        """Log a debug message with optional engine context."""
        self._log(logging.DEBUG, message, engine, **kwargs)

    def _log(self, level: int, message: str, engine: Optional[str] = None, **kwargs: Any) -> None:
        """Internal logging method with structured context."""
        extra = {"engine": engine or "observability"}
        extra.update(kwargs)
        self.logger.log(level, message, extra=extra)

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------

    def _get_or_create_metric(
        self,
        name: str,
        metric_type: str,
        labels: dict[str, str],
    ) -> Any:
        """Get existing metric or create a new one.

        Args:
            name: Metric name.
            metric_type: 'counter', 'histogram', or 'gauge'.
            labels: Label dict.

        Returns:
            The Prometheus metric object.
        """
        label_keys = sorted(labels.keys())
        key = f"{name}:{','.join(label_keys)}"

        if key in self._metrics:
            return self._metrics[key]

        full_name = f"apex_{name}"
        label_names = list(label_keys)

        if metric_type == "counter":
            metric = Counter(
                full_name,
                f"Counter metric: {name}",
                label_names,
                registry=self.registry,
            )
        elif metric_type == "histogram":
            metric = Histogram(
                full_name,
                f"Histogram metric: {name}",
                label_names,
                registry=self.registry,
            )
        elif metric_type == "gauge":
            metric = Gauge(
                full_name,
                f"Gauge metric: {name}",
                label_names,
                registry=self.registry,
            )
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")

        self._metrics[key] = metric
        return metric

    def record_counter(self, name: str, value: float, labels: Optional[dict[str, str]] = None) -> None:
        """Record a counter metric.

        Args:
            name: Metric name.
            value: Value to increment by.
            labels: Optional label dict.
        """
        labels = labels or {}
        metric = self._get_or_create_metric(name, "counter", labels)
        if labels:
            metric.labels(**labels).inc(value)
        else:
            metric.inc(value)
        # Track value for alerting
        val_key = self._value_key(name, labels)
        self._metric_values[val_key] = self._metric_values.get(val_key, 0.0) + value

    def record_histogram(self, name: str, value: float, labels: Optional[dict[str, str]] = None) -> None:
        """Record a histogram metric.

        Args:
            name: Metric name.
            value: Value to observe.
            labels: Optional label dict.
        """
        labels = labels or {}
        metric = self._get_or_create_metric(name, "histogram", labels)
        if labels:
            metric.labels(**labels).observe(value)
        else:
            metric.observe(value)
        # Track latest observed value for alerting
        val_key = self._value_key(name, labels)
        self._metric_values[val_key] = value

    def record_gauge(self, name: str, value: float, labels: Optional[dict[str, str]] = None) -> None:
        """Record a gauge metric.

        Args:
            name: Metric name.
            value: Value to set.
            labels: Optional label dict.
        """
        labels = labels or {}
        metric = self._get_or_create_metric(name, "gauge", labels)
        if labels:
            metric.labels(**labels).set(value)
        else:
            metric.set(value)
        # Track value for alerting
        val_key = self._value_key(name, labels)
        self._metric_values[val_key] = value

    def get_metric_value(self, name: str, labels: Optional[dict[str, str]] = None) -> float:
        """Get the current value of a metric.

        Args:
            name: Metric name.
            labels: Optional label dict.

        Returns:
            Current metric value.
        """
        labels = labels or {}
        val_key = self._value_key(name, labels)
        return self._metric_values.get(val_key, 0.0)

    def _value_key(self, name: str, labels: dict[str, str]) -> str:
        """Generate a unique key for metric value tracking."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    # -----------------------------------------------------------------------
    # Tracing
    # -----------------------------------------------------------------------

    def start_span(
        self,
        operation_name: str,
        engine: Optional[str] = None,
        parent_span: Optional[Span] = None,
        trace_context: Optional[TraceContext] = None,
        labels: Optional[dict[str, str]] = None,
    ) -> Span:
        """Start a new span for an operation.

        Args:
            operation_name: Name of the operation.
            engine: Engine name.
            parent_span: Optional parent span for nested tracing.
            trace_context: Optional trace context to add span to.
            labels: Optional span labels.

        Returns:
            The created Span.
        """
        span = Span(
            span_id=str(uuid.uuid4()),
            operation_name=operation_name,
            start_time=time.time(),
            parent_id=parent_span.span_id if parent_span else None,
            engine=engine,
            labels=labels or {},
        )
        self._spans[span.span_id] = span
        if trace_context:
            trace_context.add_span(span)
        return span

    def end_span(
        self,
        span: Span,
        status: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """End a span and record its duration.

        Args:
            span: The span to end.
            status: Optional status string.
            error: Optional error message.
        """
        span.end_time = time.time()
        span.status = status
        span.error = error

    def create_trace_context(self, trace_id: Optional[str] = None) -> TraceContext:
        """Create a new trace context.

        Args:
            trace_id: Optional trace ID (generated if not provided).

        Returns:
            The created TraceContext.
        """
        if trace_id is None:
            trace_id = str(uuid.uuid4())
        ctx = TraceContext(trace_id=trace_id)
        self.active_traces[trace_id] = ctx
        return ctx

    def get_trace(self, trace_id: str) -> Optional[TraceContext]:
        """Get a trace context by ID.

        Args:
            trace_id: The trace ID to look up.

        Returns:
            TraceContext if found, None otherwise.
        """
        return self.active_traces.get(trace_id)

    # -----------------------------------------------------------------------
    # Alerting
    # -----------------------------------------------------------------------

    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add an alert rule.

        Args:
            rule: The AlertRule to add.
        """
        self.alert_rules[rule.name] = rule

    def remove_alert_rule(self, name: str) -> None:
        """Remove an alert rule by name.

        Args:
            name: Name of the rule to remove.
        """
        self.alert_rules.pop(name, None)

    def check_alerts(self) -> list[Alert]:
        """Check all alert rules against current metric values.

        Returns:
            List of fired Alert instances.
        """
        fired: list[Alert] = []
        for rule in self.alert_rules.values():
            value = self.get_metric_value(rule.metric, rule.labels)
            if rule.evaluate(value):
                alert = Alert(
                    rule_name=rule.name,
                    metric=rule.metric,
                    value=value,
                    threshold=rule.threshold,
                    severity=rule.severity,
                    labels=rule.labels,
                )
                fired.append(alert)
        return fired

    # -----------------------------------------------------------------------
    # Engine Monitoring
    # -----------------------------------------------------------------------

    def monitor(
        self,
        engine_name: str,
        operation: str,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Monitor an engine operation with metrics and tracing.

        Args:
            engine_name: Name of the engine being monitored.
            operation: Name of the operation.
            func: The function to execute.
            *args: Positional args for func.
            **kwargs: Keyword args for func.

        Returns:
            The result of func(*args, **kwargs).
        """
        trace_ctx = self.create_trace_context()
        span = self.start_span(
            operation_name=operation,
            engine=engine_name,
            trace_context=trace_ctx,
        )

        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time

            # Record success metrics
            self.record_counter(
                "engine_operations_total",
                1.0,
                labels={"engine": engine_name, "operation": operation, "status": "success"},
            )
            self.record_histogram(
                "operation_latency",
                duration,
                labels={"engine": engine_name, "operation": operation},
            )

            self.end_span(span, status="success")
            self.log_info(
                f"Operation {operation} on {engine_name} completed in {duration:.4f}s",
                engine=engine_name,
            )
            return result

        except Exception as exc:
            duration = time.time() - start_time

            # Record error metrics
            self.record_counter(
                "engine_operations_total",
                1.0,
                labels={"engine": engine_name, "operation": operation, "status": "error"},
            )
            self.record_counter(
                "engine_errors_total",
                1.0,
                labels={"engine": engine_name, "error_type": type(exc).__name__},
            )
            self.record_histogram(
                "operation_latency",
                duration,
                labels={"engine": engine_name, "operation": operation},
            )

            self.end_span(span, status="error", error=str(exc))
            self.log_error(
                f"Operation {operation} on {engine_name} failed: {exc}",
                engine=engine_name,
            )
            raise

    # -----------------------------------------------------------------------
    # Prometheus Export
    # -----------------------------------------------------------------------

    def export_metrics(self) -> str:
        """Export all metrics in Prometheus text format.

        Returns:
            Prometheus-formatted metrics string.
        """
        return generate_latest(self.registry).decode("utf-8")

    # -----------------------------------------------------------------------
    # Grafana Dashboard
    # -----------------------------------------------------------------------

    def generate_dashboard(self) -> dict[str, Any]:
        """Generate a Grafana dashboard JSON for all monitored engines.

        Returns:
            Grafana dashboard configuration dict.
        """
        engines = [
            "marketmaking",
            "regtech",
            "tokenization",
            "altdata",
            "fraud",
            "portfolio",
            "execution",
            "insurtech",
        ]

        panels = []
        panel_id = 1

        # Operations panel per engine
        for engine in engines:
            panels.append({
                "id": panel_id,
                "title": f"{engine.title()} — Operations",
                "type": "timeseries",
                "targets": [
                    {
                        "expr": f'apex_engine_operations_total{{engine="{engine}"}}',
                        "legendFormat": "{{operation}} — {{status}}",
                    }
                ],
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": (panel_id - 1) * 8},
            })
            panel_id += 1

        # Latency panel per engine
        for engine in engines:
            panels.append({
                "id": panel_id,
                "title": f"{engine.title()} — Latency",
                "type": "timeseries",
                "targets": [
                    {
                        "expr": f'apex_operation_latency{{engine="{engine}"}}',
                        "legendFormat": "{{operation}}",
                    }
                ],
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": (panel_id - 1) * 8},
            })
            panel_id += 1

        # Errors panel
        panels.append({
            "id": panel_id,
            "title": "Engine Errors",
            "type": "timeseries",
            "targets": [
                {
                    "expr": "apex_engine_errors_total",
                    "legendFormat": "{{engine}} — {{error_type}}",
                }
            ],
            "gridPos": {"h": 8, "w": 24, "x": 0, "y": (panel_id - 1) * 8},
        })
        panel_id += 1

        # Health panel
        panels.append({
            "id": panel_id,
            "title": "Engine Health",
            "type": "stat",
            "targets": [
                {
                    "expr": "apex_engine_health",
                    "legendFormat": "{{engine}}",
                }
            ],
            "gridPos": {"h": 8, "w": 24, "x": 0, "y": (panel_id - 1) * 8},
        })

        return {
            "title": "Apex Fintech Platform — Observability",
            "uid": "apex-observability",
            "version": 1,
            "panels": panels,
            "time": {"from": "now-1h", "to": "now"},
            "refresh": "30s",
        }

    # -----------------------------------------------------------------------
    # Health Check
    # -----------------------------------------------------------------------

    def health_check(self) -> dict[str, Any]:
        """Perform a health check on the observability engine.

        Returns:
            Health status dict with status and details.
        """
        alerts = self.check_alerts()
        critical_alerts = [a for a in alerts if a.severity == "critical"]

        status = "unhealthy" if critical_alerts else "healthy"

        return {
            "status": status,
            "engine": self.name,
            "active_traces": len(self.active_traces),
            "alert_rules": len(self.alert_rules),
            "fired_alerts": len(alerts),
            "critical_alerts": len(critical_alerts),
            "timestamp": time.time(),
        }
