"""DevOps CI/CD automation engine for pipeline orchestration, monitoring, and deployment."""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class PipelineStage(str, Enum):
    """Stages in a CI/CD pipeline."""

    BUILD = "build"
    TEST = "test"
    SECURITY_SCAN = "security_scan"
    DEPLOY = "deploy"


class PipelineStatus(str, Enum):
    """Pipeline execution status."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DeploymentEnvironment(str, Enum):
    """Deployment target environments."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class DeploymentStatus(str, Enum):
    """Deployment status."""

    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class MetricType(str, Enum):
    """Types of metrics."""

    CPU = "cpu"
    MEMORY = "memory"
    LATENCY = "latency"
    CUSTOM = "custom"


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class StageResult(BaseModel):
    """Result of a single pipeline stage execution."""

    stage: PipelineStage
    status: PipelineStatus
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: datetime | None = None
    duration_seconds: float = 0.0


class Pipeline(BaseModel):
    """A CI/CD pipeline definition."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    environment: DeploymentEnvironment
    stages: list[PipelineStage] = Field(default_factory=list)
    status: PipelineStatus = PipelineStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    stage_results: list[StageResult] = Field(default_factory=list)


class Metric(BaseModel):
    """A monitoring metric data point."""

    name: str
    value: float
    metric_type: MetricType = MetricType.CUSTOM
    labels: dict[str, str] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class Alert(BaseModel):
    """A monitoring alert."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    severity: AlertSeverity
    message: str
    acknowledged: bool = False
    created_at: datetime = Field(default_factory=datetime.now)


class Deployment(BaseModel):
    """A deployment record."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    version: str
    environment: DeploymentEnvironment
    status: DeploymentStatus = DeploymentStatus.IN_PROGRESS
    deployed_at: datetime = Field(default_factory=datetime.now)


class DevOpsEngine:
    """Main DevOps automation engine.

    Manages CI/CD pipelines, monitoring metrics, alerts, and deployments.
    """

    def __init__(self):
        self.pipelines: dict[str, Pipeline] = {}
        self.metrics: list[Metric] = []
        self.alerts: dict[str, Alert] = {}
        self.deployments: dict[str, Deployment] = {}
        self._deployment_history: dict[str, list[str]] = {}

    # ── Pipeline Management ──────────────────────────────────────────

    def create_pipeline(
        self,
        name: str,
        environment: DeploymentEnvironment,
        stages: list[PipelineStage] | None = None,
    ) -> Pipeline:
        """Create a new CI/CD pipeline."""
        if stages is None:
            stages = [
                PipelineStage.BUILD,
                PipelineStage.TEST,
                PipelineStage.SECURITY_SCAN,
                PipelineStage.DEPLOY,
            ]
        pipeline = Pipeline(
            name=name,
            environment=environment,
            stages=stages,
        )
        self.pipelines[pipeline.id] = pipeline
        return pipeline

    def get_pipeline(self, pipeline_id: str) -> Pipeline:
        """Retrieve a pipeline by ID."""
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        return self.pipelines[pipeline_id]

    def list_pipelines(
        self, environment: DeploymentEnvironment | None = None
    ) -> list[Pipeline]:
        """List all pipelines, optionally filtered by environment."""
        pipelines = list(self.pipelines.values())
        if environment is not None:
            pipelines = [p for p in pipelines if p.environment == environment]
        return pipelines

    def run_pipeline(self, pipeline_id: str) -> Pipeline:
        """Execute a pipeline, running all stages in order."""
        pipeline = self.get_pipeline(pipeline_id)
        pipeline.status = PipelineStatus.RUNNING
        pipeline.started_at = datetime.now()
        pipeline.stage_results = []

        for stage in pipeline.stages:
            stage_result = StageResult(
                stage=stage,
                status=PipelineStatus.SUCCESS,
                started_at=datetime.now(),
                completed_at=datetime.now(),
                duration_seconds=0.1,
            )
            pipeline.stage_results.append(stage_result)

        pipeline.status = PipelineStatus.SUCCESS
        pipeline.completed_at = datetime.now()
        return pipeline

    def cancel_pipeline(self, pipeline_id: str) -> Pipeline:
        """Cancel a running pipeline."""
        pipeline = self.get_pipeline(pipeline_id)
        pipeline.status = PipelineStatus.CANCELLED
        return pipeline

    # ── Metrics ─────────────────────────────────────────────────────

    def add_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.CUSTOM,
        labels: dict[str, str] | None = None,
    ) -> Metric:
        """Record a monitoring metric."""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            labels=labels or {},
        )
        self.metrics.append(metric)
        return metric

    def get_metrics(
        self,
        name: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Metric]:
        """Query metrics with optional filters."""
        results = self.metrics
        if name is not None:
            results = [m for m in results if m.name == name]
        if start is not None:
            results = [m for m in results if m.timestamp >= start]
        if end is not None:
            results = [m for m in results if m.timestamp <= end]
        return results

    # ── Alerts ──────────────────────────────────────────────────────

    def create_alert(
        self,
        name: str,
        severity: AlertSeverity,
        message: str,
    ) -> Alert:
        """Create a new alert."""
        alert = Alert(
            name=name,
            severity=severity,
            message=message,
        )
        self.alerts[alert.id] = alert
        return alert

    def acknowledge_alert(self, alert_id: str) -> Alert:
        """Acknowledge an alert."""
        if alert_id not in self.alerts:
            raise ValueError(f"Alert {alert_id} not found")
        self.alerts[alert_id].acknowledged = True
        return self.alerts[alert_id]

    def get_alerts(
        self,
        severity: AlertSeverity | None = None,
        acknowledged: bool | None = None,
    ) -> list[Alert]:
        """Get alerts with optional filters."""
        results = list(self.alerts.values())
        if severity is not None:
            results = [a for a in results if a.severity == severity]
        if acknowledged is not None:
            results = [a for a in results if a.acknowledged == acknowledged]
        return results

    def check_thresholds(self) -> list[Alert]:
        """Check metrics against thresholds and generate alerts."""
        new_alerts: list[Alert] = []
        for metric in self.metrics:
            if metric.name == "cpu_usage" and metric.value > 90:
                alert = self.create_alert(
                    name="high-cpu-usage",
                    severity=AlertSeverity.CRITICAL,
                    message=f"CPU usage {metric.value}% exceeds threshold of 90%",
                )
                new_alerts.append(alert)
            elif metric.name == "cpu_usage" and metric.value > 80:
                alert = self.create_alert(
                    name="elevated-cpu-usage",
                    severity=AlertSeverity.WARNING,
                    message=f"CPU usage {metric.value}% exceeds threshold of 80%",
                )
                new_alerts.append(alert)
            elif metric.name == "memory_usage" and metric.value > 90:
                alert = self.create_alert(
                    name="high-memory-usage",
                    severity=AlertSeverity.CRITICAL,
                    message=f"Memory usage {metric.value}% exceeds threshold of 90%",
                )
                new_alerts.append(alert)
            elif metric.name == "memory_usage" and metric.value > 80:
                alert = self.create_alert(
                    name="elevated-memory-usage",
                    severity=AlertSeverity.WARNING,
                    message=f"Memory usage {metric.value}% exceeds threshold of 80%",
                )
                new_alerts.append(alert)
        return new_alerts

    # ── Deployments ─────────────────────────────────────────────────

    def deploy(self, version: str, environment: str) -> Deployment:
        """Deploy a version to an environment."""
        try:
            env = DeploymentEnvironment(environment)
        except ValueError:
            raise ValueError(f"Invalid environment: {environment}")

        deployment = Deployment(
            version=version,
            environment=env,
            status=DeploymentStatus.SUCCESS,
        )
        self.deployments[deployment.id] = deployment

        if env.value not in self._deployment_history:
            self._deployment_history[env.value] = []
        self._deployment_history[env.value].append(deployment.id)

        return deployment

    def get_deployments(
        self, environment: DeploymentEnvironment | None = None
    ) -> list[Deployment]:
        """Get deployments, optionally filtered by environment."""
        results = list(self.deployments.values())
        if environment is not None:
            results = [d for d in results if d.environment == environment]
        return results

    def rollback(self, environment: DeploymentEnvironment) -> Deployment:
        """Rollback to the previous deployment in an environment."""
        env_key = environment.value
        history = self._deployment_history.get(env_key, [])

        if len(history) < 2:
            raise ValueError(f"No previous deployment to rollback in {env_key}")

        # Get the second-to-last deployment (the one before the latest)
        previous_id = history[-2]
        previous = self.deployments[previous_id]

        # Mark current as rolled back
        current_id = history[-1]
        self.deployments[current_id].status = DeploymentStatus.ROLLED_BACK

        # Create a new deployment record for the rollback
        rollback_deployment = Deployment(
            version=previous.version,
            environment=environment,
            status=DeploymentStatus.SUCCESS,
        )
        self.deployments[rollback_deployment.id] = rollback_deployment
        self._deployment_history[env_key].append(rollback_deployment.id)

        return rollback_deployment

    # ── Health & Reporting ──────────────────────────────────────────

    def health_check(self) -> dict[str, Any]:
        """Return system health status."""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "pipelines": len(self.pipelines),
            "deployments": len(self.deployments),
            "alerts": len(self.alerts),
        }

    def generate_pipeline_report(self, pipeline_id: str) -> dict[str, Any]:
        """Generate a detailed report for a pipeline execution."""
        pipeline = self.get_pipeline(pipeline_id)

        total_duration = sum(
            sr.duration_seconds for sr in pipeline.stage_results
        )

        return {
            "pipeline_id": pipeline.id,
            "pipeline_name": pipeline.name,
            "environment": pipeline.environment.value,
            "status": pipeline.status.value,
            "created_at": pipeline.created_at.isoformat(),
            "stages": [
                {
                    "stage": sr.stage.value,
                    "status": sr.status.value,
                    "duration_seconds": sr.duration_seconds,
                }
                for sr in pipeline.stage_results
            ],
            "total_duration_seconds": total_duration,
        }
