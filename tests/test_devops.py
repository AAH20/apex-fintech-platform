"""Tests for DevOps CI/CD automation engine."""
from datetime import datetime, timedelta

import pytest

from src.devops.engine import (
    AlertSeverity,
    DeploymentEnvironment,
    DeploymentStatus,
    DevOpsEngine,
    MetricType,
    PipelineStage,
    PipelineStatus,
)


class TestDevOpsEngineInit:
    """Test engine initialization and basic properties."""

    def test_engine_initializes_empty(self):
        engine = DevOpsEngine()
        assert engine.pipelines == {}
        assert engine.metrics == []
        assert engine.alerts == {}
        assert engine.deployments == {}

    def test_engine_initializes_with_pipelines(self):
        engine = DevOpsEngine()
        pipeline = engine.create_pipeline("test-pipeline", DeploymentEnvironment.STAGING)
        assert len(engine.pipelines) == 1
        assert pipeline.name == "test-pipeline"


class TestPipelineManagement:
    """Test pipeline creation, retrieval, and listing."""

    def test_create_pipeline(self):
        engine = DevOpsEngine()
        pipeline = engine.create_pipeline(
            "ci-pipeline",
            DeploymentEnvironment.PRODUCTION,
            stages=[PipelineStage.BUILD, PipelineStage.TEST, PipelineStage.DEPLOY],
        )
        assert pipeline.id is not None
        assert pipeline.name == "ci-pipeline"
        assert pipeline.environment == DeploymentEnvironment.PRODUCTION
        assert pipeline.status == PipelineStatus.PENDING
        assert len(pipeline.stages) == 3

    def test_create_pipeline_default_stages(self):
        engine = DevOpsEngine()
        pipeline = engine.create_pipeline("default-pipeline", DeploymentEnvironment.DEVELOPMENT)
        assert len(pipeline.stages) == 4
        assert PipelineStage.BUILD in pipeline.stages
        assert PipelineStage.TEST in pipeline.stages
        assert PipelineStage.SECURITY_SCAN in pipeline.stages
        assert PipelineStage.DEPLOY in pipeline.stages

    def test_get_pipeline(self):
        engine = DevOpsEngine()
        created = engine.create_pipeline("my-pipeline", DeploymentEnvironment.STAGING)
        fetched = engine.get_pipeline(created.id)
        assert fetched.id == created.id
        assert fetched.name == "my-pipeline"

    def test_get_pipeline_not_found(self):
        engine = DevOpsEngine()
        with pytest.raises(ValueError, match="Pipeline .* not found"):
            engine.get_pipeline("nonexistent-id")

    def test_list_pipelines(self):
        engine = DevOpsEngine()
        engine.create_pipeline("p1", DeploymentEnvironment.DEVELOPMENT)
        engine.create_pipeline("p2", DeploymentEnvironment.STAGING)
        engine.create_pipeline("p3", DeploymentEnvironment.PRODUCTION)
        all_pipelines = engine.list_pipelines()
        assert len(all_pipelines) == 3

    def test_list_pipelines_filter_by_environment(self):
        engine = DevOpsEngine()
        engine.create_pipeline("p1", DeploymentEnvironment.DEVELOPMENT)
        engine.create_pipeline("p2", DeploymentEnvironment.STAGING)
        engine.create_pipeline("p3", DeploymentEnvironment.STAGING)
        staging = engine.list_pipelines(environment=DeploymentEnvironment.STAGING)
        assert len(staging) == 2
        assert all(p.environment == DeploymentEnvironment.STAGING for p in staging)


class TestPipelineExecution:
    """Test pipeline running and cancellation."""

    def test_run_pipeline(self):
        engine = DevOpsEngine()
        pipeline = engine.create_pipeline("run-me", DeploymentEnvironment.STAGING)
        result = engine.run_pipeline(pipeline.id)
        assert result.status == PipelineStatus.SUCCESS
        assert result.started_at is not None
        assert result.completed_at is not None
        assert len(result.stage_results) == len(pipeline.stages)

    def test_run_pipeline_stages_execute_in_order(self):
        engine = DevOpsEngine()
        pipeline = engine.create_pipeline(
            "ordered",
            DeploymentEnvironment.STAGING,
            stages=[PipelineStage.BUILD, PipelineStage.TEST, PipelineStage.DEPLOY],
        )
        result = engine.run_pipeline(pipeline.id)
        stage_names = [sr.stage for sr in result.stage_results]
        assert stage_names == [PipelineStage.BUILD, PipelineStage.TEST, PipelineStage.DEPLOY]

    def test_run_pipeline_not_found(self):
        engine = DevOpsEngine()
        with pytest.raises(ValueError, match="Pipeline .* not found"):
            engine.run_pipeline("nonexistent-id")

    def test_cancel_pipeline(self):
        engine = DevOpsEngine()
        pipeline = engine.create_pipeline("cancel-me", DeploymentEnvironment.STAGING)
        cancelled = engine.cancel_pipeline(pipeline.id)
        assert cancelled.status == PipelineStatus.CANCELLED

    def test_cancel_pipeline_not_found(self):
        engine = DevOpsEngine()
        with pytest.raises(ValueError, match="Pipeline .* not found"):
            engine.cancel_pipeline("nonexistent-id")


class TestMetrics:
    """Test metric recording and querying."""

    def test_add_metric(self):
        engine = DevOpsEngine()
        metric = engine.add_metric("cpu_usage", 75.5, labels={"host": "web-01"})
        assert metric.name == "cpu_usage"
        assert metric.value == 75.5
        assert metric.labels["host"] == "web-01"
        assert metric.timestamp is not None

    def test_add_metric_with_type(self):
        engine = DevOpsEngine()
        metric = engine.add_metric("memory_usage", 80.0, metric_type=MetricType.MEMORY)
        assert metric.metric_type == MetricType.MEMORY

    def test_get_metrics_all(self):
        engine = DevOpsEngine()
        engine.add_metric("cpu", 50.0)
        engine.add_metric("memory", 60.0)
        engine.add_metric("cpu", 70.0)
        all_metrics = engine.get_metrics()
        assert len(all_metrics) == 3

    def test_get_metrics_filter_by_name(self):
        engine = DevOpsEngine()
        engine.add_metric("cpu", 50.0)
        engine.add_metric("memory", 60.0)
        engine.add_metric("cpu", 70.0)
        cpu_metrics = engine.get_metrics(name="cpu")
        assert len(cpu_metrics) == 2
        assert all(m.name == "cpu" for m in cpu_metrics)

    def test_get_metrics_filter_by_time(self):
        engine = DevOpsEngine()
        now = datetime.now()
        engine.add_metric("cpu", 50.0)
        recent = engine.get_metrics(start=now - timedelta(minutes=5))
        assert len(recent) == 1
        old = engine.get_metrics(end=now - timedelta(minutes=5))
        assert len(old) == 0


class TestAlerts:
    """Test alert creation, acknowledgment, and filtering."""

    def test_create_alert(self):
        engine = DevOpsEngine()
        alert = engine.create_alert("high-cpu", AlertSeverity.WARNING, "CPU above 80%")
        assert alert.id is not None
        assert alert.name == "high-cpu"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.message == "CPU above 80%"
        assert alert.acknowledged is False

    def test_acknowledge_alert(self):
        engine = DevOpsEngine()
        alert = engine.create_alert("disk-full", AlertSeverity.CRITICAL, "Disk 95% full")
        acked = engine.acknowledge_alert(alert.id)
        assert acked.acknowledged is True

    def test_acknowledge_alert_not_found(self):
        engine = DevOpsEngine()
        with pytest.raises(ValueError, match="Alert .* not found"):
            engine.acknowledge_alert("nonexistent-id")

    def test_get_alerts_filter_by_severity(self):
        engine = DevOpsEngine()
        engine.create_alert("a1", AlertSeverity.INFO, "info msg")
        engine.create_alert("a2", AlertSeverity.WARNING, "warning msg")
        engine.create_alert("a3", AlertSeverity.CRITICAL, "critical msg")
        warnings = engine.get_alerts(severity=AlertSeverity.WARNING)
        assert len(warnings) == 1
        assert warnings[0].name == "a2"

    def test_get_alerts_filter_by_acknowledged(self):
        engine = DevOpsEngine()
        a1 = engine.create_alert("a1", AlertSeverity.INFO, "msg1")
        engine.create_alert("a2", AlertSeverity.INFO, "msg2")
        engine.acknowledge_alert(a1.id)
        acked = engine.get_alerts(acknowledged=True)
        unacked = engine.get_alerts(acknowledged=False)
        assert len(acked) == 1
        assert len(unacked) == 1

    def test_check_thresholds_generates_alerts(self):
        engine = DevOpsEngine()
        engine.add_metric("cpu_usage", 95.0)
        engine.add_metric("memory_usage", 50.0)
        alerts = engine.check_thresholds()
        assert len(alerts) >= 1
        cpu_alerts = [a for a in alerts if "cpu" in a.name.lower()]
        assert len(cpu_alerts) >= 1


class TestDeployments:
    """Test deployment and rollback operations."""

    def test_deploy(self):
        engine = DevOpsEngine()
        deployment = engine.deploy("v1.2.3", DeploymentEnvironment.STAGING)
        assert deployment.id is not None
        assert deployment.version == "v1.2.3"
        assert deployment.environment == DeploymentEnvironment.STAGING
        assert deployment.status == DeploymentStatus.SUCCESS
        assert deployment.deployed_at is not None

    def test_deploy_invalid_environment(self):
        engine = DevOpsEngine()
        with pytest.raises(ValueError, match="Invalid environment"):
            engine.deploy("v1.0.0", "invalid-env")

    def test_get_deployments(self):
        engine = DevOpsEngine()
        engine.deploy("v1.0.0", DeploymentEnvironment.STAGING)
        engine.deploy("v1.1.0", DeploymentEnvironment.PRODUCTION)
        all_deployments = engine.get_deployments()
        assert len(all_deployments) == 2

    def test_get_deployments_filter_by_environment(self):
        engine = DevOpsEngine()
        engine.deploy("v1.0.0", DeploymentEnvironment.STAGING)
        engine.deploy("v1.1.0", DeploymentEnvironment.PRODUCTION)
        engine.deploy("v1.2.0", DeploymentEnvironment.PRODUCTION)
        prod = engine.get_deployments(environment=DeploymentEnvironment.PRODUCTION)
        assert len(prod) == 2

    def test_rollback(self):
        engine = DevOpsEngine()
        engine.deploy("v1.0.0", DeploymentEnvironment.STAGING)
        engine.deploy("v1.1.0", DeploymentEnvironment.STAGING)
        rollback = engine.rollback(environment=DeploymentEnvironment.STAGING)
        assert rollback.version == "v1.0.0"
        assert rollback.status == DeploymentStatus.SUCCESS

    def test_rollback_no_previous_deployment(self):
        engine = DevOpsEngine()
        engine.deploy("v1.0.0", DeploymentEnvironment.STAGING)
        with pytest.raises(ValueError, match="No previous deployment to rollback"):
            engine.rollback(environment=DeploymentEnvironment.STAGING)


class TestHealthCheck:
    """Test health check endpoint."""

    def test_health_check_returns_status(self):
        engine = DevOpsEngine()
        health = engine.health_check()
        assert "status" in health
        assert health["status"] == "healthy"
        assert "timestamp" in health
        assert "pipelines" in health
        assert "deployments" in health
        assert "alerts" in health

    def test_health_check_counts(self):
        engine = DevOpsEngine()
        engine.create_pipeline("p1", DeploymentEnvironment.STAGING)
        engine.deploy("v1.0.0", DeploymentEnvironment.STAGING)
        engine.create_alert("a1", AlertSeverity.INFO, "msg")
        health = engine.health_check()
        assert health["pipelines"] == 1
        assert health["deployments"] == 1
        assert health["alerts"] == 1


class TestPipelineReport:
    """Test pipeline report generation."""

    def test_generate_pipeline_report(self):
        engine = DevOpsEngine()
        pipeline = engine.create_pipeline("report-me", DeploymentEnvironment.STAGING)
        engine.run_pipeline(pipeline.id)
        report = engine.generate_pipeline_report(pipeline.id)
        assert report["pipeline_id"] == pipeline.id
        assert report["pipeline_name"] == "report-me"
        assert report["status"] == PipelineStatus.SUCCESS
        assert "stages" in report
        assert "total_duration_seconds" in report

    def test_generate_pipeline_report_not_found(self):
        engine = DevOpsEngine()
        with pytest.raises(ValueError, match="Pipeline .* not found"):
            engine.generate_pipeline_report("nonexistent-id")
