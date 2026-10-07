"""Tests for cloud infrastructure engine — microservices, containerization, orchestration.

Covers:
    - CloudInfrastructureEngine initialization and configuration
    - Microservice deployment (ECS via boto3)
    - Containerization (Docker image build spec)
    - Kubernetes orchestration (Deployment, Service, HPA)
    - Serverless function deployment (Lambda)
    - Infrastructure health checks and monitoring
    - Auto-scaling policies
    - Multi-region deployment
    - Secrets management
    - Network configuration (VPC, subnets, security groups)
    - Cost estimation
    - Cleanup / teardown
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from cloud.infrastructure import (
    CloudInfrastructureEngine,
    ContainerSpec,
    DeploymentStatus,
    HealthStatus,
    MicroserviceConfig,
    NetworkConfig,
    OrchestrationConfig,
    ServerlessFunction,
)


# ---------------------------------------------------------------------------
# Initialization and configuration
# ---------------------------------------------------------------------------


class TestCloudInfrastructureEngineInit:
    """Tests for engine initialization and configuration."""

    def test_engine_initializes_with_default_config(self):
        """Engine initializes with sensible defaults."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        assert engine.region == "us-east-1"
        assert engine.deployed_services == {}
        assert engine.deployed_functions == {}

    def test_engine_initializes_with_custom_config(self):
        """Engine accepts custom configuration."""
        engine = CloudInfrastructureEngine(
            region="eu-west-1",
            cluster_name="prod-cluster",
            namespace="fintech",
        )
        assert engine.region == "eu-west-1"
        assert engine.cluster_name == "prod-cluster"
        assert engine.namespace == "fintech"

    def test_engine_stores_aws_credentials_config(self):
        """Engine stores AWS credentials configuration."""
        engine = CloudInfrastructureEngine(
            region="us-west-2",
            aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
            aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        )
        assert engine.region == "us-west-2"
        assert engine.aws_access_key_id == "AKIAIOSFODNN7EXAMPLE"
        assert engine.aws_secret_access_key == "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"

    def test_engine_validates_region(self):
        """Engine rejects empty region."""
        with pytest.raises(ValueError, match="region must be non-empty"):
            CloudInfrastructureEngine(region="")

    def test_engine_initializes_kubernetes_client(self):
        """Engine initializes Kubernetes client when kubeconfig is available."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        # Kubernetes client may be None if no kubeconfig, but attribute should exist
        assert hasattr(engine, "k8s_client")


# ---------------------------------------------------------------------------
# Microservice deployment
# ---------------------------------------------------------------------------


class TestMicroserviceDeployment:
    """Tests for microservice deployment via ECS."""

    def test_deploy_microservice_creates_ecs_service(self):
        """Deploying a microservice creates an ECS service."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        config = MicroserviceConfig(
            name="payment-service",
            image="fintech/payment-service:latest",
            cpu=256,
            memory=512,
            desired_count=2,
            port=8080,
        )
        with patch.object(engine, "_get_ecs_client") as mock_ecs:
            mock_client = MagicMock()
            mock_ecs.return_value = mock_client
            mock_client.create_service.return_value = {
                "service": {"serviceArn": "arn:aws:ecs:us-east-1:123456789:service/payment-service"}
            }

            result = engine.deploy_microservice(config)

            assert result["name"] == "payment-service"
            assert result["status"] == DeploymentStatus.PENDING
            assert "payment-service" in engine.deployed_services

    def test_deploy_microservice_stores_config(self):
        """Deployed microservice config is stored for later reference."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        config = MicroserviceConfig(
            name="risk-service",
            image="fintech/risk-service:v1.2.0",
            cpu=512,
            memory=1024,
            desired_count=3,
            port=9090,
        )
        with patch.object(engine, "_get_ecs_client") as mock_ecs:
            mock_client = MagicMock()
            mock_ecs.return_value = mock_client
            mock_client.create_service.return_value = {
                "service": {"serviceArn": "arn:aws:ecs:us-east-1:123456789:service/risk-service"}
            }

            engine.deploy_microservice(config)

            stored = engine.deployed_services["risk-service"]
            assert stored["config"].name == "risk-service"
            assert stored["config"].cpu == 512
            assert stored["config"].memory == 1024

    def test_deploy_microservice_validates_config(self):
        """Deploying with invalid config raises error."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        with pytest.raises(ValueError, match="name must be non-empty"):
            engine.deploy_microservice(
                MicroserviceConfig(
                    name="",
                    image="fintech/service:latest",
                    cpu=256,
                    memory=512,
                    desired_count=1,
                    port=8080,
                )
            )

    def test_deploy_microservice_with_environment_variables(self):
        """Microservice deployment includes environment variables."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        config = MicroserviceConfig(
            name="auth-service",
            image="fintech/auth-service:latest",
            cpu=256,
            memory=512,
            desired_count=2,
            port=8080,
            environment={"DB_HOST": "postgres.internal", "DB_PORT": "5432"},
        )
        with patch.object(engine, "_get_ecs_client") as mock_ecs:
            mock_client = MagicMock()
            mock_ecs.return_value = mock_client
            mock_client.create_service.return_value = {
                "service": {"serviceArn": "arn:aws:ecs:us-east-1:123456789:service/auth-service"}
            }

            result = engine.deploy_microservice(config)

            assert result["config"].environment["DB_HOST"] == "postgres.internal"

    def test_deploy_microservice_with_health_check(self):
        """Microservice deployment configures health checks."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        config = MicroserviceConfig(
            name="order-service",
            image="fintech/order-service:latest",
            cpu=512,
            memory=1024,
            desired_count=2,
            port=8080,
            health_check_path="/health",
        )
        with patch.object(engine, "_get_ecs_client") as mock_ecs:
            mock_client = MagicMock()
            mock_ecs.return_value = mock_client
            mock_client.create_service.return_value = {
                "service": {"serviceArn": "arn:aws:ecs:us-east-1:123456789:service/order-service"}
            }

            result = engine.deploy_microservice(config)

            assert result["config"].health_check_path == "/health"


# ---------------------------------------------------------------------------
# Containerization
# ---------------------------------------------------------------------------


class TestContainerization:
    """Tests for container specification and image management."""

    def test_container_spec_creation(self):
        """ContainerSpec captures all required container details."""
        spec = ContainerSpec(
            name="api-gateway",
            image="fintech/api-gateway:v2.0",
            tag="v2.0",
            dockerfile="Dockerfile",
            build_args={"VERSION": "2.0", "ENV": "production"},
            exposed_ports=[8080, 8443],
            resource_limits={"cpu": "500m", "memory": "512Mi"},
        )
        assert spec.name == "api-gateway"
        assert spec.image == "fintech/api-gateway:v2.0"
        assert spec.tag == "v2.0"
        assert spec.exposed_ports == [8080, 8443]
        assert spec.resource_limits["cpu"] == "500m"

    def test_container_spec_generates_dockerfile_content(self):
        """ContainerSpec can generate a Dockerfile."""
        spec = ContainerSpec(
            name="trading-engine",
            image="fintech/trading-engine:v1.0",
            tag="v1.0",
            dockerfile="Dockerfile",
            build_args={},
            exposed_ports=[8080],
            resource_limits={"cpu": "1000m", "memory": "1Gi"},
        )
        dockerfile = spec.generate_dockerfile()
        assert "FROM" in dockerfile
        assert "EXPOSE 8080" in dockerfile
        assert "trading-engine" in dockerfile

    def test_container_spec_generates_docker_compose_service(self):
        """ContainerSpec can generate a docker-compose service entry."""
        spec = ContainerSpec(
            name="portfolio-service",
            image="fintech/portfolio-service:v1.0",
            tag="v1.0",
            dockerfile="Dockerfile",
            build_args={},
            exposed_ports=[8080],
            resource_limits={"cpu": "250m", "memory": "256Mi"},
        )
        compose = spec.to_compose_service()
        assert "portfolio-service" in compose
        assert "8080" in compose
        assert "fintech/portfolio-service:v1.0" in compose

    def test_container_spec_validates_resource_limits(self):
        """ContainerSpec validates resource limits are positive."""
        with pytest.raises(ValueError, match="cpu limit must be positive"):
            ContainerSpec(
                name="bad-service",
                image="fintech/bad:latest",
                tag="latest",
                dockerfile="Dockerfile",
                build_args={},
                exposed_ports=[8080],
                resource_limits={"cpu": "0m", "memory": "512Mi"},
            )

    def test_container_spec_to_k8s_deployment(self):
        """ContainerSpec can generate a Kubernetes Deployment manifest."""
        spec = ContainerSpec(
            name="settlement-service",
            image="fintech/settlement-service:v1.0",
            tag="v1.0",
            dockerfile="Dockerfile",
            build_args={},
            exposed_ports=[8080],
            resource_limits={"cpu": "500m", "memory": "512Mi"},
        )
        manifest = spec.to_k8s_deployment(replicas=3)
        assert "Deployment" in manifest
        assert "settlement-service" in manifest
        assert "replicas: 3" in manifest
        assert "fintech/settlement-service:v1.0" in manifest


# ---------------------------------------------------------------------------
# Kubernetes orchestration
# ---------------------------------------------------------------------------


class TestKubernetesOrchestration:
    """Tests for Kubernetes orchestration features."""

    def test_create_k8s_deployment(self):
        """Engine creates a Kubernetes Deployment."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        config = OrchestrationConfig(
            name="market-data-service",
            image="fintech/market-data:v1.0",
            replicas=3,
            namespace="fintech",
            port=8080,
        )
        with patch.object(engine, "_get_k8s_apps_v1") as mock_apps:
            mock_client = MagicMock()
            mock_apps.return_value = mock_client

            result = engine.create_k8s_deployment(config)

            assert result["name"] == "market-data-service"
            assert result["replicas"] == 3
            assert result["namespace"] == "fintech"

    def test_create_k8s_service(self):
        """Engine creates a Kubernetes Service."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        with patch.object(engine, "_get_k8s_core_v1") as mock_core:
            mock_client = MagicMock()
            mock_core.return_value = mock_client

            result = engine.create_k8s_service(
                name="payment-service",
                port=8080,
                target_port=8080,
                service_type="ClusterIP",
            )

            assert result["name"] == "payment-service"
            assert result["port"] == 8080
            assert result["type"] == "ClusterIP"

    def test_create_k8s_hpa(self):
        """Engine creates a HorizontalPodAutoscaler."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        with patch.object(engine, "_get_k8s_autoscaling_v1") as mock_hpa:
            mock_client = MagicMock()
            mock_hpa.return_value = mock_client

            result = engine.create_k8s_hpa(
                name="trading-engine",
                min_replicas=2,
                max_replicas=10,
                target_cpu_utilization=70,
            )

            assert result["name"] == "trading-engine"
            assert result["min_replicas"] == 2
            assert result["max_replicas"] == 10
            assert result["target_cpu_utilization"] == 70

    def test_k8s_hpa_validates_replica_bounds(self):
        """HPA validates min_replicas <= max_replicas."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        with pytest.raises(ValueError, match="min_replicas must be <= max_replicas"):
            engine.create_k8s_hpa(
                name="bad-hpa",
                min_replicas=10,
                max_replicas=2,
                target_cpu_utilization=70,
            )

    def test_k8s_hpa_validates_cpu_utilization(self):
        """HPA validates CPU utilization is between 1 and 100."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        with pytest.raises(ValueError, match="target_cpu_utilization must be between 1 and 100"):
            engine.create_k8s_hpa(
                name="bad-hpa",
                min_replicas=1,
                max_replicas=5,
                target_cpu_utilization=150,
            )

    def test_scale_k8s_deployment(self):
        """Engine scales a Kubernetes Deployment."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        with patch.object(engine, "_get_k8s_apps_v1") as mock_apps:
            mock_client = MagicMock()
            mock_apps.return_value = mock_client

            result = engine.scale_k8s_deployment(
                name="payment-service",
                namespace="fintech",
                replicas=5,
            )

            assert result["name"] == "payment-service"
            assert result["replicas"] == 5

    def test_delete_k8s_deployment(self):
        """Engine deletes a Kubernetes Deployment."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        with patch.object(engine, "_get_k8s_apps_v1") as mock_apps:
            mock_client = MagicMock()
            mock_apps.return_value = mock_client

            result = engine.delete_k8s_deployment(
                name="old-service",
                namespace="fintech",
            )

            assert result["name"] == "old-service"
            assert result["deleted"] is True

    def test_k8s_deployment_generates_valid_manifest(self):
        """K8s deployment manifest is valid YAML/JSON."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        config = OrchestrationConfig(
            name="compliance-service",
            image="fintech/compliance:v1.0",
            replicas=2,
            namespace="fintech",
            port=8080,
        )
        manifest = engine._generate_deployment_manifest(config)
        assert manifest["kind"] == "Deployment"
        assert manifest["metadata"]["name"] == "compliance-service"
        assert manifest["spec"]["replicas"] == 2
        assert manifest["spec"]["template"]["spec"]["containers"][0]["image"] == "fintech/compliance:v1.0"


# ---------------------------------------------------------------------------
# Serverless functions
# ---------------------------------------------------------------------------


class TestServerlessFunctions:
    """Tests for serverless function deployment via Lambda."""

    def test_deploy_serverless_function(self):
        """Engine deploys a Lambda function."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        func = ServerlessFunction(
            name="price-alert-handler",
            runtime="python3.11",
            handler="handler.lambda_handler",
            code_bucket="fintech-lambda-deployments",
            code_key="price-alert-handler.zip",
            memory=256,
            timeout=30,
            environment={"ALERT_TOPIC": "arn:aws:sns:us-east-1:123456789:price-alerts"},
        )
        with patch.object(engine, "_get_lambda_client") as mock_lambda:
            mock_client = MagicMock()
            mock_lambda.return_value = mock_client
            mock_client.create_function.return_value = {
                "FunctionArn": "arn:aws:lambda:us-east-1:123456789:function:price-alert-handler"
            }

            result = engine.deploy_serverless_function(func)

            assert result["name"] == "price-alert-handler"
            assert result["runtime"] == "python3.11"
            assert "price-alert-handler" in engine.deployed_functions

    def test_serverless_function_validates_runtime(self):
        """Serverless function validates supported runtimes."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        with pytest.raises(ValueError, match="unsupported runtime"):
            engine.deploy_serverless_function(
                ServerlessFunction(
                    name="bad-func",
                    runtime="python2.7",
                    handler="handler.main",
                    code_bucket="bucket",
                    code_key="code.zip",
                    memory=128,
                    timeout=10,
                )
            )

    def test_serverless_function_validates_memory(self):
        """Serverless function validates memory is within Lambda limits."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        with pytest.raises(ValueError, match="memory must be between 128 and 10240"):
            engine.deploy_serverless_function(
                ServerlessFunction(
                    name="bad-func",
                    runtime="python3.11",
                    handler="handler.main",
                    code_bucket="bucket",
                    code_key="code.zip",
                    memory=64,
                    timeout=10,
                )
            )

    def test_serverless_function_validates_timeout(self):
        """Serverless function validates timeout is within Lambda limits."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        with pytest.raises(ValueError, match="timeout must be between 1 and 900"):
            engine.deploy_serverless_function(
                ServerlessFunction(
                    name="bad-func",
                    runtime="python3.11",
                    handler="handler.main",
                    code_bucket="bucket",
                    code_key="code.zip",
                    memory=128,
                    timeout=1000,
                )
            )

    def test_update_serverless_function_code(self):
        """Engine updates Lambda function code."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        with patch.object(engine, "_get_lambda_client") as mock_lambda:
            mock_client = MagicMock()
            mock_lambda.return_value = mock_client
            mock_client.update_function_code.return_value = {
                "FunctionArn": "arn:aws:lambda:us-east-1:123456789:function:my-func"
            }

            result = engine.update_serverless_function_code(
                name="my-func",
                code_bucket="fintech-lambda-deployments",
                code_key="my-func-v2.zip",
            )

            assert result["name"] == "my-func"
            assert result["updated"] is True

    def test_delete_serverless_function(self):
        """Engine deletes a Lambda function."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        with patch.object(engine, "_get_lambda_client") as mock_lambda:
            mock_client = MagicMock()
            mock_lambda.return_value = mock_client

            result = engine.delete_serverless_function(name="old-func")

            assert result["name"] == "old-func"
            assert result["deleted"] is True


# ---------------------------------------------------------------------------
# Health checks and monitoring
# ---------------------------------------------------------------------------


class TestHealthAndMonitoring:
    """Tests for health checks and monitoring."""

    def test_health_check_returns_status(self):
        """Health check returns service health status."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        engine.deployed_services["test-service"] = {
            "config": MicroserviceConfig(
                name="test-service",
                image="fintech/test:latest",
                cpu=256,
                memory=512,
                desired_count=1,
                port=8080,
            ),
            "status": DeploymentStatus.RUNNING,
        }
        with patch.object(engine, "_get_ecs_client") as mock_ecs:
            mock_client = MagicMock()
            mock_ecs.return_value = mock_client
            mock_client.describe_services.return_value = {
                "services": [
                    {
                        "serviceName": "test-service",
                        "status": "ACTIVE",
                        "runningCount": 1,
                        "desiredCount": 1,
                        "healthStatus": "HEALTHY",
                    }
                ]
            }

            health = engine.health_check("test-service")

            assert health.service_name == "test-service"
            assert health.status == HealthStatus.HEALTHY

    def test_health_check_unhealthy_service(self):
        """Health check detects unhealthy service."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        engine.deployed_services["failing-service"] = {
            "config": MicroserviceConfig(
                name="failing-service",
                image="fintech/failing:latest",
                cpu=256,
                memory=512,
                desired_count=2,
                port=8080,
            ),
            "status": DeploymentStatus.RUNNING,
        }
        with patch.object(engine, "_get_ecs_client") as mock_ecs:
            mock_client = MagicMock()
            mock_ecs.return_value = mock_client
            mock_client.describe_services.return_value = {
                "services": [
                    {
                        "serviceName": "failing-service",
                        "status": "ACTIVE",
                        "runningCount": 0,
                        "desiredCount": 2,
                        "healthStatus": "UNHEALTHY",
                    }
                ]
            }

            health = engine.health_check("failing-service")

            assert health.status == HealthStatus.UNHEALTHY

    def test_health_check_unknown_service(self):
        """Health check for unknown service returns UNKNOWN."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        health = engine.health_check("nonexistent-service")
        assert health.status == HealthStatus.UNKNOWN

    def test_get_service_metrics(self):
        """Engine retrieves CloudWatch metrics for a service."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        with patch.object(engine, "_get_cloudwatch_client") as mock_cw:
            mock_client = MagicMock()
            mock_cw.return_value = mock_client
            mock_client.get_metric_statistics.return_value = {
                "Datapoints": [
                    {"Timestamp": "2024-01-01T00:00:00Z", "Average": 45.2},
                    {"Timestamp": "2024-01-01T00:01:00Z", "Average": 52.1},
                ]
            }

            metrics = engine.get_service_metrics(
                service_name="payment-service",
                metric_name="CPUUtilization",
                period=300,
            )

            assert "cpu_utilization" in metrics
            assert len(metrics["cpu_utilization"]) == 2

    def test_setup_cloudwatch_alarms(self):
        """Engine creates CloudWatch alarms for a service."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        with patch.object(engine, "_get_cloudwatch_client") as mock_cw:
            mock_client = MagicMock()
            mock_cw.return_value = mock_client

            alarms = engine.setup_cloudwatch_alarms(
                service_name="payment-service",
                cpu_threshold=80.0,
                memory_threshold=85.0,
            )

            assert len(alarms) >= 2
            assert any("cpu" in a["name"].lower() for a in alarms)
            assert any("memory" in a["name"].lower() for a in alarms)


# ---------------------------------------------------------------------------
# Auto-scaling
# ---------------------------------------------------------------------------


class TestAutoScaling:
    """Tests for auto-scaling policies."""

    def test_configure_ecs_autoscaling(self):
        """Engine configures ECS service auto-scaling."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        with patch.object(engine, "_get_application_autoscaling_client") as mock_as:
            mock_client = MagicMock()
            mock_as.return_value = mock_client
            mock_client.register_scalable_target.return_value = {}
            mock_client.put_scaling_policy.return_value = {
                "PolicyARN": "arn:aws:autoscaling:us-east-1:123456789:scalingPolicy:abc"
            }

            result = engine.configure_ecs_autoscaling(
                service_name="payment-service",
                min_capacity=2,
                max_capacity=10,
                target_cpu_utilization=70.0,
            )

            assert result["service_name"] == "payment-service"
            assert result["min_capacity"] == 2
            assert result["max_capacity"] == 10

    def test_configure_autoscaling_validates_capacity(self):
        """Auto-scaling validates min <= max capacity."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        with pytest.raises(ValueError, match="min_capacity must be <= max_capacity"):
            engine.configure_ecs_autoscaling(
                service_name="bad-service",
                min_capacity=10,
                max_capacity=2,
                target_cpu_utilization=70.0,
            )

    def test_configure_autoscaling_validates_cpu_target(self):
        """Auto-scaling validates CPU target is reasonable."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        with pytest.raises(ValueError, match="target_cpu_utilization must be between 1 and 100"):
            engine.configure_ecs_autoscaling(
                service_name="bad-service",
                min_capacity=1,
                max_capacity=5,
                target_cpu_utilization=0,
            )


# ---------------------------------------------------------------------------
# Multi-region deployment
# ---------------------------------------------------------------------------


class TestMultiRegionDeployment:
    """Tests for multi-region deployment."""

    def test_deploy_to_multiple_regions(self):
        """Engine deploys services to multiple regions."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        config = MicroserviceConfig(
            name="global-service",
            image="fintech/global-service:v1.0",
            cpu=256,
            memory=512,
            desired_count=2,
            port=8080,
        )
        regions = ["us-east-1", "eu-west-1", "ap-southeast-1"]
        with patch.object(engine, "_get_ecs_client") as mock_ecs:
            mock_client = MagicMock()
            mock_ecs.return_value = mock_client
            mock_client.create_service.return_value = {
                "service": {"serviceArn": "arn:aws:ecs:us-east-1:123456789:service/global-service"}
            }

            results = engine.deploy_to_regions(config, regions)

            assert len(results) == 3
            assert all(r["status"] == DeploymentStatus.PENDING for r in results)

    def test_deploy_to_regions_validates_regions(self):
        """Multi-region deployment validates region list."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        config = MicroserviceConfig(
            name="global-service",
            image="fintech/global-service:v1.0",
            cpu=256,
            memory=512,
            desired_count=1,
            port=8080,
        )
        with pytest.raises(ValueError, match="regions must be non-empty"):
            engine.deploy_to_regions(config, [])


# ---------------------------------------------------------------------------
# Secrets management
# ---------------------------------------------------------------------------


class TestSecretsManagement:
    """Tests for secrets management."""

    def test_store_secret(self):
        """Engine stores a secret in AWS Secrets Manager."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        with patch.object(engine, "_get_secrets_manager_client") as mock_sm:
            mock_client = MagicMock()
            mock_sm.return_value = mock_client
            mock_client.create_secret.return_value = {
                "ARN": "arn:aws:secretsmanager:us-east-1:123456789:secret:db-password"
            }

            result = engine.store_secret(
                name="db-password",
                value="super-secret-password",
            )

            assert result["name"] == "db-password"
            assert "arn" in result

    def test_retrieve_secret(self):
        """Engine retrieves a secret from AWS Secrets Manager."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        with patch.object(engine, "_get_secrets_manager_client") as mock_sm:
            mock_client = MagicMock()
            mock_sm.return_value = mock_client
            mock_client.get_secret_value.return_value = {
                "SecretString": "super-secret-password"
            }

            result = engine.retrieve_secret(name="db-password")

            assert result == "super-secret-password"

    def test_store_secret_validates_name(self):
        """Storing a secret with empty name raises error."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        with pytest.raises(ValueError, match="name must be non-empty"):
            engine.store_secret(name="", value="secret")


# ---------------------------------------------------------------------------
# Network configuration
# ---------------------------------------------------------------------------


class TestNetworkConfiguration:
    """Tests for network configuration (VPC, subnets, security groups)."""

    def test_create_vpc(self):
        """Engine creates a VPC."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        with patch.object(engine, "_get_ec2_client") as mock_ec2:
            mock_client = MagicMock()
            mock_ec2.return_value = mock_client
            mock_client.create_vpc.return_value = {
                "Vpc": {"VpcId": "vpc-12345", "CidrBlock": "10.0.0.0/16"}
            }

            result = engine.create_vpc(cidr_block="10.0.0.0/16")

            assert result["vpc_id"] == "vpc-12345"
            assert result["cidr_block"] == "10.0.0.0/16"

    def test_create_subnet(self):
        """Engine creates a subnet."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        with patch.object(engine, "_get_ec2_client") as mock_ec2:
            mock_client = MagicMock()
            mock_ec2.return_value = mock_client
            mock_client.create_subnet.return_value = {
                "Subnet": {"SubnetId": "subnet-12345", "CidrBlock": "10.0.1.0/24"}
            }

            result = engine.create_subnet(
                vpc_id="vpc-12345",
                cidr_block="10.0.1.0/24",
                availability_zone="us-east-1a",
            )

            assert result["subnet_id"] == "subnet-12345"
            assert result["vpc_id"] == "vpc-12345"

    def test_create_security_group(self):
        """Engine creates a security group."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        with patch.object(engine, "_get_ec2_client") as mock_ec2:
            mock_client = MagicMock()
            mock_ec2.return_value = mock_client
            mock_client.create_security_group.return_value = {"GroupId": "sg-12345"}

            result = engine.create_security_group(
                name="payment-sg",
                description="Security group for payment service",
                vpc_id="vpc-12345",
            )

            assert result["group_id"] == "sg-12345"
            assert result["name"] == "payment-sg"

    def test_configure_network(self):
        """Engine configures complete network stack."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        config = NetworkConfig(
            vpc_cidr="10.0.0.0/16",
            subnet_cidrs=["10.0.1.0/24", "10.0.2.0/24"],
            availability_zones=["us-east-1a", "us-east-1b"],
        )
        with patch.object(engine, "_get_ec2_client") as mock_ec2:
            mock_client = MagicMock()
            mock_ec2.return_value = mock_client
            mock_client.create_vpc.return_value = {
                "Vpc": {"VpcId": "vpc-12345", "CidrBlock": "10.0.0.0/16"}
            }
            mock_client.create_subnet.return_value = {
                "Subnet": {"SubnetId": "subnet-12345", "CidrBlock": "10.0.1.0/24"}
            }

            result = engine.configure_network(config)

            assert result["vpc_id"] == "vpc-12345"
            assert len(result["subnets"]) == 2


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------


class TestCostEstimation:
    """Tests for infrastructure cost estimation."""

    def test_estimate_monthly_cost(self):
        """Engine estimates monthly infrastructure cost."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        engine.deployed_services["svc1"] = {
            "config": MicroserviceConfig(
                name="svc1",
                image="fintech/svc1:latest",
                cpu=256,
                memory=512,
                desired_count=2,
                port=8080,
            ),
            "status": DeploymentStatus.RUNNING,
        }
        engine.deployed_services["svc2"] = {
            "config": MicroserviceConfig(
                name="svc2",
                image="fintech/svc2:latest",
                cpu=512,
                memory=1024,
                desired_count=3,
                port=8080,
            ),
            "status": DeploymentStatus.RUNNING,
        }

        cost = engine.estimate_monthly_cost()

        assert cost > 0
        assert isinstance(cost, float)

    def test_estimate_cost_empty_infrastructure(self):
        """Cost estimation with no services returns zero."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        cost = engine.estimate_monthly_cost()
        assert cost == 0.0

    def test_estimate_cost_scales_with_service_count(self):
        """Cost increases with more services."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        for i in range(3):
            engine.deployed_services[f"svc{i}"] = {
                "config": MicroserviceConfig(
                    name=f"svc{i}",
                    image=f"fintech/svc{i}:latest",
                    cpu=256,
                    memory=512,
                    desired_count=1,
                    port=8080,
                ),
                "status": DeploymentStatus.RUNNING,
            }

        cost_3 = engine.estimate_monthly_cost()

        engine.deployed_services["svc3"] = {
            "config": MicroserviceConfig(
                name="svc3",
                image="fintech/svc3:latest",
                cpu=256,
                memory=512,
                desired_count=1,
                port=8080,
            ),
            "status": DeploymentStatus.RUNNING,
        }

        cost_4 = engine.estimate_monthly_cost()
        assert cost_4 > cost_3


# ---------------------------------------------------------------------------
# Cleanup / teardown
# ---------------------------------------------------------------------------


class TestCleanup:
    """Tests for infrastructure cleanup and teardown."""

    def test_cleanup_all_services(self):
        """Engine cleans up all deployed services."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        engine.deployed_services["svc1"] = {
            "config": MicroserviceConfig(
                name="svc1",
                image="fintech/svc1:latest",
                cpu=256,
                memory=512,
                desired_count=1,
                port=8080,
            ),
            "status": DeploymentStatus.RUNNING,
        }
        engine.deployed_functions["func1"] = {
            "name": "func1",
            "runtime": "python3.11",
        }
        with patch.object(engine, "_get_ecs_client") as mock_ecs:
            mock_ecs_client = MagicMock()
            mock_ecs.return_value = mock_ecs_client
            with patch.object(engine, "_get_lambda_client") as mock_lambda:
                mock_lambda_client = MagicMock()
                mock_lambda.return_value = mock_lambda_client

                result = engine.cleanup_all()

                assert result["services_deleted"] == 1
                assert result["functions_deleted"] == 1
                assert len(engine.deployed_services) == 0
                assert len(engine.deployed_functions) == 0

    def test_cleanup_empty_infrastructure(self):
        """Cleanup with no deployed resources succeeds."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        result = engine.cleanup_all()
        assert result["services_deleted"] == 0
        assert result["functions_deleted"] == 0

    def test_delete_microservice(self):
        """Engine deletes a specific microservice."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        engine.deployed_services["svc1"] = {
            "config": MicroserviceConfig(
                name="svc1",
                image="fintech/svc1:latest",
                cpu=256,
                memory=512,
                desired_count=1,
                port=8080,
            ),
            "status": DeploymentStatus.RUNNING,
        }
        with patch.object(engine, "_get_ecs_client") as mock_ecs:
            mock_client = MagicMock()
            mock_ecs.return_value = mock_client

            result = engine.delete_microservice("svc1")

            assert result["name"] == "svc1"
            assert result["deleted"] is True
            assert "svc1" not in engine.deployed_services

    def test_delete_unknown_microservice(self):
        """Deleting unknown microservice raises error."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        with pytest.raises(KeyError):
            engine.delete_microservice("nonexistent")


# ---------------------------------------------------------------------------
# Integration-style tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration-style tests combining multiple features."""

    def test_full_deployment_workflow(self):
        """Full workflow: deploy service, check health, scale, cleanup."""
        engine = CloudInfrastructureEngine(region="us-east-1")

        # Deploy
        config = MicroserviceConfig(
            name="trading-api",
            image="fintech/trading-api:v1.0",
            cpu=512,
            memory=1024,
            desired_count=2,
            port=8080,
            environment={"DB_HOST": "postgres.internal"},
        )
        with patch.object(engine, "_get_ecs_client") as mock_ecs:
            mock_client = MagicMock()
            mock_ecs.return_value = mock_client
            mock_client.create_service.return_value = {
                "service": {"serviceArn": "arn:aws:ecs:us-east-1:123456789:service/trading-api"}
            }
            mock_client.describe_services.return_value = {
                "services": [
                    {
                        "serviceName": "trading-api",
                        "status": "ACTIVE",
                        "runningCount": 2,
                        "desiredCount": 2,
                        "healthStatus": "HEALTHY",
                    }
                ]
            }

            deploy_result = engine.deploy_microservice(config)
            assert deploy_result["name"] == "trading-api"

            health = engine.health_check("trading-api")
            assert health.status == HealthStatus.HEALTHY

            # Scale
            mock_client.update_service.return_value = {
                "service": {"serviceArn": "arn:aws:ecs:us-east-1:123456789:service/trading-api"}
            }
            scale_result = engine.scale_service("trading-api", desired_count=5)
            assert scale_result["desired_count"] == 5

            # Cleanup
            mock_client.delete_service.return_value = {
                "service": {"serviceArn": "arn:aws:ecs:us-east-1:123456789:service/trading-api"}
            }
            cleanup_result = engine.delete_microservice("trading-api")
            assert cleanup_result["deleted"] is True

    def test_multi_service_deployment(self):
        """Deploy multiple services and verify isolation."""
        engine = CloudInfrastructureEngine(region="us-east-1")
        services = [
            MicroserviceConfig(
                name=f"service-{i}",
                image=f"fintech/service-{i}:v1.0",
                cpu=256,
                memory=512,
                desired_count=1,
                port=8080 + i,
            )
            for i in range(3)
        ]
        with patch.object(engine, "_get_ecs_client") as mock_ecs:
            mock_client = MagicMock()
            mock_ecs.return_value = mock_client
            mock_client.create_service.return_value = {
                "service": {"serviceArn": "arn:aws:ecs:us-east-1:123456789:service/test"}
            }

            for svc in services:
                engine.deploy_microservice(svc)

            assert len(engine.deployed_services) == 3
            for i in range(3):
                assert f"service-{i}" in engine.deployed_services
