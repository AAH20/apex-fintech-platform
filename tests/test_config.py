"""Tests for the Configuration Engine.

Covers configuration management, feature flags, and secrets management
using pydantic-settings for validation.
"""
import pytest

from pydantic import ValidationError

from src.config.engine import (
    ConfigurationEngine,
    AppConfig,
)


# ---------------------------------------------------------------------------
# Initialization & Defaults
# ---------------------------------------------------------------------------


class TestConfigurationEngineInit:
    """Test engine initialization and default values."""

    def test_engine_initializes_with_defaults(self):
        """Engine initializes with sensible defaults."""
        engine = ConfigurationEngine()
        assert engine is not None
        assert engine.config is not None

    def test_engine_initializes_with_dict(self):
        """Engine initializes from a dictionary."""
        engine = ConfigurationEngine.from_dict({
            "environment": "testing",
            "debug": True,
        })
        assert engine.config.environment == "testing"
        assert engine.config.debug is True

    def test_engine_initializes_with_env_vars(self, monkeypatch):
        """Engine picks up environment variables."""
        monkeypatch.setenv("APEX_ENVIRONMENT", "staging")
        monkeypatch.setenv("APEX_DEBUG", "true")
        engine = ConfigurationEngine.from_env()
        assert engine.config.environment == "staging"
        assert engine.config.debug is True

    def test_engine_initializes_from_dotenv(self, tmp_path):
        """Engine loads from a .env file."""
        env_file = tmp_path / ".env"
        env_file.write_text("APEX_ENVIRONMENT=production\nAPEX_DEBUG=false\n")
        engine = ConfigurationEngine.from_dotenv(str(env_file))
        assert engine.config.environment == "production"
        assert engine.config.debug is False


# ---------------------------------------------------------------------------
# Configuration Access
# ---------------------------------------------------------------------------


class TestConfigurationAccess:
    """Test getting and setting configuration values."""

    def test_get_config_value(self):
        """Get a configuration value by key."""
        engine = ConfigurationEngine.from_dict({"environment": "development"})
        assert engine.get("environment") == "development"

    def test_get_config_value_with_default(self):
        """Get a config value with a fallback default."""
        engine = ConfigurationEngine()
        assert engine.get("nonexistent_key", default="fallback") == "fallback"

    def test_get_config_value_missing_raises(self):
        """Get a missing config value raises KeyError."""
        engine = ConfigurationEngine()
        with pytest.raises(KeyError):
            engine.get("nonexistent_key")

    def test_set_config_value(self):
        """Set a configuration value."""
        engine = ConfigurationEngine()
        engine.set("custom_key", "custom_value")
        assert engine.get("custom_key") == "custom_value"

    def test_set_config_value_overwrites(self):
        """Setting a key overwrites the previous value."""
        engine = ConfigurationEngine.from_dict({"my_key": "old_value"})
        engine.set("my_key", "new_value")
        assert engine.get("my_key") == "new_value"

    def test_contains_key(self):
        """Check if a key exists in config."""
        engine = ConfigurationEngine.from_dict({"existing": "value"})
        assert "existing" in engine
        assert "missing" not in engine

    def test_delete_key(self):
        """Delete a key from config."""
        engine = ConfigurationEngine.from_dict({"temp": "value"})
        engine.delete("temp")
        assert "temp" not in engine


# ---------------------------------------------------------------------------
# Feature Flags
# ---------------------------------------------------------------------------


class TestFeatureFlags:
    """Test feature flag functionality."""

    def test_feature_flag_enabled(self):
        """Feature flag can be enabled."""
        engine = ConfigurationEngine.from_dict({
            "feature_flags": {"new_engine": True},
        })
        assert engine.is_enabled("new_engine") is True

    def test_feature_flag_disabled(self):
        """Feature flag can be disabled."""
        engine = ConfigurationEngine.from_dict({
            "feature_flags": {"new_engine": False},
        })
        assert engine.is_enabled("new_engine") is False

    def test_feature_flag_default_false(self):
        """Unknown feature flags default to False."""
        engine = ConfigurationEngine()
        assert engine.is_enabled("unknown_flag") is False

    def test_feature_flag_toggle(self):
        """Toggle a feature flag on and off."""
        engine = ConfigurationEngine()
        engine.enable_flag("experimental")
        assert engine.is_enabled("experimental") is True
        engine.disable_flag("experimental")
        assert engine.is_enabled("experimental") is False

    def test_enable_flag(self):
        """Enable a feature flag."""
        engine = ConfigurationEngine()
        engine.enable_flag("beta_feature")
        assert engine.is_enabled("beta_feature") is True

    def test_disable_flag(self):
        """Disable a feature flag."""
        engine = ConfigurationEngine.from_dict({
            "feature_flags": {"beta_feature": True},
        })
        engine.disable_flag("beta_feature")
        assert engine.is_enabled("beta_feature") is False

    def test_list_enabled_flags(self):
        """List all enabled feature flags."""
        engine = ConfigurationEngine.from_dict({
            "feature_flags": {
                "flag_a": True,
                "flag_b": False,
                "flag_c": True,
            },
        })
        enabled = engine.enabled_flags()
        assert "flag_a" in enabled
        assert "flag_c" in enabled
        assert "flag_b" not in enabled

    def test_feature_flags_from_env(self, monkeypatch):
        """Feature flags can be set via environment variables."""
        monkeypatch.setenv("APEX_FEATURE_NEW_UI", "true")
        monkeypatch.setenv("APEX_FEATURE_BETA", "false")
        engine = ConfigurationEngine.from_env()
        assert engine.is_enabled("new_ui") is True
        assert engine.is_enabled("beta") is False


# ---------------------------------------------------------------------------
# Secrets Management
# ---------------------------------------------------------------------------


class TestSecretsManagement:
    """Test secrets storage, retrieval, and masking."""

    def test_store_and_retrieve_secret(self):
        """Store a secret and retrieve it."""
        engine = ConfigurationEngine()
        engine.set_secret("api_key", "sk-12345")
        assert engine.get_secret("api_key") == "sk-12345"

    def test_secret_exists(self):
        """Check if a secret exists."""
        engine = ConfigurationEngine()
        engine.set_secret("db_password", "s3cret")
        assert engine.has_secret("db_password") is True
        assert engine.has_secret("nonexistent") is False

    def test_delete_secret(self):
        """Delete a secret."""
        engine = ConfigurationEngine()
        engine.set_secret("temp_secret", "value")
        engine.delete_secret("temp_secret")
        assert engine.has_secret("temp_secret") is False

    def test_secret_not_in_config_export(self):
        """Secrets are not exposed in config export."""
        engine = ConfigurationEngine()
        engine.set_secret("api_key", "sk-12345")
        exported = engine.export_config()
        assert "sk-12345" not in str(exported)

    def test_secret_masked_in_repr(self):
        """Secrets are masked in string representation."""
        engine = ConfigurationEngine()
        engine.set_secret("api_key", "sk-12345")
        repr_str = repr(engine)
        assert "sk-12345" not in repr_str

    def test_get_nonexistent_secret_raises(self):
        """Getting a nonexistent secret raises KeyError."""
        engine = ConfigurationEngine()
        with pytest.raises(KeyError):
            engine.get_secret("nonexistent")

    def test_secrets_isolated_between_engines(self):
        """Secrets are isolated between engine instances."""
        engine1 = ConfigurationEngine()
        engine2 = ConfigurationEngine()
        engine1.set_secret("key", "value1")
        assert engine2.has_secret("key") is False

    def test_update_secret(self):
        """Update an existing secret."""
        engine = ConfigurationEngine()
        engine.set_secret("api_key", "old-value")
        engine.set_secret("api_key", "new-value")
        assert engine.get_secret("api_key") == "new-value"


# ---------------------------------------------------------------------------
# Engine-Specific Configuration
# ---------------------------------------------------------------------------


class TestEngineSpecificConfig:
    """Test per-engine configuration sections."""

    def test_engine_config_section(self):
        """Get configuration for a specific engine."""
        engine = ConfigurationEngine.from_dict({
            "engines": {
                "marketmaking": {"gamma": 0.1, "sigma": 0.5},
            },
        })
        mm_config = engine.get_engine_config("marketmaking")
        assert mm_config["gamma"] == 0.1
        assert mm_config["sigma"] == 0.5

    def test_engine_config_empty(self):
        """Get config for unconfigured engine returns empty dict."""
        engine = ConfigurationEngine()
        assert engine.get_engine_config("nonexistent") == {}

    def test_set_engine_config(self):
        """Set configuration for a specific engine."""
        engine = ConfigurationEngine()
        engine.set_engine_config("regtech", {"frameworks": ["PCI_DSS", "SOX"]})
        config = engine.get_engine_config("regtech")
        assert config["frameworks"] == ["PCI_DSS", "SOX"]

    def test_multiple_engine_configs(self):
        """Multiple engines can have separate configs."""
        engine = ConfigurationEngine()
        engine.set_engine_config("marketmaking", {"gamma": 0.1})
        engine.set_engine_config("altdata", {"api_key": "test"})
        assert engine.get_engine_config("marketmaking")["gamma"] == 0.1
        assert engine.get_engine_config("altdata")["api_key"] == "test"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    """Test configuration validation."""

    def test_invalid_environment_raises(self):
        """Invalid environment value raises ValidationError."""
        with pytest.raises(ValidationError):
            ConfigurationEngine.from_dict({"environment": "invalid_env"})

    def test_valid_environment_values(self):
        """All valid environment values are accepted."""
        for env in ["development", "staging", "production", "testing"]:
            engine = ConfigurationEngine.from_dict({"environment": env})
            assert engine.config.environment == env

    def test_debug_must_be_bool(self):
        """Debug flag must be boolean."""
        with pytest.raises(ValidationError):
            ConfigurationEngine.from_dict({"debug": "not_a_bool"})

    def test_log_level_validation(self):
        """Log level must be a valid value."""
        with pytest.raises(ValidationError):
            ConfigurationEngine.from_dict({"log_level": "INVALID"})


# ---------------------------------------------------------------------------
# Reload & Export
# ---------------------------------------------------------------------------


class TestReloadAndExport:
    """Test config reload and export functionality."""

    def test_reload_from_dict(self):
        """Reload configuration from a new dict."""
        engine = ConfigurationEngine.from_dict({"environment": "development"})
        engine.reload({"environment": "production", "debug": True})
        assert engine.config.environment == "production"
        assert engine.config.debug is True

    def test_export_config(self):
        """Export configuration as a dict."""
        engine = ConfigurationEngine.from_dict({
            "environment": "staging",
            "debug": False,
        })
        exported = engine.export_config()
        assert isinstance(exported, dict)
        assert exported["environment"] == "staging"

    def test_export_config_hides_secrets(self):
        """Exported config does not contain secret values."""
        engine = ConfigurationEngine()
        engine.set_secret("db_password", "super_secret")
        engine.set_secret("api_key", "sk-abc")
        exported = engine.export_config()
        assert "super_secret" not in str(exported)
        assert "sk-abc" not in str(exported)

    def test_reload_preserves_secrets(self):
        """Reloading config does not clear secrets."""
        engine = ConfigurationEngine()
        engine.set_secret("api_key", "sk-12345")
        engine.reload({"environment": "production"})
        assert engine.get_secret("api_key") == "sk-12345"


# ---------------------------------------------------------------------------
# Pydantic Settings Integration
# ---------------------------------------------------------------------------


class TestPydanticSettings:
    """Test pydantic-settings integration."""

    def test_config_is_pydantic_model(self):
        """AppConfig is a pydantic BaseSettings model."""
        from pydantic_settings import BaseSettings
        assert issubclass(AppConfig, BaseSettings)

    def test_env_prefix(self):
        """Config uses APEX_ env prefix."""
        assert AppConfig.model_config.get("env_prefix") == "APEX_"

    def test_dotenv_support(self):
        """Config supports .env file loading."""
        assert "env_file" in AppConfig.model_config


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_config_workflow(self):
        """Complete workflow: create, configure, set flags, secrets, export."""
        engine = ConfigurationEngine.from_dict({
            "environment": "testing",
            "debug": True,
            "feature_flags": {"new_ui": True, "beta": False},
        })

        # Set engine-specific config
        engine.set_engine_config("marketmaking", {"gamma": 0.1, "sigma": 0.5})

        # Set secrets
        engine.set_secret("db_password", "test_pass")
        engine.set_secret("api_key", "test_key")

        # Verify
        assert engine.is_enabled("new_ui") is True
        assert engine.is_enabled("beta") is False
        assert engine.get_secret("db_password") == "test_pass"
        assert engine.get_engine_config("marketmaking")["gamma"] == 0.1

        # Export should not leak secrets
        exported = engine.export_config()
        assert "test_pass" not in str(exported)
        assert "test_key" not in str(exported)

    def test_env_var_override(self, monkeypatch):
        """Environment variables override dict config."""
        monkeypatch.setenv("APEX_ENVIRONMENT", "production")
        engine = ConfigurationEngine.from_env()
        assert engine.config.environment == "production"
