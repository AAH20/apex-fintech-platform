"""Configuration Engine for the Apex Fintech Platform.

Provides centralized configuration management, feature flags, and secrets
management using pydantic-settings for validation and environment variable
integration.

Usage:
    engine = ConfigurationEngine.from_env()
    engine.set_secret("api_key", "sk-12345")
    engine.enable_flag("new_ui")
    if engine.is_enabled("new_ui"):
        ...
"""
from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Environment(str, Enum):
    """Deployment environment."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Logging severity levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Pydantic Settings
# ---------------------------------------------------------------------------


class AppConfig(BaseSettings):
    """Application configuration loaded from environment variables.

    Environment variables must be prefixed with ``APEX_``.
    Supports loading from a ``.env`` file.
    """

    model_config = SettingsConfigDict(
        env_prefix="APEX_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    log_level: LogLevel = LogLevel.INFO
    app_name: str = "apex-fintech-platform"
    version: str = "0.1.0"


class EngineConfig(BaseModel):
    """Configuration for a single engine."""

    name: str
    enabled: bool = True
    settings: dict[str, Any] = Field(default_factory=dict)


class FeatureFlags(BaseModel):
    """Feature flag definitions."""

    flags: dict[str, bool] = Field(default_factory=dict)

    def is_enabled(self, name: str) -> bool:
        """Check if a feature flag is enabled."""
        return self.flags.get(name, False)

    def enable(self, name: str) -> None:
        """Enable a feature flag."""
        self.flags[name] = True

    def disable(self, name: str) -> None:
        """Disable a feature flag."""
        self.flags[name] = False

    def enabled(self) -> list[str]:
        """Return list of enabled flag names."""
        return [k for k, v in self.flags.items() if v]


# ---------------------------------------------------------------------------
# Secrets Manager
# ---------------------------------------------------------------------------


class SecretsManager:
    """Manages secrets with masking and isolation.

    Secrets are stored in-memory and never exposed through config export
    or string representation.
    """

    def __init__(self) -> None:
        self._secrets: dict[str, str] = {}

    def set(self, key: str, value: str) -> None:
        """Store a secret."""
        self._secrets[key] = value

    def get(self, key: str) -> str:
        """Retrieve a secret.

        Raises:
            KeyError: If the secret does not exist.
        """
        if key not in self._secrets:
            raise KeyError(f"Secret '{key}' not found")
        return self._secrets[key]

    def has(self, key: str) -> bool:
        """Check if a secret exists."""
        return key in self._secrets

    def delete(self, key: str) -> None:
        """Delete a secret."""
        self._secrets.pop(key, None)

    def clear(self) -> None:
        """Remove all secrets."""
        self._secrets.clear()

    def keys(self) -> list[str]:
        """Return secret keys (names only, never values)."""
        return list(self._secrets.keys())

    def masked(self) -> dict[str, str]:
        """Return secrets with masked values."""
        return {k: "********" for k in self._secrets}


# ---------------------------------------------------------------------------
# Configuration Engine
# ---------------------------------------------------------------------------


class ConfigurationEngine:
    """Central configuration engine for all platform engines.

    Manages application config, feature flags, per-engine settings,
    and secrets. Uses pydantic-settings for validation.
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        """Initialize the configuration engine.

        Args:
            config: Pre-built AppConfig. If None, uses defaults.
        """
        self._config = config or AppConfig()
        self._feature_flags = FeatureFlags()
        self._secrets = SecretsManager()
        self._engine_configs: dict[str, dict[str, Any]] = {}

    # -----------------------------------------------------------------------
    # Factory Methods
    # -----------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConfigurationEngine:
        """Create engine from a dictionary.

        Args:
            data: Configuration data. Supports 'feature_flags' and 'engines' keys.

        Returns:
            ConfigurationEngine instance.
        """
        # Extract known keys for AppConfig
        app_keys = {"environment", "debug", "log_level", "app_name", "version"}
        app_data = {k: v for k, v in data.items() if k in app_keys}

        config = AppConfig(**app_data)
        engine = cls(config=config)

        # Load feature flags
        if "feature_flags" in data:
            for name, enabled in data["feature_flags"].items():
                if enabled:
                    engine._feature_flags.enable(name)
                else:
                    engine._feature_flags.disable(name)

        # Load engine configs
        if "engines" in data:
            for name, settings in data["engines"].items():
                engine._engine_configs[name] = dict(settings)

        # Store any remaining top-level keys as generic config
        known_top = app_keys | {"feature_flags", "engines"}
        for key, value in data.items():
            if key not in known_top:
                engine._engine_configs[key] = value

        return engine

    @classmethod
    def from_env(cls) -> ConfigurationEngine:
        """Create engine from environment variables.

        Returns:
            ConfigurationEngine instance.
        """
        config = AppConfig()
        engine = cls(config=config)

        # Load feature flags from env (APEX_FEATURE_*)
        for key, value in os.environ.items():
            if key.startswith("APEX_FEATURE_"):
                flag_name = key[len("APEX_FEATURE_"):].lower()
                if value.lower() in ("true", "1", "yes", "on"):
                    engine._feature_flags.enable(flag_name)
                else:
                    engine._feature_flags.disable(flag_name)

        return engine

    @classmethod
    def from_dotenv(cls, path: str | Path) -> ConfigurationEngine:
        """Create engine from a .env file.

        Args:
            path: Path to .env file.

        Returns:
            ConfigurationEngine instance.
        """
        env_path = Path(path)
        if not env_path.exists():
            raise FileNotFoundError(f".env file not found: {path}")

        # Parse .env file
        env_vars: dict[str, str] = {}
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                env_vars[key.strip()] = value.strip().strip('"').strip("'")

        # Temporarily set env vars and create from env
        old_env = dict(os.environ)
        try:
            os.environ.update(env_vars)
            return cls.from_env()
        finally:
            os.environ.clear()
            os.environ.update(old_env)

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def config(self) -> AppConfig:
        """Access the underlying AppConfig."""
        return self._config

    @property
    def feature_flags(self) -> FeatureFlags:
        """Access feature flags."""
        return self._feature_flags

    @property
    def secrets(self) -> SecretsManager:
        """Access secrets manager."""
        return self._secrets

    # -----------------------------------------------------------------------
    # Config Access
    # -----------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.

        Args:
            key: Configuration key.
            default: Default value if key not found. If not provided,
                raises KeyError when the key is missing.

        Returns:
            Configuration value or default.

        Raises:
            KeyError: If key not found and no default provided.
        """
        if hasattr(self._config, key):
            return getattr(self._config, key)
        if key in self._engine_configs:
            return self._engine_configs[key]
        if default is not None:
            return default
        raise KeyError(f"Configuration key '{key}' not found")

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value.

        Args:
            key: Configuration key.
            value: Value to set.
        """
        if hasattr(self._config, key):
            setattr(self._config, key, value)
        else:
            self._engine_configs[key] = value

    def delete(self, key: str) -> None:
        """Delete a configuration key.

        Args:
            key: Key to delete.
        """
        if key in self._engine_configs:
            del self._engine_configs[key]

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in config."""
        return hasattr(self._config, key) or key in self._engine_configs

    # -----------------------------------------------------------------------
    # Feature Flags
    # -----------------------------------------------------------------------

    def is_enabled(self, name: str) -> bool:
        """Check if a feature flag is enabled.

        Args:
            name: Feature flag name.

        Returns:
            True if enabled, False otherwise.
        """
        return self._feature_flags.is_enabled(name)

    def enable_flag(self, name: str) -> None:
        """Enable a feature flag."""
        self._feature_flags.enable(name)

    def disable_flag(self, name: str) -> None:
        """Disable a feature flag."""
        self._feature_flags.disable(name)

    def enabled_flags(self) -> list[str]:
        """Return list of enabled feature flag names."""
        return self._feature_flags.enabled()

    # -----------------------------------------------------------------------
    # Secrets Management
    # -----------------------------------------------------------------------

    def set_secret(self, key: str, value: str) -> None:
        """Store a secret.

        Args:
            key: Secret name.
            value: Secret value.
        """
        self._secrets.set(key, value)

    def get_secret(self, key: str) -> str:
        """Retrieve a secret.

        Args:
            key: Secret name.

        Returns:
            Secret value.

        Raises:
            KeyError: If secret not found.
        """
        return self._secrets.get(key)

    def has_secret(self, key: str) -> bool:
        """Check if a secret exists."""
        return self._secrets.has(key)

    def delete_secret(self, key: str) -> None:
        """Delete a secret."""
        self._secrets.delete(key)

    # -----------------------------------------------------------------------
    # Engine-Specific Config
    # -----------------------------------------------------------------------

    def get_engine_config(self, name: str) -> dict[str, Any]:
        """Get configuration for a specific engine.

        Args:
            name: Engine name (e.g., 'marketmaking', 'regtech').

        Returns:
            Engine configuration dict (empty if not configured).
        """
        return self._engine_configs.get(name, {})

    def set_engine_config(self, name: str, config: dict[str, Any]) -> None:
        """Set configuration for a specific engine.

        Args:
            name: Engine name.
            config: Configuration dict.
        """
        self._engine_configs[name] = dict(config)

    # -----------------------------------------------------------------------
    # Export & Reload
    # -----------------------------------------------------------------------

    def export_config(self) -> dict[str, Any]:
        """Export configuration as a dictionary.

        Secrets are excluded from the export.

        Returns:
            Configuration dictionary.
        """
        return {
            "environment": self._config.environment.value,
            "debug": self._config.debug,
            "log_level": self._config.log_level.value,
            "app_name": self._config.app_name,
            "version": self._config.version,
            "feature_flags": dict(self._feature_flags.flags),
            "engines": dict(self._engine_configs),
        }

    def reload(self, data: dict[str, Any]) -> None:
        """Reload configuration from a dictionary.

        Secrets are preserved across reloads.

        Args:
            data: New configuration data.
        """
        # Preserve existing secrets
        old_secrets = dict(self._secrets._secrets)

        # Re-initialize from dict
        new_engine = self.from_dict(data)

        self._config = new_engine._config
        self._feature_flags = new_engine._feature_flags
        self._engine_configs = new_engine._engine_configs

        # Restore secrets
        self._secrets._secrets = old_secrets

    # -----------------------------------------------------------------------
    # Representation
    # -----------------------------------------------------------------------

    def __repr__(self) -> str:
        """String representation with masked secrets."""
        return (
            f"ConfigurationEngine("
            f"environment={self._config.environment.value!r}, "
            f"debug={self._config.debug}, "
            f"features={self._feature_flags.flags!r}, "
            f"secrets={self._secrets.masked()!r}"
            f")"
        )
