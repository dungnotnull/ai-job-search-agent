"""
config.py — Centralized configuration loader for ai-job-search-enhanced.
Reads config/agent_config.yaml + .env via python-dotenv.
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config" / "agent_config.yaml"
ENV_PATH = Path(__file__).parent.parent / ".env"


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class AgentConfig:
    """Loads and provides access to agent configuration."""

    def __init__(self, config_path: Optional[Path] = None, env_path: Optional[Path] = None):
        self._env_path = env_path or ENV_PATH
        self._config_path = config_path or CONFIG_PATH
        self._data: dict[str, Any] = {}

        if self._env_path.exists():
            load_dotenv(self._env_path)

        self._load_yaml()
        self._apply_env_overrides()

    def _load_yaml(self):
        if not self._config_path.exists():
            logger.warning("Config file not found: %s", self._config_path)
            return
        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                self._data = yaml.safe_load(f) or {}
            logger.info("Loaded config from %s", self._config_path)
        except Exception as exc:
            logger.warning("Failed to load config: %s", exc)

    def _apply_env_overrides(self):
        env_map = {
            "LOG_LEVEL": ("agent", "log_level"),
            "PORT": ("server", "port"),
            "HOST": ("server", "host"),
            "CLAUDE_MODEL": ("llm", "claude_model"),
            "OPENAI_MODEL": ("llm", "openai_model"),
            "OLLAMA_MODEL": ("llm", "ollama_model"),
            "OLLAMA_BASE_URL": ("llm", "ollama_base_url"),
            "PRIVACY_MODE": ("llm", "privacy_mode"),
        }
        for env_key, (section, field) in env_map.items():
            val = os.getenv(env_key)
            if val is not None:
                if section not in self._data:
                    self._data[section] = {}
                if field == "port":
                    val = int(val)
                elif field == "privacy_mode":
                    val = val.lower() == "true"
                self._data[section][field] = val

    def get(self, *keys: str, default: Any = None) -> Any:
        node = self._data
        for key in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(key)
            if node is None:
                return default
        return node

    @property
    def search(self) -> dict:
        return self._data.get("search", {})

    @property
    def llm(self) -> dict:
        return self._data.get("llm", {})

    @property
    def hf_models(self) -> dict:
        return self._data.get("hf_models", {})

    @property
    def salary_predictor(self) -> dict:
        return self._data.get("salary_predictor", {})

    @property
    def sentiment(self) -> dict:
        return self._data.get("sentiment", {})

    @property
    def knowledge_updater(self) -> dict:
        return self._data.get("knowledge_updater", {})

    @property
    def quality_gates(self) -> dict:
        return self._data.get("quality_gates", {})

    @property
    def server(self) -> dict:
        return self._data.get("server", {})

    @property
    def log_level(self) -> str:
        return os.getenv("LOG_LEVEL", self.get("agent", "log_level", default="INFO"))

    @property
    def data_dir(self) -> Path:
        return Path(self.get("agent", "data_dir", default="./data"))

    @property
    def models_dir(self) -> Path:
        return Path(self.get("agent", "models_dir", default="./models"))


_instance: Optional[AgentConfig] = None


def get_config(config_path: Optional[Path] = None, env_path: Optional[Path] = None) -> AgentConfig:
    global _instance
    if _instance is None:
        _instance = AgentConfig(config_path=config_path, env_path=env_path)
    return _instance
