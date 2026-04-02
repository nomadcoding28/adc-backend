"""
utils/config_loader.py
=======================
YAML configuration loader with environment variable interpolation.
Supports ${VAR_NAME} and ${VAR_NAME:default_value} syntax.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Updated regex to capture Group 1 (Name) and Group 2 (Optional Default Value)
_ENV_VAR_PATTERN = re.compile(r"\$\{([^}:]+)(?::([^}]*))?\}")


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load YAML config file, expanding ${ENV_VAR} or ${ENV_VAR:default} references.
    """
    p = Path(path)
    if not p.exists():
        # Try looking in parent directory (when called from subdirectory)
        p = Path("..") / path
    if not p.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. "
            f"Make sure config.yaml exists in the project root."
        )

    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML not installed. pip install PyYAML")

    raw = p.read_text(encoding="utf-8")

    # Substitute ${ENV_VAR} references
    def _substitute(text: str) -> str:
        def _replace(match: re.Match) -> str:
            var_name = match.group(1)
            default_val = match.group(2)
            
            # 1. Try to get value from environment
            value = os.getenv(var_name)
            
            if value is not None:
                return value
                
            # 2. If not in env, check if a default was provided in the string
            if default_val is not None:
                return default_val
            
            # 3. If neither, keep placeholder and warn
            logger.warning(
                "Config env var ${%s} not set and no default provided — keeping placeholder.", 
                var_name
            )
            return match.group(0)

        return _ENV_VAR_PATTERN.sub(_replace, text)

    substituted = _substitute(raw)
    config      = yaml.safe_load(substituted) or {}

    logger.debug("Config loaded from %s — keys: %s", p, list(config.keys()))
    return config


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge multiple config dicts."""
    result: Dict[str, Any] = {}
    for cfg in configs:
        _deep_merge(result, cfg)
    return result


def _deep_merge(base: Dict, override: Dict) -> None:
    """Recursively merge ``override`` into ``base`` in place."""
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val