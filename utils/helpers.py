from __future__ import annotations

import logging
import os
import re
import stat
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Env vars that MUST be set for the bot to run (validated at startup)
_REQUIRED_ENV_VARS = {
    "POLY_API_KEY",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
}


def load_config(settings_path: str = "config/settings.yaml", env_path: str = "config/secrets.env") -> dict:
    """Load settings.yaml and resolve ${ENV_VAR} references from secrets.env."""
    env_file = Path(env_path)
    if env_file.exists():
        _check_file_permissions(env_file)
        load_dotenv(env_file)

    with open(settings_path, "r") as f:
        raw = f.read()

    missing_vars: list[str] = []

    def replacer(match):
        var = match.group(1)
        value = os.environ.get(var)
        if value is None:
            if var in _REQUIRED_ENV_VARS:
                missing_vars.append(var)
            return match.group(0)  # keep placeholder for optional vars
        return value

    resolved = re.sub(r"\$\{(\w+)}", replacer, raw)

    if missing_vars:
        raise EnvironmentError(
            f"Required environment variables not set: {', '.join(sorted(missing_vars))}. "
            f"Copy config/secrets.env.example to config/secrets.env and fill in your keys."
        )

    config = yaml.safe_load(resolved)
    _validate_config(config)
    return config


def _check_file_permissions(path: Path):
    """Warn if secrets file has overly permissive permissions (non-Windows)."""
    if sys.platform == "win32":
        return
    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        logger.warning(
            "⚠️ %s has permissive mode %o (group/other can read). "
            "Run: chmod 600 %s",
            path, mode, path,
        )


def _validate_config(config: dict):
    """Validate critical config values at startup."""
    capital = config.get("general", {}).get("capital_usd")
    if not isinstance(capital, (int, float)) or capital <= 0:
        raise ValueError(f"Invalid capital_usd: {capital}. Must be a positive number.")

    mode = config.get("general", {}).get("mode")
    if mode not in ("paper", "live"):
        raise ValueError(f"Invalid mode: {mode!r}. Must be 'paper' or 'live'.")

    risk = config.get("risk", {})
    for key in ("max_daily_drawdown_pct", "max_position_pct", "max_total_exposure_pct"):
        val = risk.get(key)
        if val is not None and (not isinstance(val, (int, float)) or val <= 0):
            raise ValueError(f"Invalid risk.{key}: {val}. Must be a positive number.")
