from __future__ import annotations

import os
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv


def load_config(settings_path: str = "config/settings.yaml", env_path: str = "config/secrets.env") -> dict:
    """Load settings.yaml and resolve ${ENV_VAR} references from secrets.env."""
    env_file = Path(env_path)
    if env_file.exists():
        load_dotenv(env_file)

    with open(settings_path, "r") as f:
        raw = f.read()

    # Replace ${VAR_NAME} with environment variable values
    def replacer(match):
        var = match.group(1)
        return os.environ.get(var, match.group(0))

    resolved = re.sub(r"\$\{(\w+)}", replacer, raw)
    return yaml.safe_load(resolved)
