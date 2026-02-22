"""User configuration via ~/.config/transcription-tools/config.toml.

Provides persistent storage for API keys and preferences.
Environment variables always take precedence over config file values.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "transcription-tools"
CONFIG_FILE = CONFIG_DIR / "config.toml"


def load_config() -> dict:
    """Read the config file, returning an empty dict if missing or invalid."""
    if not CONFIG_FILE.exists():
        return {}
    try:
        return tomllib.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError):
        return {}
