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


def _write_config(data: dict) -> None:
    """Write a full config dict to disk as TOML. Sets 0o600 permissions."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    for key, value in sorted(data.items()):
        if isinstance(value, str):
            lines.append(f'{key} = "{value}"')
        elif isinstance(value, bool):
            lines.append(f"{key} = {'true' if value else 'false'}")
        elif isinstance(value, (int, float)):
            lines.append(f"{key} = {value}")
    CONFIG_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    CONFIG_FILE.chmod(0o600)


def save_config(updates: dict) -> None:
    """Merge updates into the config file, creating it if needed."""
    existing = load_config()
    existing.update(updates)
    _write_config(existing)
