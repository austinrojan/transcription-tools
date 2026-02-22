"""Install directory layout constants.

Single source of truth for the on-disk layout used by the installer,
update command, and uninstall command.
"""

from __future__ import annotations

from pathlib import Path

INSTALL_DIR = Path.home() / "Library" / "Application Support" / "transcription-tools"
PYTHON_DIR = INSTALL_DIR / "python"
VENV_DIR = INSTALL_DIR / "venv"
FFMPEG_DIR = INSTALL_DIR / "ffmpeg"
VERSION_FILE = INSTALL_DIR / "version.txt"

SERVICES_DIR = Path.home() / "Library" / "Services"

WRAPPER_COMMANDS = (
    "transcribe-veryfast",
    "transcribe-fast",
    "transcribe-medium",
    "transcribe-slow",
    "transcribe-veryslow",
    "transcription-tools",
)
