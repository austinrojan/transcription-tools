"""Tests for install directory path definitions."""

from __future__ import annotations

from transcription_tools.install_paths import (
    FFMPEG_DIR,
    INSTALL_DIR,
    PYTHON_DIR,
    SERVICES_DIR,
    VENV_DIR,
    VERSION_FILE,
    WRAPPER_COMMANDS,
)


class TestInstallPaths:
    def test_install_dir_is_in_application_support(self):
        assert "Library/Application Support/transcription-tools" in str(INSTALL_DIR)

    def test_venv_under_install_dir(self):
        assert VENV_DIR.parent == INSTALL_DIR

    def test_python_under_install_dir(self):
        assert PYTHON_DIR.parent == INSTALL_DIR

    def test_ffmpeg_under_install_dir(self):
        assert FFMPEG_DIR.parent == INSTALL_DIR

    def test_version_file_under_install_dir(self):
        assert VERSION_FILE.parent == INSTALL_DIR

    def test_services_dir_path(self):
        assert "Library/Services" in str(SERVICES_DIR)

    def test_wrapper_commands_include_all_tiers_and_meta(self):
        assert "transcribe-fast" in WRAPPER_COMMANDS
        assert "transcription-tools" in WRAPPER_COMMANDS

    def test_wrapper_commands_is_tuple(self):
        assert isinstance(WRAPPER_COMMANDS, tuple)
