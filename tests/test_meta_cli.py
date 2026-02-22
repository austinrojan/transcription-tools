"""Tests for the transcription-tools meta-command."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from transcription_tools.meta_cli import (
    _parse_version,
    check_for_update,
    config_command,
    get_uninstall_paths,
    main,
)


def _has_path_matching(paths: list, pattern: str) -> bool:
    """Check if any path in the list contains the pattern string."""
    return any(pattern in str(p) for p in paths)


class TestConfigShow:

    @patch("transcription_tools.meta_cli.load_config")
    def test_masks_api_key(self, mock_load, capsys):
        mock_load.return_value = {
            "openai_api_key": "sk-proj-abc123def456",
            "openai_model": "gpt-5-mini",
        }
        config_command(show=True, set_api_key=False, set_pair=None, unset=None)
        output = capsys.readouterr().out
        assert "sk-proj-abc123def456" not in output
        assert "sk-p" in output
        assert "gpt-5-mini" in output

    @patch("transcription_tools.meta_cli.load_config")
    def test_shows_empty_config_message(self, mock_load, capsys):
        mock_load.return_value = {}
        config_command(show=True, set_api_key=False, set_pair=None, unset=None)
        output = capsys.readouterr().out
        assert "No configuration" in output


class TestConfigSetApiKey:

    @patch("transcription_tools.meta_cli.save_config")
    @patch("builtins.input", return_value="sk-proj-newkey123")
    def test_saves_valid_key(self, mock_input, mock_save, capsys):
        config_command(show=False, set_api_key=True, set_pair=None, unset=None)
        saved = mock_save.call_args[0][0]
        assert saved["openai_api_key"] == "sk-proj-newkey123"

    @patch("builtins.input", return_value="not-a-valid-key")
    def test_rejects_invalid_key(self, mock_input, capsys):
        with pytest.raises(SystemExit):
            config_command(show=False, set_api_key=True, set_pair=None, unset=None)


class TestConfigSet:

    @patch("transcription_tools.meta_cli.save_config")
    def test_sets_arbitrary_key(self, mock_save):
        config_command(
            show=False, set_api_key=False,
            set_pair=("openai_model", "gpt-5"), unset=None,
        )
        mock_save.assert_called_once_with({"openai_model": "gpt-5"})


class TestConfigUnset:

    @patch("transcription_tools.meta_cli.delete_config_key")
    def test_removes_key(self, mock_delete):
        config_command(
            show=False, set_api_key=False, set_pair=None, unset="openai_model",
        )
        mock_delete.assert_called_once_with("openai_model")


class TestParseVersion:

    def test_parses_simple_version(self):
        assert _parse_version("2.0.0") == (2, 0, 0)

    def test_parses_double_digit_version(self):
        assert _parse_version("2.10.0") == (2, 10, 0)


class TestCheckForUpdate:

    @patch("transcription_tools.meta_cli._get_installed_version", return_value="2.0.0")
    @patch("transcription_tools.meta_cli._get_latest_version", return_value="2.1.0")
    def test_detects_available_update(self, mock_latest, mock_installed):
        has_update, current, latest = check_for_update()
        assert has_update is True
        assert current == "2.0.0"
        assert latest == "2.1.0"

    @patch("transcription_tools.meta_cli._get_installed_version", return_value="2.0.0")
    @patch("transcription_tools.meta_cli._get_latest_version", return_value="2.0.0")
    def test_no_update_when_current(self, mock_latest, mock_installed):
        has_update, _, _ = check_for_update()
        assert has_update is False

    @patch("transcription_tools.meta_cli._get_installed_version", return_value="2.9.0")
    @patch("transcription_tools.meta_cli._get_latest_version", return_value="2.10.0")
    def test_handles_double_digit_minor_version(self, mock_latest, mock_installed):
        """String comparison would incorrectly say 2.9.0 > 2.10.0."""
        has_update, _, _ = check_for_update()
        assert has_update is True


class TestGetUninstallPaths:

    def test_includes_install_dir(self):
        paths = get_uninstall_paths(keep_config=True, keep_models=True)
        assert _has_path_matching(paths["dirs"], "Application Support/transcription-tools")

    def test_includes_wrapper_scripts(self):
        paths = get_uninstall_paths(keep_config=True, keep_models=True)
        assert _has_path_matching(paths["files"], "transcribe-fast")

    def test_includes_workflows(self):
        paths = get_uninstall_paths(keep_config=True, keep_models=True)
        assert _has_path_matching(paths["dirs"], "Transcribe Audio")

    def test_excludes_config_when_keep_true(self):
        paths = get_uninstall_paths(keep_config=True, keep_models=True)
        assert not _has_path_matching(paths["dirs"], ".config/transcription-tools")

    def test_includes_config_when_keep_false(self):
        paths = get_uninstall_paths(keep_config=False, keep_models=True)
        assert _has_path_matching(paths["dirs"], ".config/transcription-tools")

    def test_excludes_model_caches_when_keep_true(self):
        paths = get_uninstall_paths(keep_config=True, keep_models=True)
        assert not _has_path_matching(paths["dirs"], ".cache/whisper")

    def test_includes_model_caches_when_keep_false(self):
        paths = get_uninstall_paths(keep_config=False, keep_models=False)
        assert _has_path_matching(paths["dirs"], ".cache/whisper")
        assert _has_path_matching(paths["dirs"], ".cache/huggingface")


class TestMainDispatch:

    @patch("sys.argv", ["transcription-tools", "version"])
    def test_version_prints_version(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code in (0, None)
        output = capsys.readouterr().out
        assert "2.0.0" in output
