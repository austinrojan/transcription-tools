"""Tests for the transcription-tools meta-command."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from transcription_tools.meta_cli import config_command, main


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

    @patch("transcription_tools.meta_cli._write_config")
    @patch(
        "transcription_tools.meta_cli.load_config",
        return_value={"openai_model": "gpt-5", "openai_api_key": "sk-x"},
    )
    def test_removes_key(self, mock_load, mock_write):
        config_command(
            show=False, set_api_key=False, set_pair=None, unset="openai_model",
        )
        written = mock_write.call_args[0][0]
        assert "openai_model" not in written
        assert "openai_api_key" in written


class TestMainDispatch:

    @patch("sys.argv", ["transcription-tools", "version"])
    def test_version_prints_version(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code in (0, None)
        output = capsys.readouterr().out
        assert "2.0.0" in output
