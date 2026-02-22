"""Tests for user configuration module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from transcription_tools.user_config import load_config


class TestLoadConfig:
    """Test reading config.toml from disk."""

    def test_returns_empty_dict_when_file_missing(self, tmp_path):
        config_file = tmp_path / "nonexistent.toml"
        with patch("transcription_tools.user_config.CONFIG_FILE", config_file):
            assert load_config() == {}

    def test_reads_valid_toml(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            'openai_api_key = "sk-test123"\nopenai_model = "gpt-5-mini"\n'
        )
        with patch("transcription_tools.user_config.CONFIG_FILE", config_file):
            config = load_config()
        assert config["openai_api_key"] == "sk-test123"
        assert config["openai_model"] == "gpt-5-mini"

    def test_returns_empty_dict_on_corrupted_toml(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text("this is not valid toml [[[")
        with patch("transcription_tools.user_config.CONFIG_FILE", config_file):
            assert load_config() == {}
