"""Tests for user configuration module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from transcription_tools.user_config import load_config, save_config


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


class TestSaveConfig:
    """Test writing config.toml to disk."""

    def test_creates_config_dir_and_file(self, tmp_path):
        config_dir = tmp_path / "config" / "transcription-tools"
        config_file = config_dir / "config.toml"
        with patch("transcription_tools.user_config.CONFIG_DIR", config_dir), \
             patch("transcription_tools.user_config.CONFIG_FILE", config_file):
            save_config({"openai_api_key": "sk-abc"})
        assert config_file.exists()
        assert "sk-abc" in config_file.read_text()

    def test_sets_file_permissions_to_600(self, tmp_path):
        config_dir = tmp_path / "config" / "transcription-tools"
        config_file = config_dir / "config.toml"
        with patch("transcription_tools.user_config.CONFIG_DIR", config_dir), \
             patch("transcription_tools.user_config.CONFIG_FILE", config_file):
            save_config({"openai_api_key": "sk-abc"})
        assert config_file.stat().st_mode & 0o777 == 0o600

    def test_merges_with_existing_config(self, tmp_path):
        config_dir = tmp_path / "config" / "transcription-tools"
        config_file = config_dir / "config.toml"
        config_dir.mkdir(parents=True)
        config_file.write_text('openai_model = "gpt-5-mini"\n')
        with patch("transcription_tools.user_config.CONFIG_DIR", config_dir), \
             patch("transcription_tools.user_config.CONFIG_FILE", config_file):
            save_config({"openai_api_key": "sk-new"})
            config = load_config()
        assert config["openai_api_key"] == "sk-new"
        assert config["openai_model"] == "gpt-5-mini"
