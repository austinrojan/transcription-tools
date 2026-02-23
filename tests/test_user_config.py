"""Tests for user configuration module."""

from __future__ import annotations

import os
from contextlib import contextmanager
from unittest.mock import patch

from transcription_tools.user_config import delete_config_key, get_config_value, load_config, save_config


@contextmanager
def _patched_config(tmp_path):
    """Patch CONFIG_DIR and CONFIG_FILE to use tmp_path."""
    config_dir = tmp_path / "config" / "transcription-tools"
    config_file = config_dir / "config.toml"
    with (
        patch("transcription_tools.user_config.CONFIG_DIR", config_dir),
        patch("transcription_tools.user_config.CONFIG_FILE", config_file),
    ):
        yield config_dir, config_file


class TestLoadConfig:
    """Test reading config.toml from disk."""

    def test_returns_empty_dict_when_file_missing(self, tmp_path):
        config_file = tmp_path / "nonexistent.toml"
        with patch("transcription_tools.user_config.CONFIG_FILE", config_file):
            assert load_config() == {}

    def test_reads_valid_toml(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text('openai_api_key = "sk-test123"\nopenai_model = "gpt-5-mini"\n')
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
        with _patched_config(tmp_path) as (_, config_file):
            save_config({"openai_api_key": "sk-abc"})
        assert config_file.exists()
        assert "sk-abc" in config_file.read_text()

    def test_sets_file_permissions_to_600(self, tmp_path):
        with _patched_config(tmp_path) as (_, config_file):
            save_config({"openai_api_key": "sk-abc"})
        assert config_file.stat().st_mode & 0o777 == 0o600

    def test_merges_with_existing_config(self, tmp_path):
        with _patched_config(tmp_path) as (config_dir, config_file):
            config_dir.mkdir(parents=True)
            config_file.write_text('openai_model = "gpt-5-mini"\n')
            save_config({"openai_api_key": "sk-new"})
            config = load_config()
        assert config["openai_api_key"] == "sk-new"
        assert config["openai_model"] == "gpt-5-mini"

    def test_roundtrips_special_characters(self, tmp_path):
        with _patched_config(tmp_path) as (_, config_file):
            save_config({"tricky": 'has "quotes" and \\backslash and \nnewline'})
            config = load_config()
        assert config["tricky"] == 'has "quotes" and \\backslash and \nnewline'


class TestGetConfigValue:
    """Test env-var > config-file > default precedence."""

    def test_env_var_wins_over_config_file(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text('openai_api_key = "sk-from-file"\n')
        with (
            patch("transcription_tools.user_config.CONFIG_FILE", config_file),
            patch.dict(os.environ, {"OPENAI_API_KEY": "sk-from-env"}),
        ):
            result = get_config_value("openai_api_key", env_var="OPENAI_API_KEY")
        assert result == "sk-from-env"

    def test_config_file_used_when_no_env_var(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text('openai_api_key = "sk-from-file"\n')
        with patch("transcription_tools.user_config.CONFIG_FILE", config_file), patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            result = get_config_value("openai_api_key", env_var="OPENAI_API_KEY")
        assert result == "sk-from-file"

    def test_returns_default_when_no_env_and_no_config(self, tmp_path):
        config_file = tmp_path / "nonexistent.toml"
        with patch("transcription_tools.user_config.CONFIG_FILE", config_file), patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            result = get_config_value("openai_api_key", env_var="OPENAI_API_KEY", default="none")
        assert result == "none"

    def test_returns_none_when_nothing_configured(self, tmp_path):
        config_file = tmp_path / "nonexistent.toml"
        with patch("transcription_tools.user_config.CONFIG_FILE", config_file), patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            result = get_config_value("openai_api_key", env_var="OPENAI_API_KEY")
        assert result is None


class TestDeleteConfigKey:
    """Test removing a key from config."""

    def test_removes_existing_key(self, tmp_path):
        with _patched_config(tmp_path) as (config_dir, config_file):
            config_dir.mkdir(parents=True)
            config_file.write_text('openai_model = "gpt-5"\nopenai_api_key = "sk-x"\n')
            delete_config_key("openai_model")
            config = load_config()
        assert "openai_model" not in config
        assert config["openai_api_key"] == "sk-x"

    def test_noop_for_missing_key(self, tmp_path):
        with _patched_config(tmp_path) as (config_dir, config_file):
            config_dir.mkdir(parents=True)
            config_file.write_text('openai_api_key = "sk-x"\n')
            delete_config_key("nonexistent")
            config = load_config()
        assert config["openai_api_key"] == "sk-x"
