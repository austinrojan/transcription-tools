"""Tests for TranscriptCleaner API key resolution.

Separated from test_cleanup.py because these tests mock the openai
import to avoid requiring the openai package at collection time.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Install a persistent openai mock so cleanup.py can be imported and
# patch() can resolve targets like "transcription_tools.cleanup.get_config_value"
if "openai" not in sys.modules:
    sys.modules["openai"] = MagicMock()

from transcription_tools.cleanup import TranscriptCleaner  # noqa: E402


class TestTranscriptCleanerInit:
    """Test API key resolution in TranscriptCleaner.__init__."""

    @patch("transcription_tools.cleanup.get_config_value", return_value="sk-from-config")
    @patch("transcription_tools.cleanup.OpenAI")
    def test_uses_config_file_when_no_env_var(self, mock_openai_cls, mock_get_config):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            TranscriptCleaner()
        mock_get_config.assert_called_once()

    @patch("transcription_tools.cleanup.get_config_value", return_value=None)
    def test_raises_when_no_key_anywhere(self, mock_get_config):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            with pytest.raises(RuntimeError, match="No OpenAI API key configured"):
                TranscriptCleaner()

    @patch("transcription_tools.cleanup.get_config_value", return_value=None)
    def test_error_message_mentions_config_command(self, mock_get_config):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            with pytest.raises(RuntimeError, match="transcription-tools config"):
                TranscriptCleaner()
