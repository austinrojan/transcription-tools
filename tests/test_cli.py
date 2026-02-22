"""Tests for CLI entry points and the run() workflow."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

# Install persistent openai mock so cleanup.py (imported by cli.py) can load
if "openai" not in sys.modules:
    sys.modules["openai"] = MagicMock()

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from transcription_tools.cli import run  # noqa: E402


class TestCleanupGracefulDegradation:
    """Test that missing API key does not crash transcription."""

    @patch("transcription_tools.cli.get_available_tiers")
    @patch("transcription_tools.cli._run_transcription")
    @patch("transcription_tools.cli.TranscriptCleaner")
    @patch("transcription_tools.cli._parse_args")
    def test_prints_note_when_no_api_key(
        self, mock_parse, mock_cleaner_cls, mock_transcribe, mock_available, tmp_path, capsys
    ):
        input_file = tmp_path / "recording.mp3"
        input_file.touch()
        output_file = tmp_path / "recording_fast.txt"
        output_file.write_text("raw transcript\n")

        mock_args = MagicMock()
        mock_args.input_file = str(input_file)
        mock_args.no_cleanup = False
        mock_args.cleanup = False
        mock_args.cleanup_only = False
        mock_args.openai_model = None
        mock_args.openai_base_url = None
        mock_parse.return_value = mock_args

        from transcription_tools.config import TIERS
        mock_available.return_value = {"fast": TIERS["fast"]}

        mock_cleaner_cls.side_effect = RuntimeError("No OpenAI API key configured.")

        run("fast")

        output = capsys.readouterr().out
        assert "API key" in output or "cleanup" in output.lower()

    @patch("transcription_tools.cli.get_available_tiers")
    @patch("transcription_tools.cli._run_transcription")
    @patch("transcription_tools.cli._parse_args")
    def test_no_cleanup_flag_skips_cleanup(
        self, mock_parse, mock_transcribe, mock_available, tmp_path
    ):
        input_file = tmp_path / "recording.mp3"
        input_file.touch()

        mock_args = MagicMock()
        mock_args.input_file = str(input_file)
        mock_args.no_cleanup = True
        mock_args.cleanup = False
        mock_args.cleanup_only = False
        mock_args.openai_model = None
        mock_args.openai_base_url = None
        mock_parse.return_value = mock_args

        from transcription_tools.config import TIERS
        mock_available.return_value = {"fast": TIERS["fast"]}

        run("fast")


class TestCleanupFlag:
    """Test that --cleanup flag requires cleanup (fails if no key)."""

    @patch("transcription_tools.cli.get_available_tiers")
    @patch("transcription_tools.cli._run_transcription")
    @patch("transcription_tools.cli.TranscriptCleaner")
    @patch("transcription_tools.cli._parse_args")
    def test_cleanup_flag_exits_on_missing_key(
        self, mock_parse, mock_cleaner_cls, mock_transcribe, mock_available, tmp_path, capsys
    ):
        input_file = tmp_path / "recording.mp3"
        input_file.touch()
        output_file = tmp_path / "recording_fast.txt"
        output_file.write_text("raw transcript\n")

        mock_args = MagicMock()
        mock_args.input_file = str(input_file)
        mock_args.no_cleanup = False
        mock_args.cleanup = True
        mock_args.cleanup_only = False
        mock_args.openai_model = None
        mock_args.openai_base_url = None
        mock_parse.return_value = mock_args

        from transcription_tools.config import TIERS
        mock_available.return_value = {"fast": TIERS["fast"]}
        mock_cleaner_cls.side_effect = RuntimeError("No OpenAI API key configured.")

        with pytest.raises(SystemExit):
            run("fast")


class TestTierAvailabilityCheck:

    @patch("transcription_tools.cli.get_available_tiers", return_value={})
    @patch("transcription_tools.cli._parse_args")
    def test_exits_when_tier_not_available(self, mock_parse, mock_available, capsys):
        mock_args = MagicMock()
        mock_args.input_file = "/fake/file.mp3"
        mock_parse.return_value = mock_args

        with pytest.raises(SystemExit):
            run("slow")

        output = capsys.readouterr().out
        assert "slow" in output.lower()
