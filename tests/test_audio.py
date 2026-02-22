"""Tests for audio conversion module."""

import subprocess
from unittest.mock import MagicMock, patch
from pathlib import Path

import pytest

from transcription_tools.audio import (
    find_ffmpeg,
    convert_to_wav,
    FFMPEG_CANDIDATES,
    ENHANCED_FILTER_CHAIN,
    SAMPLE_RATE_HZ,
)


class TestFindFfmpeg:
    def test_returns_shutil_which_result(self):
        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            assert find_ffmpeg() == "/usr/bin/ffmpeg"

    def test_falls_back_to_candidates(self):
        with patch("shutil.which", return_value=None), \
             patch.object(Path, "is_file", return_value=True):
            result = find_ffmpeg()
            assert result in FFMPEG_CANDIDATES

    def test_raises_when_not_found(self):
        with patch("shutil.which", return_value=None), \
             patch.object(Path, "is_file", return_value=False):
            with pytest.raises(FileNotFoundError, match="ffmpeg not found"):
                find_ffmpeg()


@patch("transcription_tools.audio.find_ffmpeg", return_value="/usr/bin/ffmpeg")
@patch("transcription_tools.audio._copy_input_to_temp")
@patch("transcription_tools.audio.tempfile")
@patch("transcription_tools.audio.subprocess")
class TestConvertToWav:
    """Tests for convert_to_wav."""

    def _setup_mocks(self, mock_subproc, mock_tempfile, mock_copy):
        """Configure standard mock return values."""
        # Expose real exception classes so `except subprocess.X` works
        mock_subproc.CalledProcessError = subprocess.CalledProcessError
        mock_subproc.DEVNULL = subprocess.DEVNULL
        mock_subproc.PIPE = subprocess.PIPE

        mock_tmp = MagicMock()
        mock_tmp.name = "/tmp/test_output.wav"
        mock_tempfile.NamedTemporaryFile.return_value = mock_tmp
        safe_input = MagicMock(spec=Path)
        safe_input.parent = MagicMock(spec=Path)
        mock_copy.return_value = safe_input
        return mock_tmp, safe_input

    @staticmethod
    def _get_ffmpeg_cmd(mock_subproc):
        """Extract the ffmpeg command list from the subprocess.run mock."""
        return mock_subproc.run.call_args[0][0]

    def test_returns_wav_path_on_success(
        self, mock_subproc, mock_tempfile, mock_copy, mock_ffmpeg,
    ):
        mock_tmp, safe_input = self._setup_mocks(mock_subproc, mock_tempfile, mock_copy)
        result = convert_to_wav("/input/audio.mp3")
        assert result == Path("/tmp/test_output.wav")

    def test_calls_subprocess_with_check(
        self, mock_subproc, mock_tempfile, mock_copy, mock_ffmpeg,
    ):
        self._setup_mocks(mock_subproc, mock_tempfile, mock_copy)
        convert_to_wav("/input/audio.mp3")
        mock_subproc.run.assert_called_once()
        _, kwargs = mock_subproc.run.call_args
        assert kwargs["check"] is True

    def test_enhanced_mode_adds_filter_chain(
        self, mock_subproc, mock_tempfile, mock_copy, mock_ffmpeg,
    ):
        self._setup_mocks(mock_subproc, mock_tempfile, mock_copy)
        convert_to_wav("/input/audio.mp3", enhanced=True)
        cmd = self._get_ffmpeg_cmd(mock_subproc)
        assert "-af" in cmd
        assert ENHANCED_FILTER_CHAIN in cmd

    def test_standard_mode_omits_filter_chain(
        self, mock_subproc, mock_tempfile, mock_copy, mock_ffmpeg,
    ):
        self._setup_mocks(mock_subproc, mock_tempfile, mock_copy)
        convert_to_wav("/input/audio.mp3", enhanced=False)
        cmd = self._get_ffmpeg_cmd(mock_subproc)
        assert "-af" not in cmd

    def test_raises_runtime_error_on_ffmpeg_failure(
        self, mock_subproc, mock_tempfile, mock_copy, mock_ffmpeg,
    ):
        self._setup_mocks(mock_subproc, mock_tempfile, mock_copy)
        mock_subproc.run.side_effect = subprocess.CalledProcessError(
            1, "ffmpeg", stderr=b"conversion error details",
        )
        with pytest.raises(RuntimeError, match="ffmpeg failed to convert"):
            convert_to_wav("/input/audio.mp3")

    def test_safe_input_cleaned_up_on_success(
        self, mock_subproc, mock_tempfile, mock_copy, mock_ffmpeg,
    ):
        _, safe_input = self._setup_mocks(mock_subproc, mock_tempfile, mock_copy)
        convert_to_wav("/input/audio.mp3")
        safe_input.unlink.assert_called_once_with(missing_ok=True)

    def test_safe_input_cleaned_up_on_failure(
        self, mock_subproc, mock_tempfile, mock_copy, mock_ffmpeg,
    ):
        _, safe_input = self._setup_mocks(mock_subproc, mock_tempfile, mock_copy)
        mock_subproc.run.side_effect = subprocess.CalledProcessError(
            1, "ffmpeg", stderr=b"err",
        )
        with pytest.raises(RuntimeError):
            convert_to_wav("/input/audio.mp3")
        safe_input.unlink.assert_called_once_with(missing_ok=True)

    def test_runtime_error_chains_original_exception(
        self, mock_subproc, mock_tempfile, mock_copy, mock_ffmpeg,
    ):
        self._setup_mocks(mock_subproc, mock_tempfile, mock_copy)
        cause = subprocess.CalledProcessError(1, "ffmpeg", stderr=b"err")
        mock_subproc.run.side_effect = cause
        with pytest.raises(RuntimeError) as exc_info:
            convert_to_wav("/input/audio.mp3")
        assert exc_info.value.__cause__ is cause

    def test_cmd_includes_sample_rate_and_mono(
        self, mock_subproc, mock_tempfile, mock_copy, mock_ffmpeg,
    ):
        self._setup_mocks(mock_subproc, mock_tempfile, mock_copy)
        convert_to_wav("/input/audio.mp3")
        cmd = self._get_ffmpeg_cmd(mock_subproc)
        assert "-ar" in cmd
        assert str(SAMPLE_RATE_HZ) in cmd
        assert "-ac" in cmd
        ac_idx = cmd.index("-ac")
        assert cmd[ac_idx + 1] == "1"
