"""Tests for audio conversion module."""

import json
import subprocess
from unittest.mock import MagicMock, patch
from pathlib import Path

import pytest

from transcription_tools.audio import (
    find_ffmpeg,
    find_ffprobe,
    probe_audio_streams,
    validate_has_audio,
    classify_media_file,
    convert_to_wav,
    FFMPEG_CANDIDATES,
    FFPROBE_CANDIDATES,
    AUDIO_EXTENSIONS,
    VIDEO_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
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


class TestFindFfprobe:
    def test_returns_shutil_which_result(self):
        with patch("shutil.which", return_value="/usr/bin/ffprobe"):
            assert find_ffprobe() == "/usr/bin/ffprobe"

    def test_falls_back_to_candidates(self):
        with patch("shutil.which", return_value=None), \
             patch.object(Path, "is_file", return_value=True):
            result = find_ffprobe()
            assert result in FFPROBE_CANDIDATES

    def test_raises_when_not_found(self):
        with patch("shutil.which", return_value=None), \
             patch.object(Path, "is_file", return_value=False):
            with pytest.raises(FileNotFoundError, match="ffprobe not found"):
                find_ffprobe()


@patch("transcription_tools.audio.find_ffprobe", return_value="/usr/bin/ffprobe")
@patch("transcription_tools.audio._copy_input_to_temp")
@patch("transcription_tools.audio.subprocess")
class TestProbeAudioStreams:
    """Test ffprobe-based audio stream detection."""

    def _setup_mocks(self, mock_subproc, mock_copy):
        """Configure standard mock return values."""
        mock_subproc.CalledProcessError = subprocess.CalledProcessError
        safe_input = MagicMock(spec=Path)
        safe_input.parent = MagicMock(spec=Path)
        mock_copy.return_value = safe_input
        return safe_input

    def test_returns_audio_streams(self, mock_subproc, mock_copy, _mock_ffprobe):
        safe_input = self._setup_mocks(mock_subproc, mock_copy)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "streams": [{"codec_type": "audio", "codec_name": "aac"}]
        })
        mock_subproc.run.return_value = mock_result

        streams = probe_audio_streams("/fake/video.mp4")
        assert len(streams) == 1
        assert streams[0]["codec_name"] == "aac"

    def test_returns_empty_list_when_no_audio(self, mock_subproc, mock_copy, _):
        self._setup_mocks(mock_subproc, mock_copy)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"streams": []})
        mock_subproc.run.return_value = mock_result

        assert probe_audio_streams("/fake/silent_video.mp4") == []

    def test_raises_on_ffprobe_failure(self, mock_subproc, mock_copy, _):
        self._setup_mocks(mock_subproc, mock_copy)

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Invalid data found when processing input"
        mock_subproc.run.return_value = mock_result

        with pytest.raises(RuntimeError, match="ffprobe failed"):
            probe_audio_streams("/fake/corrupt.bin")

    def test_cleans_up_temp_file_on_success(self, mock_subproc, mock_copy, _):
        safe_input = self._setup_mocks(mock_subproc, mock_copy)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"streams": []})
        mock_subproc.run.return_value = mock_result

        probe_audio_streams("/fake/file.mp4")
        safe_input.unlink.assert_called_once_with(missing_ok=True)

    def test_cleans_up_temp_file_on_failure(self, mock_subproc, mock_copy, _):
        safe_input = self._setup_mocks(mock_subproc, mock_copy)
        mock_subproc.run.side_effect = subprocess.TimeoutExpired("ffprobe", 30)

        with pytest.raises(subprocess.TimeoutExpired):
            probe_audio_streams("/fake/file.mp4")
        safe_input.unlink.assert_called_once_with(missing_ok=True)


class TestValidateHasAudio:
    @patch("transcription_tools.audio.probe_audio_streams",
           return_value=[{"codec_type": "audio"}])
    def test_passes_when_audio_present(self, _):
        validate_has_audio("/fake/video.mp4")  # Should not raise

    @patch("transcription_tools.audio.probe_audio_streams", return_value=[])
    def test_raises_when_no_audio(self, _):
        with pytest.raises(ValueError, match="No audio stream found"):
            validate_has_audio("/fake/silent_screencast.mp4")


class TestClassifyMediaFile:
    @pytest.mark.parametrize("filename,expected", [
        ("recording.mp3", "audio"),
        ("recording.wav", "audio"),
        ("recording.flac", "audio"),
        ("recording.m4a", "audio"),
        ("recording.ogg", "audio"),
        ("recording.aiff", "audio"),
        ("lecture.mp4", "video"),
        ("lecture.mov", "video"),
        ("lecture.mkv", "video"),
        ("lecture.webm", "video"),
        ("lecture.avi", "video"),
        ("lecture.m4v", "video"),
        ("document.pdf", "unknown"),
        ("notes.txt", "unknown"),
    ])
    def test_classifies_by_extension(self, filename, expected):
        assert classify_media_file(f"/fake/{filename}") == expected

    def test_case_insensitive(self):
        assert classify_media_file("/fake/video.MP4") == "video"
        assert classify_media_file("/fake/audio.WAV") == "audio"

    def test_supported_extensions_is_union(self):
        assert SUPPORTED_EXTENSIONS == AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


@patch("transcription_tools.audio.find_ffmpeg", return_value="/usr/bin/ffmpeg")
@patch("transcription_tools.audio._copy_input_to_temp")
@patch("transcription_tools.audio.tempfile")
@patch("transcription_tools.audio.subprocess")
@patch("transcription_tools.audio.validate_has_audio")
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
        self, mock_validate, mock_subproc, mock_tempfile, mock_copy, mock_ffmpeg,
    ):
        mock_tmp, safe_input = self._setup_mocks(mock_subproc, mock_tempfile, mock_copy)
        result = convert_to_wav("/input/audio.mp3")
        assert result == Path("/tmp/test_output.wav")

    def test_calls_subprocess_with_check(
        self, mock_validate, mock_subproc, mock_tempfile, mock_copy, mock_ffmpeg,
    ):
        self._setup_mocks(mock_subproc, mock_tempfile, mock_copy)
        convert_to_wav("/input/audio.mp3")
        mock_subproc.run.assert_called_once()
        _, kwargs = mock_subproc.run.call_args
        assert kwargs["check"] is True

    def test_enhanced_mode_adds_filter_chain(
        self, mock_validate, mock_subproc, mock_tempfile, mock_copy, mock_ffmpeg,
    ):
        self._setup_mocks(mock_subproc, mock_tempfile, mock_copy)
        convert_to_wav("/input/audio.mp3", enhanced=True)
        cmd = self._get_ffmpeg_cmd(mock_subproc)
        assert "-af" in cmd
        assert ENHANCED_FILTER_CHAIN in cmd

    def test_standard_mode_omits_filter_chain(
        self, mock_validate, mock_subproc, mock_tempfile, mock_copy, mock_ffmpeg,
    ):
        self._setup_mocks(mock_subproc, mock_tempfile, mock_copy)
        convert_to_wav("/input/audio.mp3", enhanced=False)
        cmd = self._get_ffmpeg_cmd(mock_subproc)
        assert "-af" not in cmd

    def test_raises_runtime_error_on_ffmpeg_failure(
        self, mock_validate, mock_subproc, mock_tempfile, mock_copy, mock_ffmpeg,
    ):
        self._setup_mocks(mock_subproc, mock_tempfile, mock_copy)
        mock_subproc.run.side_effect = subprocess.CalledProcessError(
            1, "ffmpeg", stderr=b"conversion error details",
        )
        with pytest.raises(RuntimeError, match="ffmpeg failed to convert"):
            convert_to_wav("/input/audio.mp3")

    def test_safe_input_cleaned_up_on_success(
        self, mock_validate, mock_subproc, mock_tempfile, mock_copy, mock_ffmpeg,
    ):
        _, safe_input = self._setup_mocks(mock_subproc, mock_tempfile, mock_copy)
        convert_to_wav("/input/audio.mp3")
        safe_input.unlink.assert_called_once_with(missing_ok=True)

    def test_safe_input_cleaned_up_on_failure(
        self, mock_validate, mock_subproc, mock_tempfile, mock_copy, mock_ffmpeg,
    ):
        _, safe_input = self._setup_mocks(mock_subproc, mock_tempfile, mock_copy)
        mock_subproc.run.side_effect = subprocess.CalledProcessError(
            1, "ffmpeg", stderr=b"err",
        )
        with pytest.raises(RuntimeError):
            convert_to_wav("/input/audio.mp3")
        safe_input.unlink.assert_called_once_with(missing_ok=True)

    def test_output_temp_cleaned_up_on_failure(
        self, mock_validate, mock_subproc, mock_tempfile, mock_copy, mock_ffmpeg,
    ):
        mock_tmp, _ = self._setup_mocks(mock_subproc, mock_tempfile, mock_copy)
        mock_subproc.run.side_effect = subprocess.CalledProcessError(
            1, "ffmpeg", stderr=b"err",
        )
        with patch("transcription_tools.audio.Path") as MockPath, \
             pytest.raises(RuntimeError):
            convert_to_wav("/input/audio.mp3")
        MockPath.assert_called_once_with(mock_tmp.name)
        MockPath.return_value.unlink.assert_called_once_with(missing_ok=True)

    def test_runtime_error_chains_original_exception(
        self, mock_validate, mock_subproc, mock_tempfile, mock_copy, mock_ffmpeg,
    ):
        self._setup_mocks(mock_subproc, mock_tempfile, mock_copy)
        cause = subprocess.CalledProcessError(1, "ffmpeg", stderr=b"err")
        mock_subproc.run.side_effect = cause
        with pytest.raises(RuntimeError) as exc_info:
            convert_to_wav("/input/audio.mp3")
        assert exc_info.value.__cause__ is cause

    def test_cmd_includes_sample_rate_and_mono(
        self, mock_validate, mock_subproc, mock_tempfile, mock_copy, mock_ffmpeg,
    ):
        self._setup_mocks(mock_subproc, mock_tempfile, mock_copy)
        convert_to_wav("/input/audio.mp3")
        cmd = self._get_ffmpeg_cmd(mock_subproc)
        assert "-ar" in cmd
        assert str(SAMPLE_RATE_HZ) in cmd
        assert "-ac" in cmd
        ac_idx = cmd.index("-ac")
        assert cmd[ac_idx + 1] == "1"

    def test_cmd_includes_map_flag(
        self, mock_validate, mock_subproc, mock_tempfile, mock_copy, mock_ffmpeg,
    ):
        self._setup_mocks(mock_subproc, mock_tempfile, mock_copy)
        convert_to_wav("/input/audio.mp3")
        cmd = self._get_ffmpeg_cmd(mock_subproc)
        assert "-map" in cmd
        map_idx = cmd.index("-map")
        assert cmd[map_idx + 1] == "0:a:0"

    def test_cmd_does_not_include_vn_flag(
        self, mock_validate, mock_subproc, mock_tempfile, mock_copy, mock_ffmpeg,
    ):
        self._setup_mocks(mock_subproc, mock_tempfile, mock_copy)
        convert_to_wav("/input/audio.mp3")
        cmd = self._get_ffmpeg_cmd(mock_subproc)
        assert "-vn" not in cmd

    def test_raises_value_error_when_no_audio(
        self, mock_validate, mock_subproc, mock_tempfile, mock_copy, mock_ffmpeg,
    ):
        self._setup_mocks(mock_subproc, mock_tempfile, mock_copy)
        mock_validate.side_effect = ValueError("No audio stream found")
        with pytest.raises(ValueError, match="No audio stream"):
            convert_to_wav("/input/silent_video.mp4")
        mock_subproc.run.assert_not_called()
