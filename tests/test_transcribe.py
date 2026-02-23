"""Tests for the transcribe module."""

import signal
import sys
from unittest.mock import MagicMock, patch

import pytest

from transcription_tools.config import (
    FasterWhisperParams,
    OpenAIWhisperParams,
    TranscriptionTier,
)
from transcription_tools.transcribe import (
    _detect_ctranslate2_device,
    _detect_torch_device,
    _graceful_exit_handler,
    _timed_transcription,
    transcribe,
)


# -- Fixtures ----------------------------------------------------------------

@pytest.fixture
def faster_whisper_tier():
    return TranscriptionTier(
        name="test_fw",
        label="Test FW",
        whisper_model="tiny",
        backend_params=FasterWhisperParams(),
    )


@pytest.fixture
def openai_whisper_tier():
    return TranscriptionTier(
        name="test_oai",
        label="Test OAI",
        whisper_model="tiny",
        backend_params=OpenAIWhisperParams(),
    )


# -- Device detection: ctranslate2 -------------------------------------------

class TestDetectCtranslate2Device:
    """Test the CTranslate2 CUDA detection function."""

    def test_returns_cuda_when_available(self):
        mock_ct2 = MagicMock()
        mock_ct2.get_supported_compute_types.return_value = ["float32", "int8"]
        with patch.dict(sys.modules, {"ctranslate2": mock_ct2}):
            assert _detect_ctranslate2_device() == "cuda"

    def test_returns_cpu_when_no_cuda(self):
        mock_ct2 = MagicMock()
        mock_ct2.get_supported_compute_types.return_value = []
        with patch.dict(sys.modules, {"ctranslate2": mock_ct2}):
            assert _detect_ctranslate2_device() == "cpu"

    def test_returns_cpu_when_not_installed(self):
        with patch.dict(sys.modules, {"ctranslate2": None}):
            assert _detect_ctranslate2_device() == "cpu"

    def test_returns_cpu_on_runtime_error(self):
        mock_ct2 = MagicMock()
        mock_ct2.get_supported_compute_types.side_effect = RuntimeError("no CUDA")
        with patch.dict(sys.modules, {"ctranslate2": mock_ct2}):
            assert _detect_ctranslate2_device() == "cpu"


# -- Device detection: torch -------------------------------------------------

class TestDetectTorchDevice:
    """Test the torch device detection function."""

    @pytest.mark.parametrize("cuda,mps,expected", [
        pytest.param(True, False, "cuda", id="cuda-available"),
        pytest.param(False, True, "mps", id="mps-available"),
        pytest.param(False, False, "cpu", id="cpu-fallback"),
    ])
    def test_returns_best_device(self, cuda, mps, expected):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = cuda
        mock_torch.backends.mps.is_available.return_value = mps
        with patch.dict(sys.modules, {"torch": mock_torch}):
            assert _detect_torch_device() == expected

    def test_returns_cpu_when_not_installed(self):
        with patch.dict(sys.modules, {"torch": None}):
            assert _detect_torch_device() == "cpu"


# -- Dispatch routing --------------------------------------------------------

class TestTranscribeDispatch:
    """Test that transcribe() routes to the correct backend."""

    @patch("transcription_tools.transcribe.transcribe_faster_whisper", return_value="fw text")
    @patch("transcription_tools.transcribe._detect_ctranslate2_device", return_value="cpu")
    def test_routes_to_faster_whisper(self, _mock_device, mock_fw, faster_whisper_tier):
        result = transcribe("/fake/audio.wav", faster_whisper_tier)
        mock_fw.assert_called_once_with("/fake/audio.wav", faster_whisper_tier, "cpu")
        assert result == "fw text"

    @patch("transcription_tools.transcribe.transcribe_openai_whisper", return_value="oai text")
    @patch("transcription_tools.transcribe._detect_torch_device", return_value="cpu")
    def test_routes_to_openai_whisper(self, _mock_device, mock_oai, openai_whisper_tier):
        result = transcribe("/fake/audio.wav", openai_whisper_tier)
        mock_oai.assert_called_once_with("/fake/audio.wav", openai_whisper_tier, "cpu")
        assert result == "oai text"

    @patch("transcription_tools.transcribe.transcribe_faster_whisper", return_value="text")
    @patch("transcription_tools.transcribe._detect_ctranslate2_device", return_value="cpu")
    def test_faster_whisper_calls_correct_device_detector(self, mock_ct2, _mock_fw, faster_whisper_tier):
        transcribe("/fake/audio.wav", faster_whisper_tier)
        mock_ct2.assert_called_once()


# -- Timing context manager --------------------------------------------------

class TestTimedTranscription:
    """Test the timing/logging context manager."""

    def test_prints_start_and_completion(self, capsys):
        with _timed_transcription("Test Mode"):
            pass
        output = capsys.readouterr().out
        assert "Transcribing in Test Mode mode..." in output
        assert "Transcription completed in" in output

    def test_records_elapsed_time(self):
        with _timed_transcription("Test") as timing:
            pass
        assert timing.elapsed >= 0

    def test_prints_aborted_on_exception(self, capsys):
        with pytest.raises(ValueError):
            with _timed_transcription("Test"):
                raise ValueError("boom")
        output = capsys.readouterr().out
        assert "Transcription aborted in" in output


# -- Signal handler ----------------------------------------------------------

class TestGracefulExitHandler:
    """Test the signal handler context manager."""

    def test_restores_original_handlers(self):
        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)
        with _graceful_exit_handler():
            pass
        assert signal.getsignal(signal.SIGINT) is original_sigint
        assert signal.getsignal(signal.SIGTERM) is original_sigterm

    def test_installs_custom_handler_inside_context(self):
        original_sigint = signal.getsignal(signal.SIGINT)
        with _graceful_exit_handler():
            current = signal.getsignal(signal.SIGINT)
            assert current is not original_sigint
