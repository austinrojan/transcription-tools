"""Integration tests for audio/video conversion.

These tests require ffmpeg to be installed and create real (tiny) media
files using ffmpeg's lavfi virtual device. Skip automatically when
ffmpeg is not available.
"""

import shutil
import subprocess

import pytest

from transcription_tools.audio import convert_to_wav, probe_audio_streams

requires_ffmpeg = pytest.mark.skipif(
    not shutil.which("ffmpeg"),
    reason="ffmpeg not installed",
)


@pytest.fixture(scope="session")
def tiny_audio(tmp_path_factory):
    """Generate a 0.1-second silent WAV file (~3 KB)."""
    path = tmp_path_factory.mktemp("media") / "test_audio.wav"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "anullsrc=r=8000:cl=mono",
            "-t", "0.1", "-f", "wav", str(path),
        ],
        check=True, capture_output=True,
    )
    return path


@pytest.fixture(scope="session")
def tiny_video_with_audio(tmp_path_factory):
    """Generate a 0.1-second MP4 with a black frame and silent audio (~4 KB)."""
    path = tmp_path_factory.mktemp("media") / "test_video.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=2x2:d=0.1:r=1",
            "-f", "lavfi", "-i", "anullsrc=r=8000:cl=mono",
            "-t", "0.1",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac", "-b:a", "32k",
            "-shortest", str(path),
        ],
        check=True, capture_output=True,
    )
    return path


@pytest.fixture(scope="session")
def tiny_video_no_audio(tmp_path_factory):
    """Generate a 0.1-second MP4 with video only, no audio stream."""
    path = tmp_path_factory.mktemp("media") / "silent_video.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=2x2:d=0.1:r=1",
            "-t", "0.1",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-an", str(path),
        ],
        check=True, capture_output=True,
    )
    return path


@requires_ffmpeg
class TestProbeAudioStreamsIntegration:

    def test_detects_audio_in_wav(self, tiny_audio):
        streams = probe_audio_streams(str(tiny_audio))
        assert len(streams) >= 1

    def test_detects_audio_in_video(self, tiny_video_with_audio):
        streams = probe_audio_streams(str(tiny_video_with_audio))
        assert len(streams) >= 1

    def test_detects_no_audio_in_silent_video(self, tiny_video_no_audio):
        streams = probe_audio_streams(str(tiny_video_no_audio))
        assert streams == []


@requires_ffmpeg
class TestConvertToWavIntegration:

    def test_converts_audio_file(self, tiny_audio):
        wav = convert_to_wav(str(tiny_audio))
        try:
            assert wav.exists()
            assert wav.suffix == ".wav"
            assert wav.stat().st_size > 0
        finally:
            wav.unlink(missing_ok=True)

    def test_converts_video_file(self, tiny_video_with_audio):
        wav = convert_to_wav(str(tiny_video_with_audio))
        try:
            assert wav.exists()
            assert wav.suffix == ".wav"
            assert wav.stat().st_size > 0
        finally:
            wav.unlink(missing_ok=True)

    def test_rejects_video_without_audio(self, tiny_video_no_audio):
        with pytest.raises(ValueError, match="No audio stream"):
            convert_to_wav(str(tiny_video_no_audio))
