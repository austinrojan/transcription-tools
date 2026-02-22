"""Audio and video conversion via ffmpeg.

Extracts and converts audio from any media file (audio or video) to
16kHz mono WAV for Whisper consumption. Uses ffprobe to validate that
the input contains an audio stream before conversion. Optionally
applies audio enhancement filters for the veryslow tier.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

FFMPEG_CANDIDATES = [
    "/opt/homebrew/bin/ffmpeg",
    "/usr/local/bin/ffmpeg",
]

FFPROBE_CANDIDATES = [
    "/opt/homebrew/bin/ffprobe",
    "/usr/local/bin/ffprobe",
]

SAMPLE_RATE_HZ = 16000
HIGHPASS_FREQUENCY_HZ = 100
LOWPASS_FREQUENCY_HZ = 8000
DENOISE_STRENGTH = 7
VOLUME_BOOST = 1.5

ENHANCED_FILTER_CHAIN = (
    f"highpass=f={HIGHPASS_FREQUENCY_HZ},"
    f"lowpass=f={LOWPASS_FREQUENCY_HZ},"
    f"anlmdn=s={DENOISE_STRENGTH},"
    "compand=attacks=0:points=-80/-900|-45/-15|-27/-9|-5/-5|20/20,"
    f"volume={VOLUME_BOOST}"
)

AUDIO_EXTENSIONS = frozenset({
    ".mp3", ".wav", ".flac", ".aac", ".ogg", ".wma", ".m4a",
    ".opus", ".aiff", ".ape", ".wv", ".oga",
})

VIDEO_EXTENSIONS = frozenset({
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".wmv", ".flv",
    ".mpeg", ".mpg", ".m4v", ".ts", ".3gp", ".ogv", ".vob",
    ".mts", ".m2ts",
})

SUPPORTED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


def classify_media_file(file_path: str) -> str:
    """Classify a file as 'audio', 'video', or 'unknown' by extension.

    This is a fast heuristic for logging and error messages. The
    definitive check for audio content is probe_audio_streams().
    """
    ext = Path(file_path).suffix.lower()
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    return "unknown"


def find_ffmpeg() -> str:
    """Locate the ffmpeg binary on this system."""
    path = shutil.which("ffmpeg")
    if path:
        return path
    for candidate in FFMPEG_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    raise FileNotFoundError(
        "ffmpeg not found. Install it with: brew install ffmpeg"
    )


def find_ffprobe() -> str:
    """Locate the ffprobe binary on this system."""
    path = shutil.which("ffprobe")
    if path:
        return path
    for candidate in FFPROBE_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    raise FileNotFoundError(
        "ffprobe not found. Install it with: brew install ffmpeg"
    )


def probe_audio_streams(file_path: str) -> list[dict]:
    """Return metadata for all audio streams in a media file.

    Uses ffprobe to inspect the file without decoding any media data.
    Returns an empty list if no audio streams are found.

    Raises:
        FileNotFoundError: If ffprobe is not installed.
        RuntimeError: If ffprobe exits with a non-zero code (corrupt or
                      unreadable file).
    """
    ffprobe = find_ffprobe()
    safe_input = _copy_input_to_temp(file_path)
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                "-select_streams", "a",
                str(safe_input),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    finally:
        safe_input.unlink(missing_ok=True)
        try:
            safe_input.parent.rmdir()
        except OSError:
            pass

    if result.returncode != 0:
        detail = (result.stderr or "")[-500:]
        raise RuntimeError(
            f"ffprobe failed to read '{file_path}': {detail}"
        )

    data = json.loads(result.stdout)
    return data.get("streams", [])


def validate_has_audio(file_path: str) -> None:
    """Raise ValueError if the file contains no audio streams.

    Call this before convert_to_wav() to produce a clear error message
    instead of a cryptic ffmpeg failure.
    """
    streams = probe_audio_streams(file_path)
    if not streams:
        raise ValueError(
            f"No audio stream found in '{file_path}'. "
            "The file may be a silent video or a non-media file."
        )


def _copy_input_to_temp(input_path: str) -> Path:
    """Copy input file to a temp directory so ffmpeg can read it.

    On macOS, Automator Quick Actions lack TCC authorization for
    protected directories (Desktop, Documents, Downloads).  The
    workflow shell script *can* read the file (Finder grants access),
    so we copy it to /tmp where ffmpeg has unrestricted access.
    """
    src = Path(input_path)
    tmp_input = Path(tempfile.mkdtemp()) / src.name
    shutil.copy2(str(src), str(tmp_input))
    return tmp_input


def convert_to_wav(input_path: str, enhanced: bool = False) -> Path:
    """Convert a media file to 16kHz mono WAV.

    Args:
        input_path: Path to the source audio or video file.
        enhanced: If True, apply noise reduction and dynamic range
                  compression filters (used by the veryslow tier).

    Returns:
        Path to the temporary WAV file. Caller is responsible for cleanup.
    """
    validate_has_audio(input_path)
    ffmpeg = find_ffmpeg()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    tmp_path = tmp.name

    # Copy to temp to avoid macOS TCC restrictions in Automator context
    safe_input = _copy_input_to_temp(input_path)

    result_path = None
    try:
        cmd = [ffmpeg, "-y", "-i", str(safe_input), "-map", "0:a:0"]

        if enhanced:
            cmd += ["-af", ENHANCED_FILTER_CHAIN]

        cmd += ["-ar", str(SAMPLE_RATE_HZ), "-ac", "1", "-f", "wav", tmp_path]

        subprocess.run(
            cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        result_path = Path(tmp_path)
    except subprocess.CalledProcessError as e:
        detail = e.stderr.decode(errors="replace")[-1000:] if e.stderr else ""
        raise RuntimeError(f"ffmpeg failed to convert {input_path}: {detail}") from e
    finally:
        safe_input.unlink(missing_ok=True)
        try:
            safe_input.parent.rmdir()
        except OSError:
            pass
        if result_path is None:
            Path(tmp_path).unlink(missing_ok=True)

    return result_path
