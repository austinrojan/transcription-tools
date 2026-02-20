"""Audio conversion via ffmpeg.

Converts input audio to 16kHz mono WAV for Whisper consumption.
Optionally applies audio enhancement filters for the veryslow tier.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

FFMPEG_CANDIDATES = [
    "/opt/homebrew/bin/ffmpeg",
    "/usr/local/bin/ffmpeg",
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


def convert_to_wav(input_path: str, enhanced: bool = False) -> Path:
    """Convert an audio file to 16kHz mono WAV.

    Args:
        input_path: Path to the source audio file.
        enhanced: If True, apply noise reduction and dynamic range
                  compression filters (used by the veryslow tier).

    Returns:
        Path to the temporary WAV file. Caller is responsible for cleanup.
    """
    ffmpeg = find_ffmpeg()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    tmp_path = tmp.name

    try:
        cmd = [ffmpeg, "-y", "-i", input_path]

        if enhanced:
            cmd += ["-af", ENHANCED_FILTER_CHAIN]

        cmd += ["-ar", str(SAMPLE_RATE_HZ), "-ac", "1", "-vn", "-f", "wav", tmp_path]

        subprocess.run(
            cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        Path(tmp_path).unlink(missing_ok=True)
        detail = e.stderr.decode(errors="replace")[:500] if e.stderr else ""
        raise RuntimeError(f"ffmpeg failed to convert {input_path}: {detail}")
    except Exception:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError:
            pass
        raise

    return Path(tmp_path)
