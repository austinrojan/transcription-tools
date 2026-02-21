"""Tier configuration for transcription tools.

Each tier defines which Whisper backend to use and its exact parameters.
This is the single source of truth — no parameters are hardcoded elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class FasterWhisperParams:
    """Parameters specific to the faster-whisper (CTranslate2) backend."""
    language: str | None = None
    vad_filter: bool = False
    vad_params: MappingProxyType | None = None
    without_timestamps: bool = True
    compute_type_gpu: str = "int8_float16"
    compute_type_cpu: str = "int8"


@dataclass(frozen=True)
class OpenAIWhisperParams:
    """Parameters specific to the OpenAI whisper backend."""
    initial_prompt: str | None = None
    verbose: bool = False
    fp16_on_gpu: bool = True
    signal_handling: bool = False


@dataclass(frozen=True)
class TranscriptionTier:
    """A complete transcription configuration.

    Shared parameters live here directly. Backend-specific parameters
    live in backend_params — use isinstance() to determine which backend.
    """
    name: str
    label: str
    whisper_model: str
    backend_params: FasterWhisperParams | OpenAIWhisperParams

    # Shared across both backends
    beam_size: int = 1
    best_of: int = 1
    temperature: float = 0.0
    condition_on_previous_text: bool = False

    # Output behavior
    enhanced_audio: bool = False
    save_backup: bool = False


TIERS: dict[str, TranscriptionTier] = {
    "veryfast": TranscriptionTier(
        name="veryfast",
        label="Very Fast",
        whisper_model="tiny.en",
        backend_params=FasterWhisperParams(language="en"),
    ),
    "fast": TranscriptionTier(
        name="fast",
        label="Fast",
        whisper_model="base",
        backend_params=FasterWhisperParams(
            vad_filter=True,
            vad_params=MappingProxyType({
                "min_speech_duration_ms": 250,
                "max_speech_duration_s": float("inf"),
                "min_silence_duration_ms": 1000,
                "speech_pad_ms": 200,
            }),
            without_timestamps=False,
        ),
        beam_size=3,
        best_of=2,
        condition_on_previous_text=True,
    ),
    "medium": TranscriptionTier(
        name="medium",
        label="Medium",
        whisper_model="medium",
        backend_params=FasterWhisperParams(
            vad_filter=True,
            vad_params=MappingProxyType({
                "min_speech_duration_ms": 250,
                "max_speech_duration_s": float("inf"),
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 400,
            }),
            without_timestamps=False,
            compute_type_gpu="float16",
        ),
        beam_size=5,
        best_of=5,
        condition_on_previous_text=True,
    ),
    "slow": TranscriptionTier(
        name="slow",
        label="Slow",
        whisper_model="medium",
        backend_params=OpenAIWhisperParams(),
        beam_size=3,
        best_of=3,
        temperature=0.1,
        condition_on_previous_text=True,
    ),
    "veryslow": TranscriptionTier(
        name="veryslow",
        label="Very Slow",
        whisper_model="large-v3",
        backend_params=OpenAIWhisperParams(
            initial_prompt=(
                "This is a technical presentation or tutorial. Common terms include: "
                "Subsplash, Vimeo, YouTube, WordPress, API, REST API, JSON, CSV, XML, "
                "FTP, SFTP, SSH, upload, download, streaming, encoding, transcoding, "
                "MP3, MP4, MOV, WAV, H.264, bitrate, resolution, 1080p, 4K, "
                "dashboard, admin, interface, button, click, select, navigate, menu, "
                "settings, configuration, database, server, cloud, AWS, CDN."
            ),
            verbose=True,
            signal_handling=True,
        ),
        beam_size=5,
        best_of=5,
        temperature=0.1,
        condition_on_previous_text=True,
        enhanced_audio=True,
        save_backup=True,
    ),
}

# Cleanup model configuration — single source of truth for cli.py and cleanup.py
DEFAULT_CLEANUP_MODEL = "gpt-5-nano"
ALLOWED_CLEANUP_MODELS = frozenset({"gpt-5-nano", "gpt-5-mini", "gpt-5"})
