"""Tier configuration for transcription tools.

Each tier defines which Whisper backend to use and its exact parameters.
This is the single source of truth — no parameters are hardcoded elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal


@dataclass(frozen=True)
class TranscriptionTier:
    name: str
    label: str
    backend: Literal["faster_whisper", "whisper"]
    whisper_model: str

    # faster-whisper parameters
    beam_size: int = 1
    best_of: int = 1
    temperature: float = 0.0
    language: str | None = None
    vad_filter: bool = False
    vad_params: MappingProxyType | None = None
    condition_on_previous_text: bool = False
    without_timestamps: bool = True
    compute_type_gpu: str = "int8_float16"
    compute_type_cpu: str = "int8"

    # OpenAI whisper parameters
    initial_prompt: str | None = None
    verbose: bool = False
    fp16_on_gpu: bool = True

    # veryslow extras
    enhanced_audio: bool = False
    signal_handling: bool = False
    save_backup: bool = False


TIERS: dict[str, TranscriptionTier] = {
    "veryfast": TranscriptionTier(
        name="veryfast",
        label="Very Fast",
        backend="faster_whisper",
        whisper_model="tiny.en",
        beam_size=1,
        best_of=1,
        temperature=0.0,
        language="en",
        vad_filter=False,
        condition_on_previous_text=False,
        without_timestamps=True,
        compute_type_gpu="int8_float16",
        compute_type_cpu="int8",
    ),
    "fast": TranscriptionTier(
        name="fast",
        label="Fast",
        backend="faster_whisper",
        whisper_model="base",
        beam_size=3,
        best_of=2,
        temperature=0.0,
        language=None,
        vad_filter=True,
        vad_params=MappingProxyType({
            "min_speech_duration_ms": 250,
            "max_speech_duration_s": float("inf"),
            "min_silence_duration_ms": 1000,
            "speech_pad_ms": 200,
        }),
        condition_on_previous_text=True,
        without_timestamps=False,
        compute_type_gpu="int8_float16",
        compute_type_cpu="int8",
    ),
    "medium": TranscriptionTier(
        name="medium",
        label="Medium",
        backend="faster_whisper",
        whisper_model="medium",
        beam_size=5,
        best_of=5,
        temperature=0.0,
        language=None,
        vad_filter=True,
        vad_params=MappingProxyType({
            "min_speech_duration_ms": 250,
            "max_speech_duration_s": float("inf"),
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 400,
        }),
        condition_on_previous_text=True,
        without_timestamps=False,
        compute_type_gpu="float16",
        compute_type_cpu="int8",
    ),
    "slow": TranscriptionTier(
        name="slow",
        label="Slow",
        backend="whisper",
        whisper_model="medium",
        beam_size=3,
        best_of=3,
        temperature=0.1,
        condition_on_previous_text=True,
        verbose=False,
        fp16_on_gpu=True,
    ),
    "veryslow": TranscriptionTier(
        name="veryslow",
        label="Very Slow",
        backend="whisper",
        whisper_model="large-v3",
        beam_size=5,
        best_of=5,
        temperature=0.1,
        condition_on_previous_text=True,
        initial_prompt=(
            "This is a technical presentation or tutorial. Common terms include: "
            "Subsplash, Vimeo, YouTube, WordPress, API, REST API, JSON, CSV, XML, "
            "FTP, SFTP, SSH, upload, download, streaming, encoding, transcoding, "
            "MP3, MP4, MOV, WAV, H.264, bitrate, resolution, 1080p, 4K, "
            "dashboard, admin, interface, button, click, select, navigate, menu, "
            "settings, configuration, database, server, cloud, AWS, CDN."
        ),
        verbose=True,
        fp16_on_gpu=True,
        enhanced_audio=True,
        signal_handling=True,
        save_backup=True,
    ),
}

# Cleanup model configuration — single source of truth for cli.py and cleanup.py
DEFAULT_CLEANUP_MODEL = "gpt-5-nano"
ALLOWED_CLEANUP_MODELS = frozenset({"gpt-5-nano", "gpt-5-mini", "gpt-5"})
