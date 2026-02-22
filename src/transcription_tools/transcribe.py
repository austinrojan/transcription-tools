"""Whisper transcription with dual-backend dispatch.

Supports both faster-whisper (CTranslate2) and OpenAI whisper backends.
The tier config determines which backend and parameters to use.
"""

from __future__ import annotations

import os
import signal
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass

from transcription_tools.config import FasterWhisperParams, TranscriptionTier

# -- Timing helpers ------------------------------------------------------


@dataclass
class _TimingResult:
    """Elapsed time recorded by _timed_transcription."""

    elapsed: float = 0.0


@contextmanager
def _timed_transcription(tier_label: str):
    """Print transcription start/end and measure elapsed time.

    Yields a _TimingResult whose ``elapsed`` field is populated on exit.
    Callers can read it after the ``with`` block for backend-specific logging.
    """
    result = _TimingResult()
    print(f"Transcribing in {tier_label} mode...")
    start = time.time()
    success = False
    try:
        yield result
        success = True
    finally:
        result.elapsed = time.time() - start
        status = "completed" if success else "aborted"
        print(f"Transcription {status} in {result.elapsed:.1f}s ({result.elapsed / 60:.1f} min)")


@contextmanager
def _graceful_exit_handler():
    """Install clean-exit signal handlers; restore originals on exit."""
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)

    def handler(sig, frame):
        print("\n[INTERRUPTED] Exiting gracefully...")
        sys.exit(130)  # 128 + SIGINT(2): standard interrupted-by-signal exit code

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)


# -- Device detection ----------------------------------------------------


def _detect_ctranslate2_device() -> str:
    """Return 'cuda' if ctranslate2 supports it, otherwise 'cpu'.

    IMPORTANT: this function must be called BEFORE torch is imported.
    On macOS, importing torch before ctranslate2 causes an OpenMP
    runtime conflict that segfaults the process.
    """
    try:
        import ctranslate2
        if "cuda" in ctranslate2.get_supported_compute_types("cuda"):
            return "cuda"
    except (ImportError, RuntimeError, ValueError):
        pass
    return "cpu"


def _detect_torch_device() -> str:
    """Return the best torch-compatible device: 'cuda', 'mps', or 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# -- Backend implementations ---------------------------------------------


def transcribe_faster_whisper(audio_path: str, tier: TranscriptionTier, device: str) -> str:
    """Transcribe using the faster-whisper (CTranslate2) backend."""
    from faster_whisper import WhisperModel

    params = tier.backend_params  # FasterWhisperParams
    compute_type = params.compute_type_gpu if device == "cuda" else params.compute_type_cpu

    print(f"[{tier.label}] device={device}, compute_type={compute_type}")
    print(f"Loading faster-whisper model '{tier.whisper_model}'...")

    model = WhisperModel(
        tier.whisper_model,
        device=device,
        compute_type=compute_type,
        download_root=os.path.expanduser("~/.cache"),
    )

    kwargs: dict = {
        "language": params.language,
        "beam_size": tier.beam_size,
        "best_of": tier.best_of,
        "temperature": tier.temperature,
        "condition_on_previous_text": tier.condition_on_previous_text,
        "without_timestamps": params.without_timestamps,
        "vad_filter": params.vad_filter,
    }
    if params.vad_params:
        kwargs["vad_parameters"] = dict(params.vad_params)

    with _timed_transcription(tier.label) as timing:
        segments, info = model.transcribe(audio_path, **kwargs)
        text = " ".join(seg.text.strip() for seg in segments).strip()

    if hasattr(info, "duration") and info.duration and timing.elapsed > 0:
        print(f"Audio duration: {info.duration:.1f}s ({info.duration / timing.elapsed:.1f}x realtime)")

    return text


def transcribe_openai_whisper(audio_path: str, tier: TranscriptionTier, device: str) -> str:
    """Transcribe using the OpenAI whisper backend."""
    from contextlib import nullcontext
    import whisper

    params = tier.backend_params  # OpenAIWhisperParams
    context = _graceful_exit_handler() if params.signal_handling else nullcontext()

    with context:
        print(f"[{tier.label}] device={device}")
        print(f"Loading OpenAI Whisper model '{tier.whisper_model}'...")

        model = whisper.load_model(
            tier.whisper_model,
            device=device,
            download_root=os.path.expanduser("~/.cache/whisper"),
        )

        with _timed_transcription(tier.label) as timing:
            result = model.transcribe(
                audio_path,
                task="transcribe",
                temperature=tier.temperature,
                beam_size=tier.beam_size,
                best_of=tier.best_of,
                condition_on_previous_text=tier.condition_on_previous_text,
                initial_prompt=params.initial_prompt,
                verbose=params.verbose,
                fp16=(device == "cuda" and params.fp16_on_gpu),
                # Fixed decode parameters (not configurable per-tier).
                patience=1.0,
                length_penalty=1.0,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
                suppress_tokens="-1",
            )
            text = (result.get("text") or "").strip()

        return text


# -- Public API ----------------------------------------------------------


def transcribe(audio_path: str, tier: TranscriptionTier) -> str:
    """Dispatch to the correct Whisper backend based on tier config."""
    if isinstance(tier.backend_params, FasterWhisperParams):
        device = _detect_ctranslate2_device()
        return transcribe_faster_whisper(audio_path, tier, device)

    device = _detect_torch_device()
    return transcribe_openai_whisper(audio_path, tier, device)
