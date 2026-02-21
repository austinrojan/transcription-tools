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

from transcription_tools.config import FasterWhisperParams, OpenAIWhisperParams, TranscriptionTier

# OpenAI whisper decode parameters (not configurable per-tier)
WHISPER_PATIENCE = 1.0
WHISPER_LENGTH_PENALTY = 1.0
WHISPER_COMPRESSION_RATIO_THRESHOLD = 2.4
WHISPER_LOGPROB_THRESHOLD = -1.0
WHISPER_NO_SPEECH_THRESHOLD = 0.6
WHISPER_SUPPRESS_TOKENS = "-1"


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


def _detect_device(backend: str = "faster_whisper") -> str:
    """Return the best available compute device.

    For faster-whisper: use ctranslate2's own CUDA check to avoid importing
    torch first — torch must not be imported before ctranslate2 on macOS or
    the process will segfault due to conflicting OpenMP runtimes.

    For OpenAI whisper: use torch directly, including MPS on Apple Silicon.
    """
    if backend == "faster_whisper":
        try:
            import ctranslate2
            if "cuda" in ctranslate2.get_supported_compute_types("cuda"):
                return "cuda"
        except (ImportError, RuntimeError, ValueError):
            pass
        return "cpu"

    # OpenAI whisper backend — safe to use torch
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


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

    print(f"Transcribing in {tier.label} mode...")
    start = time.time()

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

    segments, info = model.transcribe(audio_path, **kwargs)
    text = " ".join(seg.text.strip() for seg in segments).strip()

    elapsed = time.time() - start
    print(f"Transcription completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    if hasattr(info, "duration") and info.duration and elapsed > 0:
        print(f"Audio duration: {info.duration:.1f}s ({info.duration / elapsed:.1f}x realtime)")

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

        print(f"Transcribing in {tier.label} mode...")
        start = time.time()

        result = model.transcribe(
            audio_path,
            task="transcribe",
            temperature=tier.temperature,
            beam_size=tier.beam_size,
            best_of=tier.best_of,
            patience=WHISPER_PATIENCE,
            length_penalty=WHISPER_LENGTH_PENALTY,
            compression_ratio_threshold=WHISPER_COMPRESSION_RATIO_THRESHOLD,
            logprob_threshold=WHISPER_LOGPROB_THRESHOLD,
            no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD,
            condition_on_previous_text=tier.condition_on_previous_text,
            initial_prompt=params.initial_prompt,
            suppress_tokens=WHISPER_SUPPRESS_TOKENS,
            verbose=params.verbose,
            fp16=(device == "cuda" and params.fp16_on_gpu),
        )

        text = (result.get("text") or "").strip()
        elapsed = time.time() - start
        print(f"Transcription completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
        return text


def transcribe(audio_path: str, tier: TranscriptionTier) -> str:
    """Dispatch to the correct Whisper backend based on tier config."""
    if isinstance(tier.backend_params, FasterWhisperParams):
        device = _detect_device(backend="faster_whisper")
        return transcribe_faster_whisper(audio_path, tier, device)
    else:
        device = _detect_device(backend="whisper")
        return transcribe_openai_whisper(audio_path, tier, device)
