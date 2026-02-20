"""Shared CLI entry point for all transcription tiers."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Must be set before torch/ctranslate2 are imported anywhere
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from transcription_tools.audio import convert_to_wav
from transcription_tools.cleanup import TranscriptCleaner
from transcription_tools.config import (
    ALLOWED_CLEANUP_MODELS,
    DEFAULT_CLEANUP_MODEL,
    TIERS,
    TranscriptionTier,
)
from transcription_tools.transcribe import transcribe


def _parse_args(tier: TranscriptionTier) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Transcribe audio — {tier.label} tier ({tier.backend}, {tier.whisper_model} model)."
    )
    parser.add_argument("input_file", help="Path to the input audio file.")
    cleanup_group = parser.add_mutually_exclusive_group()
    cleanup_group.add_argument(
        "--no-cleanup", action="store_true",
        help="Skip the OpenAI transcript cleanup pass.",
    )
    cleanup_group.add_argument(
        "--cleanup-only", action="store_true",
        help="Skip transcription; only run cleanup on an existing transcript.",
    )
    parser.add_argument(
        "--openai-model", default=None,
        help=f"OpenAI model for cleanup (allowed: {', '.join(sorted(ALLOWED_CLEANUP_MODELS))}).",
    )
    parser.add_argument(
        "--openai-base-url", default=None,
        help="Custom OpenAI-compatible base URL.",
    )
    return parser.parse_args()


def _resolve_cleanup_model(args: argparse.Namespace) -> str:
    """Determine which OpenAI model to use for cleanup."""
    model = args.openai_model or os.environ.get("OPENAI_MODEL", DEFAULT_CLEANUP_MODEL)
    if model not in ALLOWED_CLEANUP_MODELS:
        print(f"Error: OpenAI model must be one of {sorted(ALLOWED_CLEANUP_MODELS)}; got '{model}'")
        sys.exit(1)
    return model


def _run_transcription(input_path: Path, tier: TranscriptionTier, output_path: Path) -> None:
    """Convert audio and transcribe, saving result to output_path."""
    print(f"Converting '{input_path.name}' to 16kHz mono WAV...")
    wav_path = convert_to_wav(str(input_path), enhanced=tier.enhanced_audio)
    try:
        text = transcribe(str(wav_path), tier)
        output_path.write_text(text + "\n", encoding="utf-8")
        print(f"Transcript saved to '{output_path}'.")

        if tier.save_backup:
            backup = output_path.with_suffix(".backup.txt")
            backup.write_text(text + "\n", encoding="utf-8")
            print(f"Backup saved to '{backup}'.")
    finally:
        wav_path.unlink(missing_ok=True)


def _run_cleanup(raw_text: str, model: str, base_url: str | None, output_path: Path) -> None:
    """Run OpenAI cleanup on transcript text."""
    print(f"[cleanup] Using OpenAI model: {model}")
    cleaner = TranscriptCleaner(model=model, base_url=base_url)
    cleaned = cleaner.clean(raw_text)
    clean_path = output_path.with_suffix(".clean.txt")
    clean_path.write_text(cleaned.strip() + "\n", encoding="utf-8")
    print(f"Cleaned transcript saved to '{clean_path}'.")


def run(tier_name: str) -> None:
    """Main workflow for a single transcription tier."""
    # Workaround: HF Hub creates symlinks that break on some macOS configurations
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

    tier = TIERS[tier_name]
    args = _parse_args(tier)

    input_path = Path(args.input_file)
    if not input_path.is_file():
        print(f"Error: File '{input_path}' does not exist.")
        sys.exit(1)

    output_path = input_path.with_name(f"{input_path.stem}_{tier.name}.txt")

    if args.cleanup_only:
        if not output_path.is_file():
            print(f"Error: Transcript '{output_path}' does not exist for cleanup-only mode.")
            sys.exit(1)
        print(f"[CLEANUP-ONLY] Using existing file: {output_path}")
    else:
        _run_transcription(input_path, tier, output_path)

    if args.no_cleanup:
        return

    model = _resolve_cleanup_model(args)
    base_url = args.openai_base_url or os.environ.get("OPENAI_BASE_URL")

    try:
        raw_text = output_path.read_text(encoding="utf-8")
        _run_cleanup(raw_text, model, base_url, output_path)
    except Exception as e:
        print(f"Cleanup step failed: {e}", file=sys.stderr)
        sys.exit(1)


def veryfast() -> None:
    run("veryfast")

def fast() -> None:
    run("fast")

def medium() -> None:
    run("medium")

def slow() -> None:
    run("slow")

def veryslow() -> None:
    run("veryslow")
