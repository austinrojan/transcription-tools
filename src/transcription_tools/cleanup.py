"""OpenAI-powered transcript cleanup.

Sends raw Whisper output to a chat model for spelling, grammar, and
formatting fixes. Handles chunking, rate limiting with exponential
backoff, and quality validation.
"""

from __future__ import annotations

import os
import re
import time

import openai

from transcription_tools.config import DEFAULT_CLEANUP_MODEL
from transcription_tools.text_processing import (
    sanitize_model_output,
    split_at_word_boundaries,
    split_into_chunks,
)

# -- Chunking thresholds -------------------------------------------------
MAX_CHUNK_CHARS = 2500
MIN_SUBDIVIDE_CHARS = 1000

# -- Retry and rate-limiting ---------------------------------------------
MAX_RETRIES = 3
INTER_REQUEST_DELAY_SECONDS = 0.5
DEFAULT_RATE_LIMIT_WAIT_SECONDS = 20
MAX_RATE_LIMIT_WAIT_SECONDS = 120

# -- Validation thresholds -----------------------------------------------
MIN_ACCEPTABLE_WORD_RATIO = 0.75
WORD_COUNT_TOLERANCE_LOW = 0.8
WORD_COUNT_TOLERANCE_HIGH = 1.2
COMMENTARY_CHECK_PREFIX_CHARS = 200

# -- Domain-specific corrections -----------------------------------------
TERM_CORRECTIONS = [
    ("sub splash", "Subsplash"), ("sub-splash", "Subsplash"),
    ("subsplash", "Subsplash"),
    ("cyber duck", "CyberDuck"), ("cyberduck", "CyberDuck"),
    ("4k downloader", "4K Downloader"),
    ("fd tp", "FTP"),
    (" gonna ", " going to "), (" wanna ", " want to "),
    (" gotta ", " got to "), (" kinda ", " kind of "),
]


def _compile_corrections(
    terms: list[tuple[str, str]],
) -> list[tuple[re.Pattern[str], str]]:
    """Compile TERM_CORRECTIONS into (regex, replacement) pairs.

    Terms with leading/trailing whitespace (e.g., " gonna ") use word
    boundary markers so they match at sentence edges and after punctuation.
    """
    result = []
    for raw, replacement in terms:
        stripped = raw.strip()
        escaped = re.escape(stripped)
        if raw != stripped:
            escaped = r"\b" + escaped + r"\b"
        pattern = re.compile(escaped, re.IGNORECASE)
        result.append((pattern, replacement.strip()))
    return result


_COMPILED_CORRECTIONS = _compile_corrections(TERM_CORRECTIONS)

META_PHRASES = [
    "here is", "here's the", "cleaned transcript",
    "the transcript", "the speaker", "this chunk",
]


def apply_basic_cleanup(text: str) -> str:
    """Regex-based fallback: applies TERM_CORRECTIONS to raw text."""
    for pattern, replacement in _COMPILED_CORRECTIONS:
        text = pattern.sub(replacement, text)
    return text


def response_is_valid(response: str, original_word_count: int) -> bool:
    """Check word count ratio and absence of meta-commentary."""
    output_word_count = len(response.split())
    ratio = output_word_count / original_word_count if original_word_count else 1.0

    if ratio < MIN_ACCEPTABLE_WORD_RATIO:
        return False

    prefix = response.lower()[:COMMENTARY_CHECK_PREFIX_CHARS]
    return not any(phrase in prefix for phrase in META_PHRASES)


def build_cleanup_prompt(chunk_text: str, chunk_idx: int, total: int) -> str:
    """Build the prompt for a cleanup API call."""
    word_count = len(chunk_text.split())
    correction_lines = "\n".join(
        f"- '{old.strip()}' -> '{new}'" for old, new in TERM_CORRECTIONS
    )
    return (
        f"CRITICAL WORD COUNT REQUIREMENT\n"
        f"Input: {word_count} words\n"
        f"Output MUST be: {int(word_count * WORD_COUNT_TOLERANCE_LOW)}"
        f"-{int(word_count * WORD_COUNT_TOLERANCE_HIGH)} words\n\n"
        "YOUR ONLY TASK: Fix spelling, grammar, and formatting errors.\n"
        "This is EDITING, not summarizing.\n\n"
        "FORBIDDEN:\n"
        "- NO summarizing or condensing\n"
        "- NO skipping content\n"
        "- NO headers like 'Here is the cleaned transcript'\n"
        "- NO meta-commentary\n"
        "- NO bullet points unless explicitly spoken\n"
        "- NO repeating sentences\n\n"
        f"REQUIRED FIXES:\n{correction_lines}\n"
        "- Add punctuation and paragraph breaks\n"
        "- Fix repeated words\n"
        "- Preserve ALL technical terms, numbers, examples\n\n"
        "OUTPUT: Start with first word, end with last word. No commentary.\n\n"
        f"Transcript chunk {chunk_idx}/{total}:\n\n{chunk_text}"
    )


class TranscriptCleaner:
    """Cleans raw Whisper transcripts via the OpenAI chat API."""

    def __init__(
        self,
        model: str = DEFAULT_CLEANUP_MODEL,
        base_url: str | None = None,
    ) -> None:
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment")

        self._model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._consecutive_rate_limits = 0
        self._rate_limit_wait_seconds = DEFAULT_RATE_LIMIT_WAIT_SECONDS

    # -- API interaction -------------------------------------------------

    def _send_cleanup_request(self, prompt: str) -> str:
        """Send prompt to OpenAI, return response text."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        return (response.choices[0].message.content or "").strip()

    def _maybe_raise_api_error(self, exc: Exception, chunk_idx: int) -> None:
        """Handle OpenAI API errors. Re-raises AuthenticationError; returns None for retryable errors."""
        from openai import AuthenticationError, RateLimitError

        if isinstance(exc, AuthenticationError):
            raise exc

        if isinstance(exc, RateLimitError):
            retry_after = DEFAULT_RATE_LIMIT_WAIT_SECONDS
            match = re.search(r"in (\d+)s", str(exc))
            if match:
                retry_after = int(match.group(1))
            self._consecutive_rate_limits += 1
            self._rate_limit_wait_seconds = min(
                MAX_RATE_LIMIT_WAIT_SECONDS,
                retry_after * (2 ** self._consecutive_rate_limits),
            )
            print(f"[cleanup] chunk {chunk_idx}: rate limited, waiting {self._rate_limit_wait_seconds}s")
            return

        print(f"[cleanup] chunk {chunk_idx} error: {str(exc)[:100]}")

    # -- Chunk processing ------------------------------------------------

    def _process_chunk(
        self, chunk_text: str, chunk_idx: int, total: int, attempt: int = 1,
    ) -> str | None:
        """Process a single chunk through OpenAI. Returns cleaned text or None."""
        original_word_count = len(chunk_text.split())
        prompt = build_cleanup_prompt(chunk_text, chunk_idx, total)

        if self._consecutive_rate_limits > 0:
            time.sleep(self._rate_limit_wait_seconds)
        else:
            time.sleep(INTER_REQUEST_DELAY_SECONDS * (2 ** (attempt - 1)))

        print(
            f"[cleanup] chunk {chunk_idx}/{total}: processing "
            f"(attempt {attempt}, {len(chunk_text)} chars)...",
            flush=True,
        )

        try:
            raw = self._send_cleanup_request(prompt)
        except openai.OpenAIError as exc:
            self._maybe_raise_api_error(exc, chunk_idx)
            return None

        cleaned = sanitize_model_output(raw)
        ratio = len(cleaned.split()) / original_word_count if original_word_count else 1.0

        if not response_is_valid(cleaned, original_word_count):
            print(f"[cleanup] chunk {chunk_idx}: quality issue (ratio={ratio:.2f})")
            return None

        self._consecutive_rate_limits = 0
        print(f"[cleanup] chunk {chunk_idx}/{total}: done (ratio={ratio:.2f})")
        return cleaned

    # -- Adaptive chunking -----------------------------------------------

    def _process_with_adaptive_chunking(self, text: str, idx: int, total: int) -> str:
        """Try to process a chunk, subdividing on repeated failure."""
        max_chars = len(text)

        for attempt in range(1, MAX_RETRIES + 1):
            result = self._process_chunk(text, idx, total, attempt)
            if result:
                return result

            if attempt >= MAX_RETRIES:
                break

            max_chars //= 2
            if max_chars < MIN_SUBDIVIDE_CHARS:
                continue

            print(f"[cleanup] chunk {idx}: subdividing to {max_chars} chars")
            sub_chunks = split_at_word_boundaries(text, max_chars)
            parts = []
            for sub in sub_chunks:
                sub_result = self._process_chunk(sub, idx, total, attempt + 1)
                parts.append(sub_result or apply_basic_cleanup(sub))
            return " ".join(parts)

        print(f"[cleanup] chunk {idx}: falling back to basic cleanup")
        return apply_basic_cleanup(text)

    # -- Public API -------------------------------------------------------

    def clean(self, raw_text: str) -> str:
        """Clean a raw transcript, returning the cleaned version."""
        chunks = split_into_chunks(raw_text, max_chars=MAX_CHUNK_CHARS)
        if not chunks:
            return raw_text

        total = len(chunks)
        print(f"[cleanup] model={self._model}, chunks={total}")

        results = []
        for i, chunk in enumerate(chunks):
            cleaned = self._process_with_adaptive_chunking(chunk, i + 1, total)
            results.append(cleaned)

        joined = "\n\n".join(r for r in results if r).strip()
        return re.sub(r"\n{3,}", "\n\n", joined)
