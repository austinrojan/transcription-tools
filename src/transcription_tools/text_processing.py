"""Text processing for transcript chunking and validation.

Handles chunk splitting for API calls and output sanitization.
"""

from __future__ import annotations

import re


def split_into_chunks(text: str, max_chars: int = 2500) -> list[str]:
    """Split text into chunks on sentence boundaries.

    Prefers splitting at sentence-ending punctuation. Falls back to
    hard-cutting at max_chars for sentences that exceed the limit.
    """
    if max_chars < 1:
        raise ValueError(f"max_chars must be positive, got {max_chars}")
    stripped = text.strip()
    if not stripped:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", stripped)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        if not sentence:
            continue

        if len(sentence) > max_chars:
            if current:
                chunks.append(" ".join(current).strip())
                current, current_len = [], 0
            chunks.extend(split_at_word_boundaries(sentence, max_chars))
            continue

        if current_len + len(sentence) + 1 <= max_chars:
            current.append(sentence)
            current_len += len(sentence) + 1
        else:
            chunks.append(" ".join(current).strip())
            current = [sentence]
            current_len = len(sentence) + 1

    if current:
        chunks.append(" ".join(current).strip())

    return chunks


def split_at_word_boundaries(text: str, max_chars: int) -> list[str]:
    """Split text into pieces at whitespace, respecting max_chars.

    Prefers splitting at the last whitespace before max_chars. When no
    whitespace exists within a max_chars window (e.g. a single word
    exceeds max_chars), the text is hard-cut at max_chars.
    """
    if max_chars < 1:
        raise ValueError(f"max_chars must be positive, got {max_chars}")
    pieces: list[str] = []
    pos = 0
    while pos < len(text):
        end = min(pos + max_chars, len(text))
        if end < len(text):
            space_idx = text.rfind(" ", pos, end)
            if space_idx > pos:
                end = space_idx + 1
        piece = text[pos:end].strip()
        if piece:
            pieces.append(piece)
        pos = end
    return pieces


_PREFATORY_LABEL_RE = re.compile(
    r"^(?:here\s+is\s+the\s+cleaned[- ]?up\s+transcript:"
    r"|cleaned[- ]?up\s+transcript:"
    r"|here\s+is\s+the\s+transcript:)\s*",
    re.IGNORECASE,
)


def sanitize_model_output(text: str) -> str:
    """Strip prefatory labels that models sometimes prepend."""
    return _PREFATORY_LABEL_RE.sub("", text.strip()).strip()
