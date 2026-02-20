"""Tests for transcript cleanup module."""

import pytest

from transcription_tools.cleanup import (
    TERM_CORRECTIONS,
    TranscriptCleaner,
)


class TestApplyBasicCleanup:
    """Test the regex-based fallback cleanup."""

    def test_corrects_known_terms(self):
        result = TranscriptCleaner._apply_basic_cleanup("upload to sub splash")
        assert "Subsplash" in result

    def test_case_insensitive(self):
        result = TranscriptCleaner._apply_basic_cleanup("upload to SUB SPLASH")
        assert "Subsplash" in result

    def test_corrects_contractions(self):
        result = TranscriptCleaner._apply_basic_cleanup("I gonna do it")
        assert "going to" in result

    def test_preserves_unmatched_text(self):
        text = "This is normal text without corrections needed."
        assert TranscriptCleaner._apply_basic_cleanup(text) == text


class TestResponseIsValid:
    """Test the quality-gate validation."""

    def test_valid_response(self):
        assert TranscriptCleaner._response_is_valid("word " * 100, 100) is True

    def test_too_short_rejected(self):
        assert TranscriptCleaner._response_is_valid("word " * 50, 100) is False

    def test_meta_commentary_rejected(self):
        text = "Here is the cleaned transcript: " + "word " * 100
        assert TranscriptCleaner._response_is_valid(text, 100) is False

    def test_zero_original_words(self):
        # Guard against division by zero — should accept (ratio defaults to 1.0)
        assert TranscriptCleaner._response_is_valid("some text", 0) is True
