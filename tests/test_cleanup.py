"""Tests for transcript cleanup module."""

import pytest

from transcription_tools.cleanup import (
    TERM_CORRECTIONS,
    TranscriptCleaner,
    apply_basic_cleanup,
    build_cleanup_prompt,
    response_is_valid,
)


class TestApplyBasicCleanup:
    """Test the regex-based fallback cleanup."""

    def test_corrects_known_terms(self):
        result = apply_basic_cleanup("upload to sub splash")
        assert "Subsplash" in result

    def test_case_insensitive(self):
        result = apply_basic_cleanup("upload to SUB SPLASH")
        assert "Subsplash" in result

    def test_corrects_contractions(self):
        result = apply_basic_cleanup("I gonna do it")
        assert "going to" in result

    def test_preserves_unmatched_text(self):
        text = "This is normal text without corrections needed."
        assert apply_basic_cleanup(text) == text


class TestResponseIsValid:
    """Test the quality-gate validation."""

    def test_valid_response(self):
        assert response_is_valid("word " * 100, 100) is True

    def test_too_short_rejected(self):
        assert response_is_valid("word " * 50, 100) is False

    def test_meta_commentary_rejected(self):
        text = "Here is the cleaned transcript: " + "word " * 100
        assert response_is_valid(text, 100) is False

    def test_zero_original_words(self):
        # Guard against division by zero — should accept (ratio defaults to 1.0)
        assert response_is_valid("some text", 0) is True


class TestBuildCleanupPrompt:
    """Test the extracted prompt-building function."""

    def test_word_count_range_in_prompt(self):
        text = " ".join(["word"] * 100)
        prompt = build_cleanup_prompt(text, 1, 1)
        assert "80" in prompt  # 100 * 0.8
        assert "120" in prompt  # 100 * 1.2

    def test_chunk_index_in_prompt(self):
        prompt = build_cleanup_prompt("some text", 3, 7)
        assert "3/7" in prompt

    def test_term_corrections_listed(self):
        prompt = build_cleanup_prompt("some text", 1, 1)
        for old, new in TERM_CORRECTIONS:
            assert new in prompt

    def test_chunk_text_embedded(self):
        prompt = build_cleanup_prompt("My unique transcript content", 1, 1)
        assert "My unique transcript content" in prompt
