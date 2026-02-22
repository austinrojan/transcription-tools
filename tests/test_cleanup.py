"""Tests for transcript cleanup module."""

from transcription_tools.cleanup import (
    MIN_ACCEPTABLE_WORD_RATIO,
    TERM_CORRECTIONS,
    WORD_COUNT_TOLERANCE_HIGH,
    WORD_COUNT_TOLERANCE_LOW,
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

    def test_corrects_contraction_at_start(self):
        assert apply_basic_cleanup("gonna do it") == "going to do it"

    def test_corrects_contraction_at_end(self):
        assert apply_basic_cleanup("we gonna") == "we going to"

    def test_corrects_contraction_after_punctuation(self):
        assert "going to" in apply_basic_cleanup("I gonna, you know")

    def test_empty_string(self):
        assert apply_basic_cleanup("") == ""


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

    def test_exact_boundary_ratio_accepted(self):
        """75/100 = 0.75 = MIN_ACCEPTABLE_WORD_RATIO → accepted (uses strict <)."""
        assert response_is_valid("word " * 75, 100) is True

    def test_just_below_boundary_ratio_rejected(self):
        """74/100 = 0.74 < MIN_ACCEPTABLE_WORD_RATIO → rejected."""
        assert response_is_valid("word " * 74, 100) is False


class TestBuildCleanupPrompt:
    """Test the extracted prompt-building function."""

    def test_word_count_range_in_prompt(self):
        word_count = 100
        text = " ".join(["word"] * word_count)
        prompt = build_cleanup_prompt(text, 1, 1)
        assert str(int(word_count * WORD_COUNT_TOLERANCE_LOW)) in prompt
        assert str(int(word_count * WORD_COUNT_TOLERANCE_HIGH)) in prompt

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
