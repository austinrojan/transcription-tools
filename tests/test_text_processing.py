import pytest

from transcription_tools.text_processing import (
    split_into_chunks,
    sanitize_model_output,
)


class TestSplitIntoChunks:
    def test_empty_string(self):
        assert split_into_chunks("") == []

    def test_short_text_single_chunk(self):
        assert split_into_chunks("Short text.") == ["Short text."]

    def test_respects_max_chars(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = split_into_chunks(text, max_chars=40)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 40

    def test_force_splits_huge_sentence(self):
        huge = "A" * 200
        chunks = split_into_chunks(huge, max_chars=50)
        assert len(chunks) == 4
        for chunk in chunks:
            assert len(chunk) <= 50

    def test_preserves_all_content(self):
        text = "Hello world. This is a test. Final sentence."
        chunks = split_into_chunks(text, max_chars=1000)
        rejoined = " ".join(chunks)
        assert rejoined == text

    def test_whitespace_only(self):
        assert split_into_chunks("   ") == []

    def test_force_split_preserves_words(self):
        """Force-split prefers whitespace over mid-word cuts."""
        text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
        chunks = split_into_chunks(text, max_chars=25)
        all_words = set(text.split())
        for chunk in chunks:
            chunk_words = set(chunk.split())
            assert chunk_words.issubset(all_words), f"Chunk '{chunk}' contains split words"

    def test_max_chars_zero_raises(self):
        with pytest.raises(ValueError, match="max_chars must be positive"):
            split_into_chunks("hello", max_chars=0)

    def test_max_chars_negative_raises(self):
        with pytest.raises(ValueError, match="max_chars must be positive"):
            split_into_chunks("hello", max_chars=-5)


class TestSanitizeModelOutput:
    def test_strips_cleaned_up_prefix(self):
        result = sanitize_model_output(
            "Here is the cleaned-up transcript: Hello world"
        )
        assert result == "Hello world"

    def test_strips_cleaned_transcript_prefix(self):
        result = sanitize_model_output("Cleaned-up transcript: Hello world")
        assert result == "Hello world"

    def test_no_prefix_unchanged(self):
        assert sanitize_model_output("Hello world") == "Hello world"

    def test_strips_whitespace(self):
        assert sanitize_model_output("  Hello world  ") == "Hello world"

    def test_case_insensitive(self):
        result = sanitize_model_output(
            "HERE IS THE CLEANED UP TRANSCRIPT: Hello"
        )
        assert result == "Hello"
