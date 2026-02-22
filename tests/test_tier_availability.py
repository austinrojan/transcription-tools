"""Tests for architecture-aware tier availability."""

from __future__ import annotations

from unittest.mock import patch

from transcription_tools.config import get_available_tiers, TIERS


def _mock_importable(*available_modules: str):
    """Return a patch that makes _is_importable return True only for listed modules."""
    def _is_importable(name: str) -> bool:
        return name in available_modules
    return patch("transcription_tools.config._is_importable", side_effect=_is_importable)


class TestGetAvailableTiers:

    def test_all_tiers_when_all_backends_importable(self):
        with _mock_importable("faster_whisper", "torch"):
            available = get_available_tiers()
        assert set(available.keys()) == set(TIERS.keys())

    def test_excludes_openai_whisper_tiers_when_torch_missing(self):
        with _mock_importable("faster_whisper"):
            available = get_available_tiers()
        assert "veryfast" in available
        assert "fast" in available
        assert "medium" in available
        assert "slow" not in available
        assert "veryslow" not in available

    def test_excludes_faster_whisper_tiers_when_fw_missing(self):
        with _mock_importable("torch"):
            available = get_available_tiers()
        assert "slow" in available
        assert "veryslow" in available
        assert "veryfast" not in available

    def test_returns_empty_when_no_backends(self):
        with _mock_importable():
            assert get_available_tiers() == {}

    def test_return_values_are_same_tier_objects(self):
        with _mock_importable("faster_whisper", "torch"):
            available = get_available_tiers()
        for name, tier in available.items():
            assert tier is TIERS[name]
