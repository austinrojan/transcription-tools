import dataclasses

import pytest

from transcription_tools.config import (
    ALLOWED_CLEANUP_MODELS,
    DEFAULT_CLEANUP_MODEL,
    TIERS,
    TranscriptionTier,
)

EXPECTED_TIERS = ["veryfast", "fast", "medium", "slow", "veryslow"]


class TestTiersExist:
    def test_all_five_tiers_present(self):
        assert set(TIERS.keys()) == set(EXPECTED_TIERS)

    def test_no_duplicate_names(self):
        names = [t.name for t in TIERS.values()]
        assert len(names) == len(set(names))


class TestTierBackends:
    @pytest.mark.parametrize("name", ["veryfast", "fast", "medium"])
    def test_faster_whisper_tiers(self, name):
        assert TIERS[name].backend == "faster_whisper"

    @pytest.mark.parametrize("name", ["slow", "veryslow"])
    def test_openai_whisper_tiers(self, name):
        assert TIERS[name].backend == "whisper"

    @pytest.mark.parametrize("name", EXPECTED_TIERS)
    def test_valid_backend(self, name):
        assert TIERS[name].backend in ("faster_whisper", "whisper")


class TestTierModels:
    @pytest.mark.parametrize("name", EXPECTED_TIERS)
    def test_has_model(self, name):
        assert TIERS[name].whisper_model

    @pytest.mark.parametrize("name,expected_model", [
        ("veryfast", "tiny.en"),
        ("fast", "base"),
        ("medium", "medium"),
        ("slow", "medium"),
        ("veryslow", "large-v3"),
    ])
    def test_tier_model(self, name, expected_model):
        assert TIERS[name].whisper_model == expected_model


class TestFrozenDataclass:
    def test_immutable(self):
        tier = TIERS["fast"]
        with pytest.raises(dataclasses.FrozenInstanceError):
            tier.name = "hacked"

    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(TranscriptionTier)

    def test_vad_params_immutable(self):
        tier = TIERS["fast"]
        with pytest.raises(TypeError):
            tier.vad_params["min_speech_duration_ms"] = 99999


class TestVeryslowExtras:
    def test_enhanced_audio(self):
        assert TIERS["veryslow"].enhanced_audio is True

    def test_signal_handling(self):
        assert TIERS["veryslow"].signal_handling is True

    def test_save_backup(self):
        assert TIERS["veryslow"].save_backup is True

    def test_has_initial_prompt(self):
        assert TIERS["veryslow"].initial_prompt is not None
        assert "Subsplash" in TIERS["veryslow"].initial_prompt

    @pytest.mark.parametrize("name", ["veryfast", "fast", "medium", "slow"])
    def test_other_tiers_no_extras(self, name):
        tier = TIERS[name]
        assert tier.enhanced_audio is False
        assert tier.signal_handling is False
        assert tier.save_backup is False


class TestCleanupConfig:
    def test_default_model_in_allowed(self):
        assert DEFAULT_CLEANUP_MODEL in ALLOWED_CLEANUP_MODELS

    def test_allowed_models_is_frozenset(self):
        assert isinstance(ALLOWED_CLEANUP_MODELS, frozenset)

    def test_allowed_models_has_entries(self):
        assert len(ALLOWED_CLEANUP_MODELS) >= 1
