import dataclasses

import pytest

from transcription_tools.config import (
    ALLOWED_CLEANUP_MODELS,
    DEFAULT_CLEANUP_MODEL,
    TIERS,
    FasterWhisperParams,
    OpenAIWhisperParams,
    TranscriptionTier,
)

EXPECTED_TIERS = ["veryfast", "fast", "medium", "slow", "veryslow"]


class TestFrozenDataclasses:
    """All config dataclasses must be frozen and truly immutable."""

    def test_transcription_tier_is_dataclass(self):
        assert dataclasses.is_dataclass(TranscriptionTier)

    def test_transcription_tier_is_immutable(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            TIERS["fast"].name = "hacked"

    def test_faster_whisper_params_is_immutable(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            FasterWhisperParams().vad_filter = True

    def test_openai_whisper_params_is_immutable(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            OpenAIWhisperParams().verbose = True

    def test_vad_params_mapping_is_immutable(self):
        with pytest.raises(TypeError):
            TIERS["fast"].backend_params.vad_params["min_speech_duration_ms"] = 99999


class TestBackendParamDefaults:
    """Default values for each backend param dataclass."""

    def test_faster_whisper_defaults(self):
        params = FasterWhisperParams()
        assert params.language is None
        assert params.vad_filter is False
        assert params.vad_params is None
        assert params.without_timestamps is True
        assert params.compute_type_gpu == "int8_float16"
        assert params.compute_type_cpu == "int8"

    def test_openai_whisper_defaults(self):
        params = OpenAIWhisperParams()
        assert params.initial_prompt is None
        assert params.verbose is False
        assert params.fp16_on_gpu is True
        assert params.signal_handling is False


class TestTiersExist:
    def test_all_five_tiers_present(self):
        assert set(TIERS.keys()) == set(EXPECTED_TIERS)

    def test_no_duplicate_names(self):
        names = [t.name for t in TIERS.values()]
        assert len(names) == len(set(names))

    def test_name_matches_dict_key(self):
        for key, tier in TIERS.items():
            assert tier.name == key, f"TIERS['{key}'].name is '{tier.name}', expected '{key}'"


class TestTierModels:
    @pytest.mark.parametrize(
        "name,expected_model",
        [
            ("veryfast", "tiny.en"),
            ("fast", "base"),
            ("medium", "medium"),
            ("slow", "medium"),
            ("veryslow", "large-v3"),
        ],
    )
    def test_tier_model(self, name, expected_model):
        assert TIERS[name].whisper_model == expected_model


class TestTierBackendParams:
    """Verify each tier carries the correct backend_params type and values."""

    @pytest.mark.parametrize("name", ["veryfast", "fast", "medium"])
    def test_faster_whisper_tiers(self, name):
        assert isinstance(TIERS[name].backend_params, FasterWhisperParams)

    @pytest.mark.parametrize("name", ["slow", "veryslow"])
    def test_openai_whisper_tiers(self, name):
        assert isinstance(TIERS[name].backend_params, OpenAIWhisperParams)

    def test_veryfast_language_is_english(self):
        assert TIERS["veryfast"].backend_params.language == "en"

    def test_fast_has_vad(self):
        params = TIERS["fast"].backend_params
        assert params.vad_filter is True
        assert params.vad_params is not None

    def test_medium_compute_type_gpu_is_float16(self):
        assert TIERS["medium"].backend_params.compute_type_gpu == "float16"

    def test_veryslow_has_initial_prompt(self):
        params = TIERS["veryslow"].backend_params
        assert params.initial_prompt is not None
        assert "Subsplash" in params.initial_prompt

    def test_veryslow_signal_handling(self):
        assert TIERS["veryslow"].backend_params.signal_handling is True

    def test_veryslow_enhanced_audio_and_backup(self):
        tier = TIERS["veryslow"]
        assert tier.enhanced_audio is True
        assert tier.save_backup is True

    @pytest.mark.parametrize("name", ["veryfast", "fast", "medium", "slow"])
    def test_other_tiers_no_extras(self, name):
        tier = TIERS[name]
        assert tier.enhanced_audio is False
        assert tier.save_backup is False


class TestCleanupConfig:
    def test_default_model_in_allowed(self):
        assert DEFAULT_CLEANUP_MODEL in ALLOWED_CLEANUP_MODELS

    def test_allowed_models_is_frozenset(self):
        assert isinstance(ALLOWED_CLEANUP_MODELS, frozenset)

    def test_allowed_models_has_entries(self):
        assert len(ALLOWED_CLEANUP_MODELS) >= 1
