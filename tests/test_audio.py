"""Tests for audio conversion module."""

from unittest.mock import patch
from pathlib import Path

import pytest

from transcription_tools.audio import find_ffmpeg, FFMPEG_CANDIDATES


class TestFindFfmpeg:
    def test_returns_shutil_which_result(self):
        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            assert find_ffmpeg() == "/usr/bin/ffmpeg"

    def test_falls_back_to_candidates(self):
        with patch("shutil.which", return_value=None), \
             patch.object(Path, "is_file", return_value=True):
            result = find_ffmpeg()
            assert result in FFMPEG_CANDIDATES

    def test_raises_when_not_found(self):
        with patch("shutil.which", return_value=None), \
             patch.object(Path, "is_file", return_value=False):
            with pytest.raises(FileNotFoundError, match="ffmpeg not found"):
                find_ffmpeg()
