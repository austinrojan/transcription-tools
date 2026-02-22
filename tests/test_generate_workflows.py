"""Tests for programmatic Automator workflow generation."""

from __future__ import annotations

import pytest

from scripts.generate_workflows import build_shell_command, WORKFLOW_TIERS


class TestBuildShellCommand:

    @pytest.fixture(autouse=True)
    def _fast_cmd(self):
        self.cmd = build_shell_command("transcribe-fast")

    def test_includes_path_setup(self):
        assert "/usr/local/bin" in self.cmd
        assert "transcription-tools/ffmpeg" in self.cmd

    def test_includes_installation_check(self):
        assert "command -v" in self.cmd

    def test_includes_correct_command(self):
        assert 'transcribe-fast "$f"' in self.cmd

    def test_includes_notification(self):
        assert "display notification" in self.cmd

    def test_includes_logging(self):
        assert "tee" in self.cmd

    def test_different_commands_produce_different_scripts(self):
        slow = build_shell_command("transcribe-slow")
        assert "transcribe-fast" in self.cmd
        assert "transcribe-slow" in slow
        assert "transcribe-fast" not in slow


class TestWorkflowTiers:

    def test_all_five_tiers_defined(self):
        assert len(WORKFLOW_TIERS) == 5

    def test_tier_names_match_commands(self):
        for tier in WORKFLOW_TIERS:
            assert tier["command"].startswith("transcribe-")
