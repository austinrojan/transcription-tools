"""Tests for programmatic Automator workflow generation."""

from __future__ import annotations

from scripts.generate_workflows import build_shell_command, WORKFLOW_TIERS


class TestBuildShellCommand:

    def test_includes_path_setup(self):
        cmd = build_shell_command("transcribe-fast")
        assert "/usr/local/bin" in cmd
        assert "transcription-tools/ffmpeg" in cmd

    def test_includes_installation_check(self):
        cmd = build_shell_command("transcribe-fast")
        assert "command -v" in cmd

    def test_includes_correct_command(self):
        cmd = build_shell_command("transcribe-fast")
        assert 'transcribe-fast "$f"' in cmd

    def test_includes_notification(self):
        cmd = build_shell_command("transcribe-fast")
        assert "display notification" in cmd

    def test_includes_logging(self):
        cmd = build_shell_command("transcribe-fast")
        assert "tee" in cmd

    def test_different_commands_produce_different_scripts(self):
        fast = build_shell_command("transcribe-fast")
        slow = build_shell_command("transcribe-slow")
        assert "transcribe-fast" in fast
        assert "transcribe-slow" in slow
        assert "transcribe-fast" not in slow


class TestWorkflowTiers:

    def test_all_five_tiers_defined(self):
        assert len(WORKFLOW_TIERS) == 5

    def test_tier_names_match_commands(self):
        for tier in WORKFLOW_TIERS:
            assert tier["command"].startswith("transcribe-")
