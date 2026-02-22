"""Tests for programmatic Automator workflow generation."""

from __future__ import annotations

import pytest

import plistlib

from scripts.generate_workflows import build_shell_command, WORKFLOW_TIERS, _patch_info_plist


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

    def test_uses_pipestatus_for_exit_code(self):
        assert "PIPESTATUS[0]" in self.cmd
        assert "STATUS=$?" not in self.cmd

    def test_strips_quotes_from_basename_for_notification(self):
        assert "tr -d" in self.cmd

    def test_different_commands_produce_different_scripts(self):
        slow = build_shell_command("transcribe-slow")
        assert "transcribe-fast" in self.cmd
        assert "transcribe-slow" in slow
        assert "transcribe-fast" not in slow


class TestPatchInfoPlist:

    def test_creates_nsmenuitem_when_missing(self, tmp_path):
        """Regression: .get('NSMenuItem', {}) returned a detached dict."""
        plist_dir = tmp_path / "Test.workflow" / "Contents"
        plist_dir.mkdir(parents=True)
        info_plist = plist_dir / "Info.plist"
        data = {"NSServices": [{"NSMessage": "runWorkflowAsService"}]}
        with open(info_plist, "wb") as f:
            plistlib.dump(data, f)

        _patch_info_plist(tmp_path / "Test.workflow", "Fast")

        with open(info_plist, "rb") as f:
            result = plistlib.load(f)
        assert result["NSServices"][0]["NSMenuItem"]["default"] == "Transcribe Audio - Fast"


class TestWorkflowTiers:

    def test_all_five_tiers_defined(self):
        assert len(WORKFLOW_TIERS) == 5

    def test_tier_names_match_commands(self):
        for tier in WORKFLOW_TIERS:
            assert tier["command"].startswith("transcribe-")
