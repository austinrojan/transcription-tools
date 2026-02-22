#!/usr/bin/env python3
"""Generate Automator .workflow bundles programmatically.

Reads the Fast workflow as a template and generates all five tier
workflows with improved shell scripts (installation check, logging,
notifications). Also patches Info.plist so each workflow has the
correct service menu name.

Usage: python scripts/generate_workflows.py
"""

from __future__ import annotations

import copy
import plistlib
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_WORKFLOW = REPO_ROOT / "workflows" / "Transcribe Audio - Fast.workflow"
OUTPUT_DIR = REPO_ROOT / "workflows"

WORKFLOW_TIERS = [
    {"command": "transcribe-veryfast", "label": "Very Fast"},
    {"command": "transcribe-fast", "label": "Fast"},
    {"command": "transcribe-medium", "label": "Medium"},
    {"command": "transcribe-slow", "label": "Slow"},
    {"command": "transcribe-veryslow", "label": "Very Slow"},
]


def build_shell_command(command_name: str) -> str:
    """Build the shell script embedded in a workflow."""
    return (
        'export PATH="/usr/local/bin:/opt/homebrew/bin:$PATH"\n'
        'export PATH="$HOME/Library/Application Support/transcription-tools/ffmpeg:$PATH"\n'
        "\n"
        'for f in "$@"; do\n'
        f"    if ! command -v {command_name} &>/dev/null; then\n"
        "        osascript -e 'display dialog \"Transcription Tools is not installed.\\n\\n"
        'Visit https://github.com/austinrojan/transcription-tools for installation instructions."'
        ' buttons {"OK"} default button "OK" with icon caution'
        " with title \"Transcription Tools\"'\n"
        "        exit 1\n"
        "    fi\n"
        f'    {command_name} "$f" 2>&1 | tee -a "$HOME/Library/Logs/transcription-tools.log"\n'
        "    STATUS=$?\n"
        '    BASENAME=$(basename "$f")\n'
        "    if [ $STATUS -eq 0 ]; then\n"
        '        osascript -e "display notification \\"Transcription complete: $BASENAME\\"'
        ' with title \\"Transcription Tools\\""\n'
        "    else\n"
        '        osascript -e "display notification \\"Transcription failed: $BASENAME.'
        ' Check log for details.\\" with title \\"Transcription Tools\\""\n'
        "    fi\n"
        "done"
    )


def _patch_info_plist(workflow_dir: Path, label: str) -> None:
    """Update Info.plist so the service menu name matches the tier."""
    info_plist_path = workflow_dir / "Contents" / "Info.plist"
    if not info_plist_path.exists():
        return
    with open(info_plist_path, "rb") as f:
        info = plistlib.load(f)
    services = info.get("NSServices", [])
    if services:
        menu_item = services[0].get("NSMenuItem", {})
        menu_item["default"] = f"Transcribe Audio - {label}"
    with open(info_plist_path, "wb") as f:
        plistlib.dump(info, f)


def generate_all() -> None:
    """Generate all five workflow bundles from the template."""
    template_plist_path = TEMPLATE_WORKFLOW / "Contents" / "document.wflow"
    with open(template_plist_path, "rb") as f:
        template = plistlib.load(f)

    for tier in WORKFLOW_TIERS:
        name = f"Transcribe Audio - {tier['label']}"
        workflow_dir = OUTPUT_DIR / f"{name}.workflow"

        if workflow_dir != TEMPLATE_WORKFLOW:
            if workflow_dir.exists():
                shutil.rmtree(workflow_dir)
            shutil.copytree(TEMPLATE_WORKFLOW, workflow_dir)

        plist = copy.deepcopy(template)
        action_params = plist["actions"][0]["action"]["ActionParameters"]
        action_params["COMMAND_STRING"] = build_shell_command(tier["command"])

        plist_path = workflow_dir / "Contents" / "document.wflow"
        with open(plist_path, "wb") as f:
            plistlib.dump(plist, f)

        _patch_info_plist(workflow_dir, tier["label"])

        print(f"  Generated: {name}.workflow")


if __name__ == "__main__":
    generate_all()
    print("Done.")
