"""Meta-command entry point: transcription-tools config|update|uninstall|version."""

from __future__ import annotations

import argparse
import sys

from transcription_tools import __version__
from transcription_tools.install_paths import (
    INSTALL_DIR,
    SERVICES_DIR,
    VERSION_FILE,
    WRAPPER_COMMANDS,
)
from transcription_tools.user_config import CONFIG_DIR, load_config, save_config, _write_config


def _parse_version(v: str) -> tuple[int, ...]:
    """Parse a semver string into a tuple for correct numeric comparison."""
    return tuple(int(x) for x in v.split("."))


def _get_installed_version() -> str:
    """Read installed version from version.txt, falling back to package version."""
    if VERSION_FILE.exists():
        return VERSION_FILE.read_text(encoding="utf-8").strip()
    return __version__


def _get_latest_version() -> str:
    """Fetch the latest release version from GitHub."""
    import json
    import urllib.request

    url = "https://api.github.com/repos/austinrojan/transcription-tools/releases/latest"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        return data.get("tag_name", "").lstrip("v")
    except Exception:
        return __version__


def check_for_update() -> tuple[bool, str, str]:
    """Check if a newer version is available on GitHub.

    Returns (has_update, current_version, latest_version).
    """
    current = _get_installed_version()
    latest = _get_latest_version()
    has_update = _parse_version(latest) > _parse_version(current)
    return has_update, current, latest


def get_uninstall_paths(
    *, keep_config: bool, keep_models: bool,
) -> dict[str, list]:
    """Build lists of directories and files to remove during uninstall."""
    from pathlib import Path

    dirs: list[Path] = [INSTALL_DIR]
    files: list[Path] = [Path(f"/usr/local/bin/{cmd}") for cmd in WRAPPER_COMMANDS]

    for name in ["Very Fast", "Fast", "Medium", "Slow", "Very Slow"]:
        dirs.append(SERVICES_DIR / f"Transcribe Audio - {name}.workflow")

    if not keep_config:
        dirs.append(CONFIG_DIR)
    if not keep_models:
        dirs.append(Path.home() / ".cache" / "whisper")
        dirs.append(Path.home() / ".cache" / "huggingface")

    return {"dirs": dirs, "files": files}


def _mask_key(key: str) -> str:
    """Mask an API key for display, showing first 4 and last 3 chars."""
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}...{key[-3:]}"


def config_command(
    *,
    show: bool,
    set_api_key: bool,
    set_pair: tuple[str, str] | None,
    unset: str | None,
) -> None:
    """Handle the 'config' subcommand."""
    if show:
        config = load_config()
        if not config:
            print("No configuration set. Use --set-api-key to get started.")
            return
        for key, value in sorted(config.items()):
            display = _mask_key(str(value)) if "key" in key.lower() else value
            print(f"  {key} = {display}")
        return

    if set_api_key:
        key = input("Enter your OpenAI API key (starts with 'sk-'): ").strip()
        if not key.startswith("sk-") or len(key) < 10:
            print("Error: API key must start with 'sk-' and be at least 10 characters.")
            sys.exit(1)
        save_config({"openai_api_key": key})
        print("API key saved. Transcript cleanup is now enabled for all tiers.")
        return

    if set_pair is not None:
        save_config({set_pair[0]: set_pair[1]})
        print(f"Set {set_pair[0]} = {set_pair[1]}")
        return

    if unset is not None:
        config = load_config()
        config.pop(unset, None)
        _write_config(config)
        print(f"Removed {unset} from config.")
        return


def main() -> None:
    """Entry point for 'transcription-tools' command."""
    parser = argparse.ArgumentParser(
        prog="transcription-tools",
        description="Manage transcription tools installation and configuration.",
    )
    sub = parser.add_subparsers(dest="command")

    config_parser = sub.add_parser("config", help="Manage configuration")
    config_parser.add_argument(
        "--show", action="store_true", help="Show current config",
    )
    config_parser.add_argument(
        "--set-api-key", action="store_true",
        help="Set OpenAI API key interactively",
    )
    config_parser.add_argument(
        "--set", nargs=2, metavar=("KEY", "VALUE"),
        help="Set a config value",
    )
    config_parser.add_argument(
        "--unset", metavar="KEY", help="Remove a config value",
    )

    sub.add_parser("version", help="Show version")
    sub.add_parser("update", help="Check for and install updates")
    sub.add_parser("uninstall", help="Remove transcription tools")

    args = parser.parse_args()

    if args.command == "version":
        print(f"transcription-tools {__version__}")
        sys.exit(0)

    if args.command == "config":
        set_pair = tuple(args.set) if args.set else None
        config_command(
            show=args.show,
            set_api_key=args.set_api_key,
            set_pair=set_pair,
            unset=args.unset,
        )
        return

    if args.command == "update":
        has_update, current, latest = check_for_update()
        if not has_update:
            print(f"Already up to date (v{current}).")
        else:
            print(f"Update available: v{current} -> v{latest}")
            print("To update, re-run the install script:")
            print(
                "  curl -fsSL https://raw.githubusercontent.com/"
                "austinrojan/transcription-tools/main/install.sh | bash"
            )
        sys.exit(0)

    if args.command == "uninstall":
        import shutil

        print("This will remove Transcription Tools from your system.\n")

        keep_config = True
        keep_models = True
        resp = input("  Also remove configuration (API key, settings)? [y/N] ").strip()
        if resp.lower() in ("y", "yes"):
            keep_config = False
        resp = input("  Also remove downloaded model files (~2-5 GB)? [y/N] ").strip()
        if resp.lower() in ("y", "yes"):
            keep_models = False

        paths = get_uninstall_paths(keep_config=keep_config, keep_models=keep_models)

        print("\nThe following will be removed:")
        for d in paths["dirs"]:
            if d.exists():
                print(f"  [dir]  {d}")
        for f in paths["files"]:
            if f.exists():
                print(f"  [file] {f}")

        resp = input("\nProceed? [y/N] ").strip()
        if resp.lower() not in ("y", "yes"):
            print("Uninstall cancelled.")
            sys.exit(0)

        for f in paths["files"]:
            try:
                f.unlink(missing_ok=True)
            except PermissionError:
                import subprocess
                subprocess.run(["sudo", "rm", "-f", str(f)], check=False)
        for d in paths["dirs"]:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)

        print("\nTranscription Tools has been removed.")
        sys.exit(0)

    parser.print_help()
    sys.exit(1)
