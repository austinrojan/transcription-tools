"""Meta-command entry point: transcription-tools config|update|uninstall|version."""

from __future__ import annotations

import argparse
import sys

from transcription_tools import __version__
from transcription_tools.user_config import load_config, save_config, _write_config


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
        print("Update command not yet implemented.")
        sys.exit(0)

    if args.command == "uninstall":
        print("Uninstall command not yet implemented.")
        sys.exit(0)

    parser.print_help()
    sys.exit(1)
