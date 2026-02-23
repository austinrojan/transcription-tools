"""macOS audio transcription with Finder integration."""

try:
    from importlib.metadata import version

    __version__ = version("transcription-tools")
except Exception:
    __version__ = "0.0.0-dev"
