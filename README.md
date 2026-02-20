# Transcription Tools

macOS audio transcription with right-click Finder integration.
Uses Whisper for local speech-to-text and OpenAI for transcript cleanup.

## How It Works

```
audio file → ffmpeg (16kHz mono WAV) → Whisper → raw transcript → OpenAI cleanup → clean transcript
```

Five speed/quality tiers:

| Tier | Backend | Model | Use Case |
|------|---------|-------|----------|
| Very Fast | faster-whisper | tiny.en | Quick notes, rough drafts |
| Fast | faster-whisper | base | General transcription |
| Medium | faster-whisper | medium | Balanced quality/speed |
| Slow | OpenAI whisper | medium | High-quality transcripts |
| Very Slow | OpenAI whisper | large-v3 | Maximum accuracy |

After transcription, an OpenAI chat model (gpt-5-nano by default) cleans up spelling, grammar, and formatting. Skip with `--no-cleanup`.

## Install

Requires macOS, Python 3.10+, and ffmpeg (`brew install ffmpeg`).

```bash
git clone <repo-url> && cd transcription-tools-package
python3 -m venv ~/.local/share/transcription-tools/venv
~/.local/share/transcription-tools/venv/bin/pip install -e .
```

Then symlink the commands:

```bash
for tier in veryfast fast medium slow veryslow; do
  sudo ln -sf ~/.local/share/transcription-tools/venv/bin/transcribe-$tier /usr/local/bin/transcribe-$tier
done
```

Set your OpenAI API key for the cleanup step:

```bash
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.zshrc
```

## Usage

### Right-click (Finder)

Copy the workflows from `workflows/` to `~/Library/Services/`, then right-click any audio file → Quick Actions → choose a tier.

### Command line

```bash
transcribe-fast recording.mp3
transcribe-medium recording.mp3 --no-cleanup
transcribe-slow recording.mp3 --openai-model gpt-5-mini
transcribe-veryslow recording.mp3 --cleanup-only
```

Output files are written next to the source audio:
- `recording_fast.txt` — raw Whisper output
- `recording_fast.clean.txt` — cleaned by OpenAI

## Flags

| Flag | Description |
|------|-------------|
| `--no-cleanup` | Skip the OpenAI cleanup pass |
| `--cleanup-only` | Re-run cleanup on an existing transcript |
| `--openai-model` | Choose cleanup model (gpt-5-nano, gpt-5-mini, gpt-5) |
| `--openai-base-url` | Custom OpenAI-compatible endpoint |

## Development

```bash
PYTHONPATH=src python3 -m pytest tests/ -v
```

## License

MIT
