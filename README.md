# Transcription Tools

Local audio and video transcription for macOS, using Whisper for speech-to-text and (optionally) OpenAI for transcript cleanup.

## What It Does

Transcribes audio and video files to text. Works with common formats like MP3, MP4, MOV, WAV, and [many more](#supported-formats).

Produces two output files:

- **Raw transcript** — direct Whisper output
- **Cleaned transcript** — spelling, grammar, and formatting corrected by AI (optional, requires an [OpenAI API key](#optional-ai-powered-cleanup))

## Speed / Quality Tiers

| Tier | Quality | Best For |
|------|---------|----------|
| Very Fast | Rough | Quick notes, scanning content |
| Fast | Good | General transcription |
| Medium | Better | Meetings, interviews |
| Slow | High | Polished transcripts |
| Very Slow | Highest | Technical or specialized content |

Higher tiers are slower but more accurate. Speed depends on file length and your hardware.

## Requirements

- macOS 12 (Monterey) or later
- 3 GB free disk space
- Internet connection (for installation only — transcription runs offline)
- Apple Silicon recommended (Intel Macs supported; see [Apple Silicon vs Intel](#apple-silicon-vs-intel))

## Install

Run this in Terminal:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/austinrojan/transcription-tools/main/install.sh)"
```

The installer sets up its own Python, ffmpeg, and Whisper — it does not touch your system Python or any existing Homebrew installation.

Everything is installed to `~/Library/Application Support/transcription-tools/`.

You'll be prompted for your password once so the installer can place commands in `/usr/local/bin`. If Xcode Command Line Tools aren't already installed, you'll be prompted to install them first.

## Usage

### Right-Click (Finder)

After installation, right-click any audio or video file in Finder → **Quick Actions** → choose a tier (e.g. "Transcribe Audio - Fast").

If Quick Actions don't appear, go to **System Settings → General → Login Items & Extensions → Extensions → Finder** and toggle on each "Transcribe Audio" tier.

### Command Line

```bash
transcribe-fast recording.mp3
transcribe-medium lecture.mp4
transcribe-slow interview.wav
```

Output files are saved next to the original:

- `recording_fast.txt` — raw transcript
- `recording_fast.clean.txt` — cleaned transcript (if an API key is set)

Use `--no-cleanup` to skip the AI cleanup step:

```bash
transcribe-fast recording.mp3 --no-cleanup
```

## Optional: AI-Powered Cleanup

After transcription, an AI model can fix spelling, grammar, punctuation, and formatting. This step is optional and requires an OpenAI API key.

To enable it:

```bash
transcription-tools config --set-api-key
```

You'll need an API key from [platform.openai.com](https://platform.openai.com). The default cleanup model is gpt-5-nano. You can choose a different model:

```bash
transcribe-fast recording.mp3 --openai-model gpt-5-mini
transcribe-fast recording.mp3 --openai-model gpt-5
```

Without an API key, transcription still works — you just get the raw transcript.

## Flags

| Flag | What it does |
|------|-------------|
| `--no-cleanup` | Skip the AI cleanup step |
| `--cleanup-only` | Re-run cleanup on an existing transcript |
| `--openai-model MODEL` | Choose cleanup model (`gpt-5-nano`, `gpt-5-mini`, `gpt-5`) |
| `--openai-base-url URL` | Use a custom OpenAI-compatible API |

## Supported Formats

Anything ffmpeg can read is supported. Common examples:

**Audio:** MP3, WAV, FLAC, AAC, OGG, M4A, AIFF, OPUS, WMA

**Video:** MP4, MOV, MKV, AVI, WebM, WMV, FLV, M4V

Video files have their audio track extracted automatically.

## Managing Your Installation

```bash
transcription-tools config --show       # View current settings
transcription-tools config --set-api-key # Set OpenAI API key
transcription-tools version             # Check installed version
transcription-tools update              # Check for updates
transcription-tools uninstall           # Remove everything
```

## Uninstall

```bash
transcription-tools uninstall
```

This removes all installed files. You'll be asked whether to also remove your configuration (API key) and downloaded Whisper models.

## Apple Silicon vs Intel

- **Apple Silicon (M1/M2/M3/M4):** All 5 tiers installed.
- **Intel:** All 5 tiers installed. Slow and Very Slow depend on PyTorch — the installer will warn you if PyTorch fails to install on your system.

## Development

```bash
git clone https://github.com/austinrojan/transcription-tools.git
cd transcription-tools
python3 -m venv venv && source venv/bin/activate
pip install -e .
python3 -m pytest tests/ -v
```

## License

MIT
