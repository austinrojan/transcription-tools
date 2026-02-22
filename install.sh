#!/bin/bash
# Transcription Tools Installer
# Usage: curl -fsSL https://raw.githubusercontent.com/austinrojan/transcription-tools/main/install.sh | bash
set -euo pipefail

# -- Version pins (update with each release) --
TT_VERSION="2.0.0"
PYTHON_VERSION="3.12.8"
PBS_VERSION="20250107"
PYTORCH_VERSION_INTEL="2.2.2"

# -- Paths --
INSTALL_DIR="$HOME/Library/Application Support/transcription-tools"
VENV_DIR="$INSTALL_DIR/venv"
PYTHON_DIR="$INSTALL_DIR/python"
FFMPEG_DIR="$INSTALL_DIR/ffmpeg"
SERVICES_DIR="$HOME/Library/Services"
LOGFILE="$HOME/Library/Logs/transcription-tools-install.log"

# -- Color helpers (degrade gracefully if no tty) --
if [ -t 1 ] && tput colors &>/dev/null; then
    BOLD=$(tput bold); BLUE=$(tput setaf 4); GREEN=$(tput setaf 2)
    RED=$(tput setaf 1); YELLOW=$(tput setaf 3); RESET=$(tput sgr0)
else
    BOLD=""; BLUE=""; GREEN=""; RED=""; YELLOW=""; RESET=""
fi

ohai()    { echo "${BOLD}${BLUE}==> $*${RESET}"; }
success() { echo "${GREEN}  ✓ $*${RESET}"; }
warn()    { echo "${YELLOW}  ! $*${RESET}"; }
abort()   { echo "${RED}  ✗ $*${RESET}" >&2; exit 1; }

# -- Global cleanup trap --
_cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo ""
        echo "  ${RED}Installation failed. See log for details:${RESET}"
        echo "    $LOGFILE"
        echo ""
        echo "  For help, open an issue at:"
        echo "    https://github.com/austinrojan/transcription-tools/issues"
    fi
}
trap _cleanup EXIT

# -- Architecture detection --
detect_architecture() {
    ARCH=$(uname -m)
    case "$ARCH" in
        arm64)
            PLATFORM="aarch64-apple-darwin"
            IS_APPLE_SILICON=true
            ;;
        x86_64)
            PLATFORM="x86_64-apple-darwin"
            IS_APPLE_SILICON=false
            ;;
        *)
            abort "Unsupported architecture: $ARCH"
            ;;
    esac
    MACOS_VERSION=$(sw_vers -productVersion | cut -d. -f1)
}

# -- Existing installation detection --
detect_existing_install() {
    IS_UPGRADE=false
    INSTALLED_VERSION=""
    if [ -f "$INSTALL_DIR/version.txt" ]; then
        INSTALLED_VERSION=$(cat "$INSTALL_DIR/version.txt")
        IS_UPGRADE=true
    fi
}

# -- Prerequisite checks --
check_prerequisites() {
    local failures=0

    ohai "Checking prerequisites..."

    # macOS version >= 12
    if [ "$MACOS_VERSION" -lt 12 ]; then
        warn "macOS 12 or later required (found $(sw_vers -productVersion))"
        failures=$((failures + 1))
    else
        success "macOS $(sw_vers -productVersion)"
    fi

    # Disk space (3GB minimum)
    local free_kb
    free_kb=$(df -k "$HOME/Library" | tail -1 | awk '{print $4}')
    local free_gb=$((free_kb / 1024 / 1024))
    if [ "$free_gb" -lt 3 ]; then
        warn "At least 3 GB free disk space required (found ${free_gb} GB)"
        failures=$((failures + 1))
    else
        success "${free_gb} GB free disk space"
    fi

    # Internet connectivity
    if ! curl -fsS --max-time 5 https://github.com > /dev/null 2>&1; then
        warn "No internet connection. Check your network and try again."
        failures=$((failures + 1))
    else
        success "Internet connection"
    fi

    # Xcode Command Line Tools
    if ! xcode-select -p &>/dev/null; then
        warn "Xcode Command Line Tools not found. Installing..."
        xcode-select --install 2>/dev/null || true
        echo "  Please complete the installation dialog, then re-run this script."
        failures=$((failures + 1))
    else
        success "Xcode Command Line Tools"
    fi

    if [ $failures -gt 0 ]; then
        abort "Prerequisites check failed ($failures issue(s) above)."
    fi
}

# -- Download helper with retries --
download_with_retry() {
    local url="$1"
    local output="$2"
    local description="${3:-file}"
    local max_attempts=3
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        ohai "Downloading $description (attempt $attempt/$max_attempts)..."
        if curl -fSL --progress-bar --retry 2 "$url" -o "$output"; then
            success "Downloaded $description"
            return 0
        fi
        warn "Download failed (attempt $attempt/$max_attempts). Retrying..."
        attempt=$((attempt + 1))
        sleep 2
    done

    abort "Failed to download $description after $max_attempts attempts."
}

# -- Install Python from python-build-standalone --
install_python() {
    # Skip if same version already installed
    if "$PYTHON_DIR/bin/python3" --version 2>/dev/null | grep -q "$PYTHON_VERSION"; then
        success "Python $PYTHON_VERSION already installed (skipped)"
        return
    fi

    # Clean old version if present
    [ -d "$PYTHON_DIR" ] && rm -rf "$PYTHON_DIR"

    ohai "Installing Python $PYTHON_VERSION..."

    local python_url="https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_VERSION}/cpython-${PYTHON_VERSION}+${PBS_VERSION}-${PLATFORM}-install_only_stripped.tar.gz"
    local tmp_archive
    tmp_archive=$(mktemp)

    download_with_retry "$python_url" "$tmp_archive" "Python $PYTHON_VERSION"

    mkdir -p "$PYTHON_DIR"
    tar xzf "$tmp_archive" -C "$PYTHON_DIR" --strip-components=1
    rm -f "$tmp_archive"

    if ! "$PYTHON_DIR/bin/python3" --version &>/dev/null; then
        abort "Python installation failed — binary not executable."
    fi

    success "Python $("$PYTHON_DIR/bin/python3" --version 2>&1)"
}

# -- Create virtual environment --
create_venv() {
    if [ "$IS_UPGRADE" = true ] && [ -d "$VENV_DIR" ]; then
        if [ "$INSTALLED_VERSION" = "$TT_VERSION" ]; then
            # Same version — upgrade in place (preserves site-packages)
            ohai "Upgrading virtual environment..."
            "$PYTHON_DIR/bin/python3" -m venv --upgrade "$VENV_DIR"
            "$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel --quiet
            success "Virtual environment upgraded"
        else
            # Version changed — Python binary may differ, recreate cleanly
            ohai "Recreating virtual environment for v${TT_VERSION}..."
            rm -rf "$VENV_DIR"
            "$PYTHON_DIR/bin/python3" -m venv "$VENV_DIR"
            "$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel --quiet
            success "Virtual environment created"
        fi
    else
        ohai "Creating virtual environment..."
        "$PYTHON_DIR/bin/python3" -m venv "$VENV_DIR"
        "$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel --quiet
        success "Virtual environment created"
    fi
}

# -- Install Python dependencies --
install_dependencies() {
    local pip="$VENV_DIR/bin/pip"

    if [ "$IS_APPLE_SILICON" = true ]; then
        ohai "[1/4] Installing PyTorch (CPU)..."
        "$pip" install torch --index-url https://download.pytorch.org/whl/cpu --quiet || \
            warn "PyTorch install failed (non-fatal)"

        ohai "[2/4] Installing OpenAI Whisper..."
        "$pip" install openai-whisper --quiet || \
            warn "openai-whisper install failed (Slow/Very Slow tiers may be unavailable)"

        ohai "[3/4] Installing faster-whisper..."
        "$pip" install faster-whisper --quiet || \
            abort "faster-whisper installation failed."

        ohai "[4/4] Installing OpenAI client..."
        "$pip" install openai --quiet
    else
        ohai "[1/4] Installing PyTorch ${PYTORCH_VERSION_INTEL} (Intel)..."
        "$pip" install "torch==${PYTORCH_VERSION_INTEL}" \
            --index-url https://download.pytorch.org/whl/cpu --quiet 2>/dev/null || {
            warn "PyTorch install failed on Intel Mac."
            warn "Slow and Very Slow tiers will not be available."
        }

        ohai "[2/4] Installing OpenAI Whisper..."
        "$pip" install openai-whisper --quiet 2>/dev/null || \
            warn "openai-whisper install failed (non-fatal for Intel)"

        ohai "[3/4] Installing faster-whisper..."
        "$pip" install faster-whisper --quiet || \
            abort "faster-whisper installation failed."

        ohai "[4/4] Installing OpenAI client..."
        "$pip" install openai --quiet
    fi

    success "Dependencies installed"
}

# -- Install static ffmpeg binary --
install_ffmpeg() {
    # Skip if ffmpeg already installed and working
    if [ -x "$FFMPEG_DIR/ffmpeg" ] && "$FFMPEG_DIR/ffmpeg" -version &>/dev/null; then
        success "ffmpeg already installed (skipped)"
        return
    fi

    ohai "Installing ffmpeg..."

    local ffmpeg_url="https://evermeet.cx/ffmpeg/getrelease/zip"
    local ffprobe_url="https://evermeet.cx/ffmpeg/getrelease/ffprobe/zip"
    local tmp_ffmpeg tmp_ffprobe
    tmp_ffmpeg=$(mktemp)
    tmp_ffprobe=$(mktemp)

    download_with_retry "$ffmpeg_url" "$tmp_ffmpeg" "ffmpeg"
    download_with_retry "$ffprobe_url" "$tmp_ffprobe" "ffprobe"

    mkdir -p "$FFMPEG_DIR"
    unzip -qo "$tmp_ffmpeg" -d "$FFMPEG_DIR/"
    unzip -qo "$tmp_ffprobe" -d "$FFMPEG_DIR/"
    chmod +x "$FFMPEG_DIR/ffmpeg" "$FFMPEG_DIR/ffprobe"
    rm -f "$tmp_ffmpeg" "$tmp_ffprobe"

    if ! "$FFMPEG_DIR/ffmpeg" -version &>/dev/null; then
        warn "ffmpeg download failed. Checking system ffmpeg..."
        if command -v ffmpeg &>/dev/null; then
            success "Using system ffmpeg: $(which ffmpeg)"
        else
            abort "ffmpeg not found. Install with: brew install ffmpeg"
        fi
    else
        success "ffmpeg $("$FFMPEG_DIR/ffmpeg" -version | head -1 | awk '{print $3}')"
    fi
}

# -- Install transcription-tools package --
install_transcription_tools() {
    ohai "Installing transcription-tools..."
    "$VENV_DIR/bin/pip" install \
        "transcription-tools @ https://github.com/austinrojan/transcription-tools/archive/refs/tags/v${TT_VERSION}.tar.gz" \
        --quiet
    echo "$TT_VERSION" > "$INSTALL_DIR/version.txt"
    success "transcription-tools $TT_VERSION installed"
}

# -- Create wrapper scripts in /usr/local/bin --
install_wrapper_scripts() {
    ohai "Installing commands to /usr/local/bin..."

    local commands=(transcribe-veryfast transcribe-fast transcribe-medium transcription-tools)
    if [ "$IS_APPLE_SILICON" = true ]; then
        commands+=(transcribe-slow transcribe-veryslow)
    fi

    local needs_sudo=false
    if [ ! -w "/usr/local/bin" ]; then
        needs_sudo=true
        echo ""
        echo "  Creating commands in /usr/local/bin/ requires administrator access."
        echo "  This lets you run 'transcribe-fast' from anywhere in Terminal,"
        echo "  and is required for the Finder right-click integration."
        echo ""
    fi

    for cmd in "${commands[@]}"; do
        local wrapper
        wrapper=$(mktemp)
        cat > "$wrapper" << WRAPPER
#!/bin/bash
export PATH="\$HOME/Library/Application Support/transcription-tools/ffmpeg:\$PATH"
exec "\$HOME/Library/Application Support/transcription-tools/venv/bin/${cmd}" "\$@"
WRAPPER
        chmod +x "$wrapper"

        if [ "$needs_sudo" = true ]; then
            sudo mv "$wrapper" "/usr/local/bin/$cmd" || \
                abort "Failed to install $cmd (sudo required)"
        else
            mv "$wrapper" "/usr/local/bin/$cmd" || \
                abort "Failed to install $cmd"
        fi
    done

    success "Commands installed"
}

# -- Install Finder Quick Actions --
install_workflows() {
    ohai "Installing Finder Quick Actions..."

    mkdir -p "$SERVICES_DIR"

    # Download workflows from the release tarball
    local tmp_tar
    tmp_tar=$(mktemp)
    local workflows_url="https://github.com/austinrojan/transcription-tools/archive/refs/tags/v${TT_VERSION}.tar.gz"

    if ! curl -fSL --progress-bar "$workflows_url" -o "$tmp_tar" 2>/dev/null; then
        warn "Could not download workflows. Skipping Finder integration."
        return
    fi

    local tmp_extract
    tmp_extract=$(mktemp -d)
    tar xzf "$tmp_tar" -C "$tmp_extract" --strip-components=1

    local source_dir="$tmp_extract/workflows"
    if [ ! -d "$source_dir" ]; then
        warn "Workflow source not found in release. Skipping."
        rm -rf "$tmp_tar" "$tmp_extract"
        return
    fi

    local workflows=("Very Fast" "Fast" "Medium")
    if [ "$IS_APPLE_SILICON" = true ]; then
        workflows+=("Slow" "Very Slow")
    fi

    for tier in "${workflows[@]}"; do
        rm -rf "$SERVICES_DIR/Transcribe Audio - ${tier}.workflow"
        cp -R "$source_dir/Transcribe Audio - ${tier}.workflow" "$SERVICES_DIR/"
    done

    rm -rf "$tmp_tar" "$tmp_extract"
    /System/Library/CoreServices/pbs -flush 2>/dev/null || true
    success "Finder Quick Actions installed"
}

# -- UI: Banner, confirmation, success message --
print_banner() {
    echo ""
    echo "  ${BOLD}Transcription Tools Installer${RESET}"
    echo "  macOS audio & video transcription"
    echo ""
    echo "  This will install:"
    echo "    - Python $PYTHON_VERSION (self-contained)"
    echo "    - Whisper speech recognition"
    echo "    - ffmpeg for audio/video processing"
    echo "    - Right-click Quick Actions for Finder"
    echo ""
    echo "  Install location: $INSTALL_DIR"
    echo ""
}

confirm_installation() {
    if [ "$IS_UPGRADE" = true ]; then
        echo "  ${YELLOW}Existing installation detected (v${INSTALLED_VERSION}).${RESET}"
        echo "  This will upgrade to v${TT_VERSION}."
        echo ""
    fi

    if [ "$IS_APPLE_SILICON" = true ]; then
        echo "  Architecture:   Apple Silicon (arm64)"
    else
        echo "  Architecture:   Intel (x86_64)"
    fi
    echo "  macOS version:  $(sw_vers -productVersion)"

    if [ "$IS_APPLE_SILICON" = false ]; then
        echo ""
        echo "  ${YELLOW}Note: Slow and Very Slow tiers require Apple Silicon.${RESET}"
    fi

    echo ""
    if [ "$IS_UPGRADE" = true ]; then
        read -rp "  Proceed with upgrade? [Y/n] " response
    else
        read -rp "  Proceed with installation? [Y/n] " response
    fi
    case "${response:-Y}" in
        [yY]|[yY][eE][sS]|"") ;;
        *) echo "  Installation cancelled."; exit 0 ;;
    esac
}

print_success() {
    echo ""
    echo "  ${GREEN}${BOLD}Transcription Tools installed successfully!${RESET}"
    echo ""
    echo "  ${BOLD}QUICK START${RESET}"
    echo ""
    echo "  Right-click any audio or video file in Finder, then choose"
    echo "  Quick Actions > Transcribe Audio - Fast (or any tier)."
    echo ""
    echo "  Or from the command line:"
    echo ""
    echo "      transcribe-fast recording.mp3"
    echo "      transcribe-medium lecture.mp4"
    echo ""
    echo "  ${BOLD}OPTIONAL: Enable AI-powered transcript cleanup${RESET}"
    echo ""
    echo "      transcription-tools config --set-api-key"
    echo ""
    open "x-apple.systempreferences:com.apple.ExtensionsPreferences" 2>/dev/null || true
}

# -- Main entry point --
main() {
    print_banner
    detect_architecture
    detect_existing_install
    check_prerequisites
    confirm_installation
    install_python
    create_venv
    install_dependencies
    install_ffmpeg
    install_transcription_tools
    install_wrapper_scripts
    install_workflows
    print_success
}

# Allow sourcing without running main (for testing)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
