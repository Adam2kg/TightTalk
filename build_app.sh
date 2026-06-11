#!/usr/bin/env bash
# build_app.sh — Build TightTalk.app for macOS
#
# Prerequisites:
#   • Python 3.11 from python.org  (https://www.python.org/downloads/)
#   • Xcode Command Line Tools     (xcode-select --install)
#   • Internet access (first run — downloads ffmpeg + pip packages)
#
# Output: dist/TightTalk.app
# Usage:  bash build_app.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "  ████████╗██╗ ██████╗ ██╗  ██╗████████╗ █████╗ ██╗     ██╗  ██╗"
echo "     ██╔══╝ ██║██╔════╝ ██║  ██║╚══██╔══╝██╔══██╗██║     ██║ ██╔╝"
echo "     ██║    ██║██║  ███╗███████║   ██║   ███████║██║     █████╔╝ "
echo "     ██║    ██║██║   ██║██╔══██║   ██║   ██╔══██║██║     ██╔═██╗ "
echo "     ██║    ██║╚██████╔╝██║  ██║   ██║   ██║  ██║███████╗██║  ██╗"
echo "     ╚═╝    ╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝"
echo "  macOS .app Builder"
echo ""

# ── Step 1: Pre-flight checks ──────────────────────────────────────────────────
echo "→ Step 1/6  Pre-flight checks"

# 1a. Verify python3.11 exists
PYTHON="$(command -v python3.11 || true)"
if [[ -z "$PYTHON" ]]; then
    echo ""
    echo "  ERROR: python3.11 not found."
    echo "  Install Python 3.11 from: https://www.python.org/downloads/"
    echo "  (Do NOT use Homebrew Python — Tcl/Tk paths differ and tkinter will break.)"
    exit 1
fi

# 1b. Verify it's the python.org installer, not Homebrew or conda
PYTHON_PREFIX="$("$PYTHON" -c "import sys; print(sys.prefix)")"
if [[ "$PYTHON_PREFIX" == *"conda"* || "$PYTHON_PREFIX" == *"miniforge"* ]]; then
    echo ""
    echo "  ERROR: python3.11 is from conda/miniforge:"
    echo "         $PYTHON_PREFIX"
    echo "  Use python.org or Homebrew Python instead."
    exit 1
fi

# If Homebrew Python, ensure python-tk@3.11 is installed
if [[ "$PYTHON_PREFIX" == *"homebrew"* ]] || [[ "$PYTHON_PREFIX" == *"Cellar"* ]]; then
    if ! python3.11 -c "import _tkinter" 2>/dev/null; then
        echo "     tkinter missing — installing python-tk@3.11 via Homebrew..."
        brew install python-tk@3.11 --quiet
    fi
fi

PYTHON_VERSION="$("$PYTHON" --version 2>&1 | awk '{print $2}')"
echo "     Python:  $PYTHON ($PYTHON_VERSION)"
echo "     Prefix:  $PYTHON_PREFIX"

# 1c. Detect build architecture
ARCH="$(uname -m)"   # 'arm64' or 'x86_64'
echo "     Arch:    $ARCH"

# ── Step 2: Clean prior build artifacts ───────────────────────────────────────
echo ""
echo "→ Step 2/6  Cleaning previous build"
rm -rf build/ dist/ __pycache__/
find . -name "*.pyc" -delete 2>/dev/null || true
echo "     Done."

# ── Step 3: Install pinned dependencies ───────────────────────────────────────
echo ""
echo "→ Step 3/6  Installing pinned dependencies"
echo "     (installing into Python 3.11 user site — ~2 min on first run)"

"$PYTHON" -m pip install --upgrade pip --quiet

"$PYTHON" -m pip install --quiet \
    "faster-whisper==1.1.1" \
    "ctranslate2==4.3.1" \
    "onnxruntime==1.20.1" \
    "pydub==0.25.1" \
    "scipy==1.13.1" \
    "numpy==1.26.4" \
    "pyinstaller==6.11.1" \
    "pyinstaller-hooks-contrib==2024.11"

echo "     Dependencies installed."

# ── Step 4: Download arch-specific static ffmpeg + ffprobe ────────────────────
echo ""
echo "→ Step 4/6  Checking bundled ffmpeg + ffprobe"

BIN_DIR="$SCRIPT_DIR/bin"
mkdir -p "$BIN_DIR"

_download_ff() {
    local TOOL="$1"   # ffmpeg or ffprobe
    local BIN="$BIN_DIR/$TOOL-$ARCH"

    if [[ -f "$BIN" ]]; then
        echo "     Already present: $BIN (skipping)"
        echo "$BIN"
        return
    fi

    echo "     Downloading static $TOOL for $ARCH..."
    local TMP
    TMP="$(mktemp -d)"

    if [[ "$ARCH" == "arm64" ]]; then
        if [[ "$TOOL" == "ffmpeg" ]]; then
            URL="https://www.osxexperts.net/ffmpeg7arm.zip"
        else
            URL="https://www.osxexperts.net/ffprobe7arm.zip"
        fi
    else
        URL="https://evermeet.cx/ffmpeg/getrelease/$TOOL/zip"
    fi

    echo "     URL: $URL"
    curl -L --fail --progress-bar "$URL" -o "$TMP/$TOOL.zip"
    unzip -q "$TMP/$TOOL.zip" -d "$TMP/extracted"

    local FOUND
    FOUND="$(find "$TMP/extracted" -type f -name "$TOOL" | head -1)"
    if [[ -z "$FOUND" ]]; then
        echo "  ERROR: Could not find $TOOL binary in archive."
        rm -rf "$TMP"
        exit 1
    fi

    cp "$FOUND" "$BIN"
    chmod +x "$BIN"
    xattr -d com.apple.quarantine "$BIN" 2>/dev/null || true
    rm -rf "$TMP"
    echo "     Saved: $BIN"
}

FFMPEG_BIN="$BIN_DIR/ffmpeg-$ARCH"
FFPROBE_BIN="$BIN_DIR/ffprobe-$ARCH"
_download_ff ffmpeg
_download_ff ffprobe

# Validate the binary runs on this arch
if ! "$FFMPEG_BIN" -version > /dev/null 2>&1; then
    echo ""
    echo "  ERROR: ffmpeg binary does not execute on $ARCH."
    echo "         Delete bin/ffmpeg-$ARCH and re-run to re-download."
    exit 1
fi
_ffmpeg_ver="$("$FFMPEG_BIN" -version 2>&1 | head -1)"
echo "     ffmpeg OK ($_ffmpeg_ver)"

# ── Step 5: Run PyInstaller ────────────────────────────────────────────────────
echo ""
echo "→ Step 5/6  Running PyInstaller"
echo "     This takes 2–5 minutes..."
"$PYTHON" -m PyInstaller TightTalk.spec --clean --noconfirm

# ── Step 6: Validate the build ────────────────────────────────────────────────
echo ""
echo "→ Step 6/6  Validating build"

APP="dist/TightTalk.app"
EXE="$APP/Contents/MacOS/TightTalk"

[[ -d "$APP" ]] || { echo "  ERROR: $APP not created."; exit 1; }
[[ -f "$EXE" ]] || { echo "  ERROR: executable missing at $EXE"; exit 1; }

# Check the binary is the right arch
EXE_ARCH="$(lipo -archs "$EXE" 2>/dev/null || echo unknown)"
if [[ "$EXE_ARCH" != "$ARCH" ]]; then
    echo "  WARNING: built for $EXE_ARCH but expected $ARCH"
else
    echo "     Arch check: $EXE_ARCH ✓"
fi

# Check ffmpeg is bundled
# PyInstaller 6.x places all datas under Contents/Frameworks/ (sys._MEIPASS)
BUNDLED_FFMPEG="$APP/Contents/Frameworks/bin/ffmpeg-$ARCH"
if [[ -f "$BUNDLED_FFMPEG" ]]; then
    echo "     ffmpeg bundled: bin/ffmpeg-$ARCH ✓"
else
    echo "  WARNING: ffmpeg-$ARCH not found in bundle at $BUNDLED_FFMPEG"
fi

APP_SIZE="$(du -sh "$APP" | cut -f1)"
echo "     Bundle size: $APP_SIZE"

# ── Ad-hoc codesigning (free, no Apple account required) ──────────────────────
echo ""
echo "→ Codesigning with ad-hoc identity (no certificate required)..."

# Sign all .dylib and .so files first (inner objects before outer container)
find "$APP" -type f \( -name "*.dylib" -o -name "*.so" \) | while read -r lib; do
    codesign --force --sign "-" --timestamp=none "$lib" 2>/dev/null || true
done

# Sign ffmpeg binaries
find "$APP" -path "*/bin/ffmpeg-*" -type f | while read -r bin; do
    codesign --force --sign "-" --timestamp=none "$bin" 2>/dev/null || true
done

# Sign the main executable
codesign --force --sign "-" --timestamp=none "$EXE"

# Sign the bundle itself (must be last)
codesign --force --sign "-" --timestamp=none --deep "$APP"

# Verify
if codesign --verify --deep --strict "$APP" 2>/dev/null; then
    echo "     Codesigning: valid ✓"
else
    echo "  WARNING: codesign verify failed — app may show Gatekeeper warning on other Macs."
    echo "           Recipients can run: xattr -cr $APP"
fi

echo ""
echo "  ✓ Build complete!"
echo "  App:  $SCRIPT_DIR/$APP"
echo "  Size: $APP_SIZE"
echo ""
echo "  To distribute:"
echo "    • Zip the .app:  zip -r TightTalk-macos-$ARCH.zip dist/TightTalk.app"
echo "    • Recipients:    unzip, right-click → Open on first launch"
echo "    • If 'damaged':  xattr -cr TightTalk.app  (in Terminal)"
echo ""
