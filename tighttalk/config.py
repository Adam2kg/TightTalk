#!/usr/bin/env python3
"""
TightTalk — centralised path configuration.

Works in two modes:
  • Dev mode   : running `python3 tighttalk/tight_talk.py` directly
  • Frozen mode: inside a PyInstaller .app bundle (sys._MEIPASS is set)

All other modules should import paths from here — never hardcode paths.
"""

from __future__ import annotations

import sys
import platform
from pathlib import Path


# ── Base directory ─────────────────────────────────────────────────────────────

def _base() -> Path:
    """
    Frozen:  sys._MEIPASS  (temp dir where PyInstaller unpacks the bundle)
    Dev:     project root  (two levels up from this file: tighttalk/ → TightTalk/)
    """
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent


BASE_DIR: Path = _base()
BIN_DIR:  Path = BASE_DIR / "bin"


# ── User-facing directories (always writable) ──────────────────────────────────

# Whisper model cache root — lives outside the bundle so it survives app
# updates and is never re-downloaded when a new TightTalk version is installed.
# faster-whisper uses HuggingFace hub layout under this directory:
#   MODEL_DIR/models--Systran--faster-whisper-base/snapshots/<hash>/model.bin
MODEL_DIR: Path = (
    Path.home() / "Library" / "Application Support" / "TightTalk" / "models"
)

# Processed audio output:
#   • Dev mode  → project/output/        (keeps project tidy)
#   • Frozen    → ~/Movies/TightTalk/    (no TCC permission prompt; conventional
#                                         for media apps; survives app reinstalls)
if hasattr(sys, "_MEIPASS"):
    OUTPUT_DIR: Path = Path.home() / "Movies" / "TightTalk"
else:
    OUTPUT_DIR = BASE_DIR / "output"


# ── Helper functions ───────────────────────────────────────────────────────────

def resource_path(relative: str) -> Path:
    """
    Resolve a path that is bundled inside the app.

    In dev mode:  BASE_DIR / relative  (project root)
    In frozen:    sys._MEIPASS / relative

    Use for: bundled ffmpeg, any data file included via TightTalk.spec datas/binaries.
    Do NOT use for MODEL_DIR or OUTPUT_DIR — those live outside the bundle.
    """
    return BASE_DIR / relative


def ffmpeg_path() -> Path:
    """
    Return the path to the bundled static ffmpeg binary for this CPU arch.

    Arch detection uses platform.machine() which returns:
      'arm64'   — Apple Silicon (M1/M2/M3/M4)
      'x86_64'  — Intel Mac

    Raises FileNotFoundError if the expected binary is not present.
    This is intentional: callers should surface a clear error rather than
    silently falling back to a missing system ffmpeg.
    """
    arch = platform.machine()   # 'arm64' or 'x86_64'
    binary_name = f"ffmpeg-{arch}"
    path = BIN_DIR / binary_name
    if not path.exists():
        raise FileNotFoundError(
            f"Bundled ffmpeg binary not found: {path}\n"
            f"Expected a static ffmpeg build for arch '{arch}' at bin/{binary_name}.\n"
            f"Run build_app.sh to download it, or install ffmpeg system-wide for dev use."
        )
    return path


def model_is_present() -> bool:
    """
    Return True if the configured Whisper model has already been downloaded.

    faster-whisper uses HuggingFace hub layout under MODEL_DIR:
      MODEL_DIR/models--Systran--faster-whisper-<size>/snapshots/<hash>/model.bin

    We glob for model.bin anywhere under the expected cache subdirectory
    and verify the file is larger than 1 MB (guards against corrupt stubs).
    """
    import os
    model_size = os.environ.get("TIGHTTALK_MODEL", "base")
    cache_root = MODEL_DIR / f"models--Systran--faster-whisper-{model_size}"
    try:
        matches = list(cache_root.glob("snapshots/*/model.bin"))
        return any(
            m.exists() and m.stat().st_size > 1_000_000
            for m in matches
        )
    except OSError:
        return False


def ensure_dirs() -> None:
    """
    Create directories that must exist before the app runs.
    Safe to call multiple times (idempotent).
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
