# -*- mode: python ; coding: utf-8 -*-
# TightTalk.spec — PyInstaller build spec for macOS
#
# Build requirements:
#   • Python 3.11 from python.org (NOT Homebrew — Tcl/Tk paths differ)
#   • PyInstaller 6.11.1
#   • pyinstaller-hooks-contrib 2024.11
#   • All deps installed in a fresh venv (run build_app.sh, don't use this directly)
#
# Usage:
#   bash build_app.sh        ← recommended (handles venv + ffmpeg + validation)
#   pyinstaller TightTalk.spec --clean --noconfirm   ← manual

import os
import sys
import subprocess
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

# ── sounddevice: bundle the PortAudio dylib it ships in _sounddevice_data ──────
# sounddevice loads libportaudio via cffi at runtime; without these the new
# in-app player/editor silently disable (SOUNDDEVICE_OK = False).
_sd_datas = collect_data_files("_sounddevice_data")
_sd_binaries = collect_dynamic_libs("_sounddevice_data") + collect_dynamic_libs("sounddevice")

# ── Resolve site-packages of the CURRENT Python (the build venv) ──────────────
_site_packages = subprocess.check_output(
    [sys.executable, "-c", "import site; print(site.getsitepackages()[0])"],
    text=True,
).strip()
_sp = Path(_site_packages)

# ── Spec directory (project root) ─────────────────────────────────────────────
_spec_dir = Path(SPECPATH)   # PyInstaller sets SPECPATH to the .spec file's directory

# ── Collect ctranslate2 dynamic libraries ─────────────────────────────────────
_ct2_dir = _sp / "ctranslate2"
_ct2_binaries = [
    (str(p), "ctranslate2")
    for p in _ct2_dir.glob("*.dylib")
] + [
    (str(p), "ctranslate2")
    for p in _ct2_dir.glob("*.so")
]

# ── Collect onnxruntime dynamic libraries ─────────────────────────────────────
_ort_capi = _sp / "onnxruntime" / "capi"
_ort_binaries = [
    (str(p), "onnxruntime/capi")
    for p in _ort_capi.glob("*.dylib")
] + [
    (str(p), "onnxruntime/capi")
    for p in _ort_capi.glob("*.so")
]

# ── ffmpeg static binaries (both arches; correct one selected at runtime) ──────
# NOTE: placed in datas (not binaries) so PyInstaller puts them inside
# Contents/MacOS/bin/ where sys._MEIPASS resolves them correctly.
# binaries go to Contents/Frameworks/ which is outside MEIPASS.
_ffmpeg_datas = []
for _ffbin in (_spec_dir / "bin").glob("ffmpeg-*"):
    _ffmpeg_datas.append((str(_ffbin), "bin"))
for _ffbin in (_spec_dir / "bin").glob("ffprobe-*"):
    _ffmpeg_datas.append((str(_ffbin), "bin"))

# ── Tcl/Tk data directories (auto-detected — works for python.org and Homebrew)
_tcl_root = subprocess.check_output(
    [sys.executable, "-c", "import tkinter; print(tkinter.Tcl().eval('info library'))"],
    text=True,
).strip()
_tk_root = _tcl_root.replace("/tcl8.6", "/tk8.6")

block_cipher = None

a = Analysis(
    [str(_spec_dir / "tighttalk" / "tight_talk.py")],
    pathex=[str(_spec_dir / "tighttalk")],   # so `import config` resolves
    binaries=[
        *_ct2_binaries,
        *_ort_binaries,
        *_sd_binaries,
    ],
    datas=[
        # ctranslate2 Python package (includes kernel configs)
        (str(_ct2_dir), "ctranslate2"),
        # faster-whisper tokenizer assets
        (str(_sp / "faster_whisper" / "assets"), "faster_whisper/assets"),
        # onnxruntime Python package
        (str(_sp / "onnxruntime"), "onnxruntime"),
        # Tcl/Tk data (required for tkinter on macOS)
        (_tcl_root, "lib/tcl8.6"),
        (_tk_root,  "lib/tk8.6"),
        # ffmpeg — must be in datas so it lands under sys._MEIPASS (Contents/MacOS/)
        # (binaries go to Contents/Frameworks/ which is outside MEIPASS)
        *_ffmpeg_datas,
        # sounddevice's bundled PortAudio dylib + metadata
        *_sd_datas,
    ],
    hiddenimports=[
        # ── ctranslate2
        "ctranslate2",
        "ctranslate2.specs",
        # ── faster-whisper
        "faster_whisper",
        "faster_whisper.transcribe",
        "faster_whisper.tokenizer",
        "faster_whisper.feature_extractor",
        "faster_whisper.audio",
        "faster_whisper.utils",
        "faster_whisper.vad",
        # ── onnxruntime
        "onnxruntime",
        "onnxruntime.capi",
        "onnxruntime.capi._pybind_state",
        "onnxruntime.capi.onnxruntime_inference_collection",
        # ── scipy (only what signal.welch needs)
        "scipy.signal",
        "scipy.signal._signaltools",
        "scipy.signal.windows",
        "scipy.signal.windows._windows",
        # "scipy.signal._spectral_helper",  # removed in scipy 1.13+
        "scipy.fft",
        "scipy.fft._pocketfft",
        "scipy.fft._pocketfft.helper",
        "scipy.linalg",
        "scipy.special",
        "scipy.special._ufuncs",
        "scipy.special.cython_special",
        "scipy.linalg.cython_blas",
        "scipy.linalg.cython_lapack",
        # ── numpy internals
        "numpy.core._multiarray_umath",
        "numpy.core._multiarray_tests",
        # "numpy.random.common",          # removed in numpy 2.x
        # "numpy.random.bounded_integers",  # removed in numpy 2.x
        # "numpy.random.entropy",           # removed in numpy 2.x
        # ── pydub
        "pydub",
        "pydub.utils",
        "pydub.effects",
        "pydub.silence",
        # ── sounddevice (in-app playback + waveform editor)
        "sounddevice",
        "_sounddevice",
        "_sounddevice_data",
        "cffi",
        "_cffi_backend",
        # ── TightTalk local modules (imported via guarded try/except)
        "player",
        "editor",
        # ── tkinter
        "tkinter",
        "tkinter.ttk",
        "tkinter.messagebox",
        "tkinter.filedialog",
        "tkinter.font",
        "_tkinter",
        # ── stdlib
        "queue",
        "threading",
        "platform",
        "logging.handlers",
        "importlib.metadata",
        "importlib.resources",
    ],
    hookspath=["hooks"],
    runtime_hooks=["hooks/runtime_hook_tcl.py"],
    excludes=[
        # scipy submodules not used by TightTalk (saves ~150 MB)
        "scipy.sparse",
        "scipy.sparse.linalg",
        "scipy.spatial",
        "scipy.ndimage",
        "scipy.interpolate",
        "scipy.optimize",
        "scipy.integrate",
        "scipy.io",
        "scipy.io.matlab",
        "scipy.stats",
        "scipy.cluster",
        "scipy.odr",
        "scipy.datasets",
        # Unused heavy packages
        "matplotlib",
        "IPython",
        "pytest",
        "setuptools",
        "pkg_resources",
        "distutils",
        "unittest",
        "PIL",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TightTalk",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,        # Never strip on macOS — invalidates codesigning
    upx=False,          # Never UPX on macOS arm64 — corrupts Mach-O in ctranslate2
    console=False,      # GUI app — no terminal window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,   # Builds for current machine arch; set "arm64" or "x86_64" to force
    codesign_identity=None,
    entitlements_file="entitlements.plist",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="TightTalk",
)

app = BUNDLE(
    coll,
    name="TightTalk.app",
    bundle_identifier="com.tighttalk.app",
    info_plist={
        "CFBundleName":              "TightTalk",
        "CFBundleDisplayName":       "TightTalk",
        "CFBundleIdentifier":        "com.tighttalk.app",
        "CFBundleVersion":           "1.0.0",
        "CFBundleShortVersionString":"1.0",
        "NSHighResolutionCapable":   True,
        "LSMinimumSystemVersion":    "12.0",
        "NSMicrophoneUsageDescription":
            "TightTalk processes audio files for transcription. "
            "Microphone access may be requested by macOS for audio operations.",
        "NSAppleScriptEnabled":      False,
    },
)
