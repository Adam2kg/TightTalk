#!/usr/bin/env python3
"""
TightTalk — Audio cleaner for Reels-style speech.
Accepts: M4A, WAV (and any format ffmpeg supports). Output: WAV.

Install dependencies:
    pip install faster-whisper pydub numpy scipy
    # ffmpeg: bundled in bin/ for .app, or install system-wide for dev use

Usage:
    python3 tight_talk.py
"""

from __future__ import annotations

import os
import sys
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

# ── Config must be imported before pydub (ffmpeg wiring happens here) ──────────
# Support both `python3 tight_talk.py` (adds tighttalk/ to sys.path automatically)
# and PyInstaller frozen mode (all files extracted to _MEIPASS flat directory).
try:
    from config import (
        OUTPUT_DIR, MODEL_DIR, model_is_present, ffmpeg_path, ensure_dirs
    )
except ModuleNotFoundError:
    # Fallback for edge cases: add this file's directory to path, retry
    sys.path.insert(0, str(Path(__file__).parent))
    from config import (
        OUTPUT_DIR, MODEL_DIR, model_is_present, ffmpeg_path, ensure_dirs
    )

import numpy as np
from scipy import signal
from pydub import AudioSegment

try:
    from faster_whisper import WhisperModel
    WHISPER_OK = True
except ImportError:
    WHISPER_OK = False


# ── Wire bundled ffmpeg/ffprobe to pydub ──────────────────────────────────────
# pydub.utils.get_prober_name() uses shutil.which() — it ignores
# AudioSegment.ffprobe entirely. We must monkey-patch the function directly.
import pydub.utils as _pydub_utils

try:
    _ffmpeg  = str(ffmpeg_path())
    _ffprobe = _ffmpeg.replace("ffmpeg-", "ffprobe-")

    AudioSegment.converter = _ffmpeg
    AudioSegment.ffmpeg    = _ffmpeg

    if Path(_ffprobe).exists():
        AudioSegment.ffprobe = _ffprobe
        _pydub_utils.get_prober_name = lambda: _ffprobe

except FileNotFoundError:
    pass   # dev mode without bin/ — rely on system ffmpeg/ffprobe


# ── Constants ──────────────────────────────────────────────────────────────────

CROSSFADE_MS = 25
BREATH_MODES = [
    "Keep",
    "Attenuate −12 dB",
    "Replace with micro-silence",
    "Remove",
]

# ── Audio analysis ─────────────────────────────────────────────────────────────

def _mono_samples(audio: AudioSegment) -> np.ndarray:
    raw = np.array(audio.get_array_of_samples(), dtype=np.float32)
    if audio.channels == 2:
        raw = raw.reshape(-1, 2).mean(axis=1)
    peak = np.max(np.abs(raw))
    return raw / peak if peak > 0 else raw


def _is_breath(samples: np.ndarray, sr: int, start_ms: int, end_ms: int) -> bool:
    """
    Breath fingerprint: wide-band energy concentrated in 2–8 kHz
    with relatively little low-frequency content (< 400 Hz).
    """
    if end_ms - start_ms < 60:
        return False
    s, e = int(start_ms * sr / 1000), int(end_ms * sr / 1000)
    chunk = samples[s:e]
    if len(chunk) < 256:
        return False
    nperseg = min(512, len(chunk))
    freqs, psd = signal.welch(chunk, fs=sr, nperseg=nperseg)
    breath_band = (freqs >= 2000) & (freqs <= 8000)
    low_band    = freqs < 400
    if not breath_band.any() or not low_band.any():
        return False
    return float(psd[breath_band].mean()) > float(psd[low_band].mean()) * 1.8


def _prosodic_islands(
    words: list,
    samples: np.ndarray,
    sr: int,
    sensitivity: float,
) -> list:
    """
    Mark words that are prosodic islands: high amplitude or preceded by a
    longer-than-average pause in the original recording.
    These are protected from gap compression.
    """
    amps = []
    for start, end, _ in words:
        chunk = samples[int(start * sr): int(end * sr)]
        amps.append(float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) > 0 else 0.0)

    mean_a, std_a = float(np.mean(amps)), float(np.std(amps))
    threshold = mean_a + sensitivity * std_a

    islands = []
    for i, (start, _, _) in enumerate(words):
        high_amp   = amps[i] > threshold
        long_pause = (i > 0) and (start - words[i - 1][1] > 0.35)
        islands.append(high_amp or long_pause)
    return islands


# ── Core processing ────────────────────────────────────────────────────────────

def _output_path(input_path: str) -> str:
    p = Path(input_path)
    return str(OUTPUT_DIR / f"{p.stem}_tight.wav")


LOG_PATH = Path.home() / "Movies" / "TightTalk" / "last_run.log"


def process(
    input_path: str,
    aggression: float,
    breath_mode: str,
    island_protection: bool,
    island_sensitivity: float,
    progress_cb,
    done_cb,
    error_cb,
):
    try:
        progress_cb("Loading audio…")
        audio   = AudioSegment.from_file(input_path)
        samples = _mono_samples(audio)
        sr      = audio.frame_rate

        model_size = os.environ.get("TIGHTTALK_MODEL", "base")
        progress_cb(f"Transcribing with Whisper ({model_size})…")
        model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            download_root=str(MODEL_DIR),
        )
        segs, info = model.transcribe(
            input_path,
            word_timestamps=True,
            language="pt",
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        words = []
        for seg in segs:
            pct = min(99, int(seg.end / info.duration * 100))
            progress_cb(f"Transcribing… {pct}%")
            words.extend((w.start, w.end, w.word) for w in seg.words)

        if not words:
            error_cb("Whisper found no speech in this file.")
            return

        # ── Sample room tone from pre-speech silence ───────────────────────────
        # Use the first 500 ms before the first word as ambient background.
        # Falls back to true silence if the recording starts immediately.
        first_word_ms  = int(words[0][0] * 1000)
        room_tone_ms   = min(500, first_word_ms)
        _room_src      = audio[:room_tone_ms] if room_tone_ms >= 50 else AudioSegment.silent(500)

        def _room_tone(duration_ms: int) -> AudioSegment:
            """Tile room-tone sample to fill duration_ms."""
            if duration_ms <= 0:
                return AudioSegment.empty()
            out = AudioSegment.empty()
            while len(out) < duration_ms:
                out += _room_src
            return out[:duration_ms]

        progress_cb(f"Found {len(words)} words. Analysing prosody…")

        islands = (
            _prosodic_islands(words, samples, sr, island_sensitivity)
            if island_protection
            else [False] * len(words)
        )

        # pause_floor: minimum silence kept after compression
        #   0.0 → 250 ms (barely touched)
        #   0.5 → 200 ms (moderate)
        #   1.0 → 150 ms (tight but natural)
        pause_floor_ms = int(250 - 100 * aggression)

        # protected_max: ceiling for pauses around emphasis moments
        #   keeps sentence-ending beats from sounding chopped
        #   0.0 → 800 ms   0.5 → 550 ms   1.0 → 300 ms
        protected_max_ms = int(800 - 500 * aggression)

        # ── Build processing log ───────────────────────────────────────────────
        log_lines = [
            f"TightTalk run — {Path(input_path).name}",
            f"  aggression={aggression:.2f}  breath_mode={breath_mode!r}"
            f"  island_protection={island_protection}  island_sensitivity={island_sensitivity:.2f}",
            f"  pause_floor={pause_floor_ms} ms  words={len(words)}",
            "",
            f"{'#':>4}  {'word':<18} {'orig_gap':>8}  {'new_gap':>8}  {'type':<28}  {'island'}",
            "-" * 80,
        ]

        progress_cb("Rebuilding audio…")
        chunks: list[AudioSegment] = []

        # Brief lead-in before first word (capped at 200 ms)
        first_ms = int(words[0][0] * 1000)
        if first_ms > 0:
            chunks.append(audio[: min(first_ms, 200)])

        for i, (start, end, word) in enumerate(words):
            start_ms = int(start * 1000)
            end_ms   = int(end   * 1000)
            chunks.append(audio[start_ms:end_ms])

            if i >= len(words) - 1:
                break

            next_start_ms = int(words[i + 1][0] * 1000)
            gap_ms        = next_start_ms - end_ms

            # 0ms gaps: words run naturally together in the original speech.
            # Keep the original audio slice (no synthetic silence) — just crossfade.
            if gap_ms <= 0:
                continue

            # Short gaps below the cut threshold: keep original audio, no compression.
            # Threshold scales with aggression — at 0 nothing is cut, at 1 only
            # gaps >50ms are candidates.
            #   aggression 0.0 → threshold 400ms (barely anything touched)
            #   aggression 0.5 → threshold 225ms
            #   aggression 1.0 → threshold  50ms
            cut_threshold_ms = int(400 - 350 * aggression)
            if gap_ms <= cut_threshold_ms:
                chunks.append(audio[end_ms:next_start_ms])
                log_lines.append(
                    f"{i:>4}  {word.strip():<18} {gap_ms:>7}ms  {gap_ms:>7}ms  {'below threshold — kept':<28}"
                )
                continue

            gap_breath  = _is_breath(samples, sr, end_ms, next_start_ms)
            protect     = island_protection and (islands[i] or islands[i + 1])
            island_flag = ("←island" if islands[i] else "") + ("→island" if islands[i + 1] else "")

            if gap_breath:
                if breath_mode == "Keep":
                    chunks.append(audio[end_ms:next_start_ms])
                    new_ms   = gap_ms
                    decision = "breath — kept"
                elif breath_mode == "Attenuate −12 dB":
                    new_ms = max(pause_floor_ms, int(gap_ms * 0.35))
                    chunks.append(audio[end_ms: end_ms + new_ms] - 12)
                    decision = "breath — attenuated"
                elif breath_mode == "Replace with micro-silence":
                    # Use the tail of the gap (post-breath air) rather than silence
                    new_ms  = max(pause_floor_ms, min(100, gap_ms))
                    tail_ms = max(0, next_start_ms - new_ms)
                    chunks.append(audio[tail_ms:next_start_ms])
                    decision = "breath — tail kept"
                else:  # Remove
                    new_ms = pause_floor_ms
                    chunks.append(audio[next_start_ms - new_ms : next_start_ms])
                    decision = "breath — trimmed to tail"
            else:
                if protect:
                    new_ms   = max(pause_floor_ms, min(gap_ms, protected_max_ms))
                    decision = "silence — protected"
                else:
                    tau = max(1.0, 800 * (1.0 - aggression * 0.7))
                    compressed = int(tau * np.log1p(gap_ms / tau))
                    is_sentence_end = word.strip().rstrip('"\'»)').endswith(('.', '?', '!', '…', '...'))
                    if is_sentence_end:
                        compressed = int(compressed * 1.4)
                        decision = "silence — compressed (sentence)"
                    else:
                        decision = "silence — compressed"
                    new_ms = max(pause_floor_ms, compressed)
                # Keep the END of the gap (natural lead-in to next word)
                chunks.append(audio[next_start_ms - new_ms : next_start_ms])

            log_lines.append(
                f"{i:>4}  {word.strip():<18} {gap_ms:>7}ms  {new_ms:>7}ms  {decision:<28}  {island_flag}"
            )

        # Write log — overwrite last_run.log AND append a numbered archive
        from datetime import datetime
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(log_lines) + "\n"
        LOG_PATH.write_text(content)
        # numbered archive: run_001.log, run_002.log, …
        existing = sorted(LOG_PATH.parent.glob("run_*.log"))
        next_n   = len(existing) + 1
        archive  = LOG_PATH.parent / f"run_{next_n:03d}.log"
        archive.write_text(content)

        progress_cb("Splicing with crossfades…")
        if not chunks:
            error_cb("No audio segments to assemble.")
            return

        output = chunks[0]
        for chunk in chunks[1:]:
            xf = CROSSFADE_MS if (len(output) > CROSSFADE_MS and len(chunk) > CROSSFADE_MS) else 0
            output = output.append(chunk, crossfade=xf)

        out_path = _output_path(input_path)
        progress_cb(f"Exporting {Path(out_path).name}…")
        output.export(out_path, format="wav")
        done_cb(out_path)

    except Exception as exc:
        import traceback
        error_cb(f"{exc}\n\n{traceback.format_exc()}")


# ── First-run: Whisper model download splash ───────────────────────────────────

class DownloadSplash:
    """
    Shown on first launch when the Whisper model hasn't been downloaded yet.
    Blocks until the download completes (or fails), then destroys itself so
    the main App window can open.
    """

    BG     = "#1e1e2e"
    FG     = "#cdd6f4"
    ACCENT = "#89b4fa"
    YELLOW = "#f9e2af"

    def __init__(self) -> None:
        self._q: queue.Queue = queue.Queue()
        self._error: str | None = None

        self.root = tk.Tk()
        self.root.title("TightTalk — First Launch")
        self.root.configure(bg=self.BG)
        self.root.resizable(False, False)
        self.root.geometry("420x160")
        self._center(self.root)

        tk.Label(
            self.root, text="TightTalk",
            bg=self.BG, fg=self.ACCENT,
            font=("Helvetica", 18, "bold"),
        ).pack(pady=(20, 4))

        tk.Label(
            self.root, text="Downloading Whisper model (~150 MB) — one-time setup…",
            bg=self.BG, fg=self.FG, font=("Helvetica", 11),
        ).pack()

        self._bar = ttk.Progressbar(
            self.root, orient="horizontal",
            length=380, mode="indeterminate",
        )
        self._bar.pack(padx=20, pady=10)
        self._bar.start(10)

        self._status_var = tk.StringVar(value="Connecting…")
        tk.Label(
            self.root, textvariable=self._status_var,
            bg=self.BG, fg=self.YELLOW, font=("Helvetica", 10),
        ).pack()

        threading.Thread(target=self._download_worker, daemon=True).start()
        self.root.after(150, self._poll)
        self.root.mainloop()

        if self._error:
            raise RuntimeError(self._error)

    @staticmethod
    def _center(win: tk.Tk) -> None:
        win.update_idletasks()
        w, h = win.winfo_width(), win.winfo_height()
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        win.geometry(f"+{(sw - w) // 2}+{(sh - h) // 2}")

    def _download_worker(self) -> None:
        try:
            self._q.put(("status", "Downloading model…"))
            WhisperModel(
                os.environ.get("TIGHTTALK_MODEL", "base"),
                device="cpu",
                compute_type="int8",
                download_root=str(MODEL_DIR),   # HF hub cache root
            )
            self._q.put(("done", None))
        except Exception as exc:
            self._q.put(("error", str(exc)))

    def _poll(self) -> None:
        try:
            while True:
                kind, data = self._q.get_nowait()
                if kind == "done":
                    self._bar.stop()
                    self.root.destroy()
                    return
                elif kind == "error":
                    self._bar.stop()
                    self._status_var.set(f"Error: {data}")
                    self._error = data
                    tk.Button(
                        self.root, text="Quit",
                        command=self.root.destroy,
                        bg="#f38ba8", fg=self.BG,
                    ).pack(pady=6)
                    return
                elif kind == "status":
                    self._status_var.set(data)
        except queue.Empty:
            pass
        self.root.after(150, self._poll)


# ── Main application window ────────────────────────────────────────────────────

BG      = "#1e1e2e"
FG      = "#cdd6f4"
ACCENT  = "#89b4fa"
GREEN   = "#a6e3a1"
YELLOW  = "#f9e2af"
SURFACE = "#313244"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TightTalk")
        self.configure(bg=BG)
        self.resizable(False, False)
        self._setup_style()
        self._build_ui()
        if not WHISPER_OK:
            self.status_var.set("⚠  Run: pip install faster-whisper pydub numpy scipy")
            self.process_btn.configure(state="disabled")

    def _setup_style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".",             background=BG, foreground=FG)
        s.configure("TFrame",        background=BG)
        s.configure("TLabel",        background=BG, foreground=FG,     font=("Helvetica", 12))
        s.configure("Head.TLabel",   background=BG, foreground=ACCENT, font=("Helvetica", 20, "bold"))
        s.configure("File.TLabel",   background=BG, foreground=GREEN,  font=("Helvetica", 11))
        s.configure("Status.TLabel", background=BG, foreground=YELLOW, font=("Helvetica", 11))
        s.configure("TButton",       background=ACCENT, foreground=BG,
                    font=("Helvetica", 12, "bold"), padding=8, relief="flat")
        s.map("TButton",             background=[("disabled", SURFACE), ("active", "#74c7ec")])
        s.configure("TCheckbutton",  background=BG, foreground=FG, font=("Helvetica", 12))
        s.configure("TScale",        background=BG, troughcolor=SURFACE)
        s.configure("TCombobox",     fieldbackground=SURFACE, foreground=FG, background=SURFACE)
        s.configure("TSeparator",    background=SURFACE)

    def _build_ui(self):
        P = 16
        f = ttk.Frame(self, padding=P)
        f.grid(row=0, column=0)

        # ── Header
        ttk.Label(f, text="TightTalk", style="Head.TLabel").grid(
            row=0, column=0, columnspan=3, pady=(0, 16))

        # ── File picker
        ttk.Label(f, text="Input audio").grid(row=1, column=0, sticky="w")
        self.file_var = tk.StringVar(value="No file selected")
        ttk.Label(f, textvariable=self.file_var, style="File.TLabel",
                  wraplength=260, anchor="w").grid(row=1, column=1, sticky="w", padx=8)
        ttk.Button(f, text="Browse…", command=self._browse).grid(row=1, column=2)

        ttk.Separator(f).grid(row=2, column=0, columnspan=3, sticky="ew", pady=14)

        # ── Aggression
        ttk.Label(f, text="Aggression").grid(row=3, column=0, sticky="w")
        self.aggression_var = tk.DoubleVar(value=0.4)
        ttk.Scale(f, from_=0.0, to=1.0, variable=self.aggression_var,
                  orient="horizontal", length=210).grid(row=3, column=1, sticky="ew", padx=8)
        self._agg_lbl = ttk.Label(f, text="0.70", width=5)
        self._agg_lbl.grid(row=3, column=2, sticky="w")
        self.aggression_var.trace_add(
            "write", lambda *_: self._agg_lbl.configure(
                text=f"{self.aggression_var.get():.2f}"))

        ttk.Label(f, text="  0 = gentle tightening · 1 = full Reels pace",
                  font=("Helvetica", 9), foreground=SURFACE).grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(0, 8))

        # ── Breath mode
        ttk.Label(f, text="Breath mode").grid(row=5, column=0, sticky="w")
        self.breath_var = tk.StringVar(value=BREATH_MODES[2])
        ttk.Combobox(f, textvariable=self.breath_var, values=BREATH_MODES,
                     state="readonly", width=30).grid(
            row=5, column=1, columnspan=2, sticky="w", padx=8)

        ttk.Separator(f).grid(row=6, column=0, columnspan=3, sticky="ew", pady=14)

        # ── Island protection
        self.island_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="Protect emphasis moments (prosodic islands)",
                        variable=self.island_var, command=self._toggle_island).grid(
            row=7, column=0, columnspan=3, sticky="w")

        ttk.Label(f, text="Sensitivity").grid(row=8, column=0, sticky="w", pady=(8, 0))
        self.sens_var       = tk.DoubleVar(value=0.5)
        self._island_scale  = ttk.Scale(f, from_=0.0, to=1.5, variable=self.sens_var,
                                        orient="horizontal", length=210)
        self._island_scale.grid(row=8, column=1, sticky="ew", padx=8, pady=(8, 0))
        self._sens_lbl = ttk.Label(f, text="0.50", width=5)
        self._sens_lbl.grid(row=8, column=2, sticky="w", pady=(8, 0))
        self.sens_var.trace_add(
            "write", lambda *_: self._sens_lbl.configure(
                text=f"{self.sens_var.get():.2f}"))

        ttk.Separator(f).grid(row=9, column=0, columnspan=3, sticky="ew", pady=14)

        # ── Process
        self.process_btn = ttk.Button(f, text="  Process  ", command=self._start)
        self.process_btn.grid(row=10, column=0, columnspan=3, pady=(0, 10), ipadx=10)

        # ── Status
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(f, textvariable=self.status_var, style="Status.TLabel",
                  wraplength=360).grid(row=11, column=0, columnspan=3)

    def _toggle_island(self):
        self._island_scale.configure(
            state="normal" if self.island_var.get() else "disabled")

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[
                ("Audio files", "*.m4a *.wav *.mp3 *.aac *.flac *.ogg"),
                ("M4A files",   "*.m4a"),
                ("WAV files",   "*.wav"),
                ("All files",   "*.*"),
            ],
        )
        if path:
            self.file_var.set(path)

    def _start(self):
        path = self.file_var.get()
        if not path or path == "No file selected" or not os.path.isfile(path):
            messagebox.showerror("TightTalk", "Please select an audio file first.")
            return
        self.process_btn.configure(state="disabled")
        threading.Thread(
            target=process,
            args=(
                path,
                self.aggression_var.get(),
                self.breath_var.get(),
                self.island_var.get(),
                self.sens_var.get(),
                lambda msg: self.after(0, lambda m=msg: self.status_var.set(m)),
                lambda out: self.after(0, lambda o=out: self._done(o)),
                lambda err: self.after(0, lambda e=err: self._error(e)),
            ),
            daemon=True,
        ).start()

    def _done(self, out_path: str):
        name = Path(out_path).name
        self.status_var.set(f"✓  Saved: {name}")
        self.process_btn.configure(state="normal")
        messagebox.showinfo("TightTalk", f"Done!\n\n{out_path}")

    def _error(self, err: str):
        self.status_var.set("Error — see dialog")
        self.process_btn.configure(state="normal")
        messagebox.showerror("TightTalk — Error", err)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    # 1. Create required directories (MODEL_DIR, OUTPUT_DIR)
    ensure_dirs()

    # 2. First-run: download Whisper model if not present
    if WHISPER_OK and not model_is_present():
        try:
            DownloadSplash()          # blocks until done
        except RuntimeError as exc:
            # Download failed — show a plain error and exit
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "TightTalk — Setup Failed",
                f"Could not download the Whisper model:\n\n{exc}\n\n"
                "Check your internet connection and try again.",
            )
            root.destroy()
            return

    # 3. Launch main window
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
