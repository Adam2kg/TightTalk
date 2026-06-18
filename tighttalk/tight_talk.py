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
import random
import hashlib
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict

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

try:
    from player import WavePlayer, segment_to_float, SOUNDDEVICE_OK
except ImportError:
    WavePlayer = None           # type: ignore[assignment,misc]
    segment_to_float = None     # type: ignore[assignment]
    SOUNDDEVICE_OK = False

try:
    from editor import WaveformEditor
    EDITOR_OK = True
except ImportError:
    WaveformEditor = None       # type: ignore[assignment,misc]
    EDITOR_OK = False


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
    Mark words that are prosodic islands: emphasised (high-amplitude) words.
    These are protected from gap compression.

    Emphasis is measured by amplitude only. A long pause *before* a word is
    silence we want to compress, not emphasis — including it here was circular
    (a long gap flagged the next word as an island, which then protected that
    very gap), so 18/37 gaps self-protected and the sensitivity slider was
    inert. Amplitude-only protection routes long pauses into the mapping and
    restores the slider's effect.
    """
    amps = []
    for start, end, _ in words:
        chunk = samples[int(start * sr): int(end * sr)]
        amps.append(float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) > 0 else 0.0)

    mean_a, std_a = float(np.mean(amps)), float(np.std(amps))
    threshold = mean_a + sensitivity * std_a

    return [a > threshold for a in amps]


# ── Core processing ────────────────────────────────────────────────────────────

def _output_path(input_path: str) -> str:
    p = Path(input_path)
    return str(OUTPUT_DIR / f"{p.stem}_tight.wav")


LOG_PATH = Path.home() / "Movies" / "TightTalk" / "last_run.log"


@dataclass
class ProcessResult:
    """Everything the post-processing UI (Play / waveform editor) needs."""
    path: str
    splice_samples: list = field(default_factory=list)   # output-timeline offsets
    sample_rate: int = 44100
    channels: int = 1
    orig_sec: float = 0.0
    out_sec: float = 0.0


def _file_seed(input_path: str, settings: tuple) -> int:
    """Deterministic jitter seed: file content (first 4 MB + size) + settings.
    Same file + same settings → bit-identical output, so run logs stay diffable."""
    h = hashlib.sha1()
    p = Path(input_path)
    with open(p, "rb") as f:
        h.update(f.read(4 * 1024 * 1024))
    h.update(str(p.stat().st_size).encode())
    h.update(repr(settings).encode())
    return int.from_bytes(h.digest()[:8], "big")


def _rank_targets(
    gaps_ms: np.ndarray,
    floors_ms: np.ndarray,
    total_reduction: float,
    shape_retention: float,
) -> tuple[np.ndarray, float]:
    """
    Distribution-preserving gap targets (two-pass, global).

    Shrinks the log-domain gap distribution toward its mean by
    (1 - shape_retention), rescales to the reduction budget, applies
    per-gap adaptive floors, then redistributes the floor-induced deficit
    among gaps that still have headroom — capped so no gap drops below
    60% of its raw mapped target. Returns (targets_ms, shortfall_ms);
    shortfall > 0 means the budget was unreachable (logged, never iterated).
    """
    gaps = np.asarray(gaps_ms, dtype=np.float64)
    floors = np.asarray(floors_ms, dtype=np.float64)
    target_total = gaps.sum() * (1.0 - total_reduction)

    logd = np.log(np.maximum(gaps, 1.0))
    shaped = np.exp(logd.mean() + (logd - logd.mean()) * shape_retention)
    shaped *= target_total / shaped.sum()

    out = np.maximum(shaped, floors)
    deficit = out.sum() - target_total
    lower = np.maximum(floors, 0.6 * shaped)      # redistribution cap
    headroom = np.maximum(0.0, out - lower)
    if deficit > 0 and headroom.sum() > 0:
        out -= np.minimum(headroom, deficit * headroom / headroom.sum())
    out = np.minimum(out, gaps)                    # never lengthen a gap
    shortfall = max(0.0, out.sum() - target_total)
    return out, shortfall


def _snap_joint_ms(
    samples: np.ndarray,
    sr: int,
    end_ms: int,
    next_start_ms: int,
    nominal_new_ms: int,
    search_ms: int = 40,
) -> int:
    """
    Silence-snapped cut placement: the splice joint (where discarded gap
    audio meets the kept tail) lands on the quietest 10 ms frame within
    ±search_ms of its nominal position. Keeps the joint at least 100 ms
    away from the next word onset (the kept tail provides the lead-in).
    Returns the adjusted kept-tail duration in ms.
    """
    nominal_joint = next_start_ms - nominal_new_ms
    lo = max(end_ms, nominal_joint - search_ms)
    hi = min(next_start_ms - 100, nominal_joint + search_ms)
    if hi <= lo:
        return nominal_new_ms
    frame = max(1, int(0.010 * sr))
    s0 = int(lo * sr / 1000)
    best_ms, best_rms = nominal_joint, float("inf")
    for j_ms in range(lo, hi + 1, 5):
        s = int(j_ms * sr / 1000)
        chunk = samples[s:s + frame]
        if len(chunk) == 0:
            continue
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        if rms < best_rms:
            best_rms, best_ms = rms, j_ms
    return next_start_ms - best_ms


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

        progress_cb(f"Found {len(words)} words. Analysing prosody…")

        islands = (
            _prosodic_islands(words, samples, sr, island_sensitivity)
            if island_protection
            else [False] * len(words)
        )

        cut_threshold_ms = int(400 - 350 * aggression)

        # ── Env-tunable knobs (documented in every log header) ─────────────────
        curve           = os.environ.get("TIGHTTALK_CURVE", "rank")          # rank|log
        shape_retention = float(os.environ.get("TIGHTTALK_SHAPE_RETENTION", "0.85"))
        jitter_on       = os.environ.get("TIGHTTALK_JITTER", "1") != "0"
        total_reduction = 0.3 + 0.45 * aggression

        # ── Pass 1: classify every gap into its precedence tier ────────────────
        #   zero / below / protected / breath_keep  → rendered untouched
        #   breath / normal                          → mapped (get rank targets)
        gaps = []
        for i in range(len(words) - 1):
            end_ms        = int(words[i][1] * 1000)
            next_start_ms = int(words[i + 1][0] * 1000)
            gap_ms        = next_start_ms - end_ms
            if gap_ms <= 0:
                tier = "zero"
            elif gap_ms <= cut_threshold_ms:
                tier = "below"
            elif island_protection and (islands[i] or islands[i + 1]):
                tier = "protected"
            elif _is_breath(samples, sr, end_ms, next_start_ms):
                tier = "breath_keep" if breath_mode == "Keep" else "breath"
            else:
                tier = "normal"
            gaps.append({
                "i": i, "end_ms": end_ms, "next_start_ms": next_start_ms,
                "gap_ms": gap_ms, "tier": tier, "final_ms": gap_ms,
                "floor_ms": 0, "floor_src": "", "jit_ms": 0,
            })

        mapped = [g for g in gaps if g["tier"] in ("normal", "breath")]

        # ── Adaptive floors: boundary_strength from punctuation + lengthening ──
        # Final lengthening (pre-boundary syllable stretch) predicts pause size.
        # On thin data (<40 mapped words or <3 samples per token) the word-
        # duration statistics are noise — fall back to punctuation only.
        use_lengthening = len(words) >= 40
        tok_durs: dict[str, list[float]] = defaultdict(list)
        for (w_start, w_end, w_text) in words:
            tok_durs[w_text.strip().lower()].append(w_end - w_start)
        global_mean = float(np.mean([w[1] - w[0] for w in words])) or 0.25

        for g in mapped:
            w_start, w_end, w_text = words[g["i"]]
            punct = w_text.strip().rstrip('"\'»)').endswith(('.', '?', '!', '…'))
            bs = 0.5 if punct else 0.0
            src = "punct" if punct else "base"
            if use_lengthening:
                durs = tok_durs[w_text.strip().lower()]
                expected = float(np.mean(durs)) if len(durs) >= 3 else global_mean
                ll = (w_end - w_start) / max(expected, 1e-3)
                len_term = max(0.0, min(0.5, (ll - 1.0) * 1.2))
                if len_term > 0:
                    bs += len_term
                    src = "punct+len" if punct else "lengthening"
            bs = min(1.0, bs)
            g["floor_ms"] = int(150 + 250 * bs)
            g["floor_src"] = src

        # ── Targets: distribution-preserving rank mapping (or legacy log) ──────
        shortfall_ms = 0.0
        if mapped:
            arr    = np.array([g["gap_ms"] for g in mapped], dtype=np.float64)
            floors = np.array([g["floor_ms"] for g in mapped], dtype=np.float64)
            if curve == "log":
                tau = max(1.0, 800 * (1.0 - aggression * 0.7))
                targets = np.minimum(np.maximum(tau * np.log1p(arr / tau), floors), arr)
            else:
                targets, shortfall_ms = _rank_targets(
                    arr, floors, total_reduction, shape_retention)

            # Deterministic jitter: seeded by file content + settings, indexed
            # by gap rank so a changed gap doesn't reshuffle the rest.
            if jitter_on:
                settings = (round(aggression, 3), breath_mode, island_protection,
                            round(island_sensitivity, 3), curve, shape_retention)
                seed = _file_seed(input_path, settings)
                rng = random.Random(seed)
                draws = [rng.gauss(0.0, 1.0) for _ in range(len(mapped))]
                ranks = np.argsort(np.argsort(arr))      # rank of each gap
                for k, g in enumerate(mapped):
                    cap = min(0.10 * targets[k], 60.0)
                    jit = max(-cap, min(cap, draws[int(ranks[k])] * cap * 0.5))
                    g["jit_ms"] = int(jit)
                    g["final_ms"] = int(min(g["gap_ms"],
                                            max(g["floor_ms"], targets[k] + jit)))
            else:
                seed = 0
                for k, g in enumerate(mapped):
                    g["final_ms"] = int(min(g["gap_ms"],
                                            max(g["floor_ms"], targets[k])))

        # ── Pass 2: render ──────────────────────────────────────────────────────
        progress_cb("Rebuilding audio…")
        chunks: list[tuple[AudioSegment, bool]] = []   # (segment, splice_marker)

        first_ms = int(words[0][0] * 1000)
        if first_ms > 0:
            chunks.append((audio[: min(first_ms, 200)], False))

        log_gap_lines = []
        gap_by_i = {g["i"]: g for g in gaps}

        for i, (start, end, word) in enumerate(words):
            start_ms = int(start * 1000)
            end_ms   = int(end   * 1000)
            chunks.append((audio[start_ms:end_ms], False))

            if i >= len(words) - 1:
                break
            g = gap_by_i[i]
            tier, gap_ms, next_start_ms = g["tier"], g["gap_ms"], g["next_start_ms"]

            if tier == "zero":
                continue
            if tier in ("below", "protected", "breath_keep"):
                chunks.append((audio[end_ms:next_start_ms], False))
                label = {"below": "below threshold — kept",
                         "protected": "PROTECTED",
                         "breath_keep": "breath — kept"}[tier]
                log_gap_lines.append(
                    f"{i:>4}  {word.strip():<16} {gap_ms:>6}ms → {gap_ms:>6}ms  {label}")
                continue

            new_ms = g["final_ms"]
            if tier == "breath":
                if breath_mode == "Attenuate −12 dB":
                    # Keep the breath audible but ducked; head slice covers it
                    new_ms = max(new_ms, int(gap_ms * 0.35))
                    chunks.append((audio[end_ms: end_ms + new_ms] - 12, True))
                    decision = "breath — attenuated"
                else:  # "Replace with micro-silence" / "Remove": post-breath tail
                    chunks.append((audio[next_start_ms - new_ms: next_start_ms], True))
                    decision = "breath — tail kept"
            else:
                # Silence-snapped cut: joint lands on the quietest frame near
                # its nominal position (±40 ms), never inside the next onset
                new_ms = _snap_joint_ms(samples, sr, end_ms, next_start_ms, new_ms)
                new_ms = min(gap_ms, max(g["floor_ms"], new_ms))
                chunks.append((audio[next_start_ms - new_ms: next_start_ms], True))
                decision = f"compressed (floor={g['floor_ms']},{g['floor_src']}"
                decision += f", jit={g['jit_ms']:+d})" if jitter_on else ")"
            g["final_ms"] = new_ms
            log_gap_lines.append(
                f"{i:>4}  {word.strip():<16} {gap_ms:>6}ms → {new_ms:>6}ms  {decision}")

        # ── Splice with crossfades, tracking output-timeline splice points ─────
        progress_cb("Splicing with crossfades…")
        if not chunks:
            error_cb("No audio segments to assemble.")
            return

        output = chunks[0][0]
        splice_samples: list[int] = []
        for seg, mark in chunks[1:]:
            xf = CROSSFADE_MS if (len(output) > CROSSFADE_MS and len(seg) > CROSSFADE_MS) else 0
            if mark:
                splice_samples.append(int((len(output) - xf) * audio.frame_rate / 1000))
            output = output.append(seg, crossfade=xf)

        # ── Log: header (env knobs), per-gap lines, distribution summary ───────
        n_below     = sum(1 for g in gaps if g["tier"] == "below")
        n_protected = sum(1 for g in gaps if g["tier"] == "protected")
        n_breath    = sum(1 for g in gaps if g["tier"] in ("breath", "breath_keep"))
        floors_hit  = sum(1 for g in mapped if g["final_ms"] <= g["floor_ms"])

        log_lines = [
            f"TightTalk run — {Path(input_path).name}",
            f"  aggression={aggression:.2f}  breath_mode={breath_mode!r}"
            f"  island_protection={island_protection}  island_sensitivity={island_sensitivity:.2f}",
            f"  curve={curve}  shape_retention={shape_retention}  jitter={'on' if jitter_on else 'off'}"
            f"  total_reduction={total_reduction:.3f}  words={len(words)}"
            f"  lengthening={'on' if use_lengthening else 'off (<40 words)'}",
            "",
            f"{'#':>4}  {'word':<16} {'orig':>6}     {'new':>6}   decision",
            "-" * 80,
            *log_gap_lines,
        ]

        if mapped:
            before = np.array([g["gap_ms"] for g in mapped], dtype=np.float64)
            after  = np.array([g["final_ms"] for g in mapped], dtype=np.float64)
            p_before = np.percentile(before, [10, 50, 90])
            p_after  = np.percentile(after,  [10, 50, 90])
            std_b = float(np.std(np.log(np.maximum(before, 1.0))))
            std_a = float(np.std(np.log(np.maximum(after, 1.0))))
            # Spearman rank correlation: monotonicity check (healthy ≥ 0.98)
            rb = np.argsort(np.argsort(before)).astype(np.float64)
            ra = np.argsort(np.argsort(after)).astype(np.float64)
            denom = float(np.std(rb) * np.std(ra))
            spearman = float(np.mean((rb - rb.mean()) * (ra - ra.mean())) / denom) if denom > 0 else 1.0
            actual_red = 1.0 - after.sum() / before.sum()
            log_lines += [
                "",
                "== GAP DISTRIBUTION ==",
                f"population: {len(mapped)} mapped / {n_below} below-thresh / "
                f"{n_protected} protected / {n_breath} breath",
                f"            before      after       ratio",
                f"P10    {p_before[0]:>8.0f}ms  {p_after[0]:>8.0f}ms  {p_after[0]/max(p_before[0],1):>6.2f}",
                f"P50    {p_before[1]:>8.0f}ms  {p_after[1]:>8.0f}ms  {p_after[1]/max(p_before[1],1):>6.2f}",
                f"P90    {p_before[2]:>8.0f}ms  {p_after[2]:>8.0f}ms  {p_after[2]/max(p_before[2],1):>6.2f}",
                f"std(log)  {std_b:>6.3f}     {std_a:>6.3f}     {std_a/max(std_b,1e-9):>6.2f}"
                f"   <- target shape_retention={shape_retention}",
                f"rank-corr (Spearman): {spearman:.3f}   (healthy ≥ 0.98)",
                f"silence: {before.sum()/1000:.1f}s -> {after.sum()/1000:.1f}s "
                f"(reduction {actual_red:.1%}, budget {total_reduction:.1%}"
                + (f", budget missed by {shortfall_ms/1000:.1f}s — floors)" if shortfall_ms > 50 else ")"),
                f"floors hit: {floors_hit}/{len(mapped)}"
                + (f"   jitter seed={seed:x}" if jitter_on else "   jitter off"),
            ]

        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(log_lines) + "\n"
        LOG_PATH.write_text(content)
        existing = sorted(LOG_PATH.parent.glob("run_*.log"))
        archive  = LOG_PATH.parent / f"run_{len(existing) + 1:03d}.log"
        archive.write_text(content)

        out_path = _output_path(input_path)
        progress_cb(f"Exporting {Path(out_path).name}…")
        output.export(out_path, format="wav")
        done_cb(ProcessResult(
            path=out_path,
            splice_samples=splice_samples,
            sample_rate=audio.frame_rate,
            channels=audio.channels,
            orig_sec=len(audio) / 1000.0,
            out_sec=len(output) / 1000.0,
        ))

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
        self._player: WavePlayer | None = None
        self._last_result: ProcessResult | None = None
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

        # ── Done-state action bar (hidden until processing completes)
        self._done_frame = ttk.Frame(f)
        self._done_frame.grid(row=12, column=0, columnspan=3, pady=(10, 0))
        self._done_frame.grid_remove()

        self._play_btn = ttk.Button(
            self._done_frame, text="▶  Play",
            command=self._play_pause, takefocus=0,
            state="normal" if SOUNDDEVICE_OK else "disabled",
        )
        self._play_btn.grid(row=0, column=0, padx=(0, 8))

        ttk.Button(
            self._done_frame, text="Edit waveform…",
            command=self._open_editor, takefocus=0,
            state="normal" if EDITOR_OK else "disabled",
        ).grid(row=0, column=1)

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

    def _done(self, result: ProcessResult):
        self._last_result = result
        name = Path(result.path).name
        orig_m, orig_s = divmod(int(result.orig_sec), 60)
        out_m,  out_s  = divmod(int(result.out_sec),  60)
        self.status_var.set(
            f"✓  {name} — {out_m}:{out_s:02d} saved (from {orig_m}:{orig_s:02d})")
        self.process_btn.configure(state="normal")
        self._play_btn.configure(text="▶  Play")
        self._done_frame.grid()

        if SOUNDDEVICE_OK and segment_to_float is not None:
            if self._player is not None:
                self._player.close()
            seg = AudioSegment.from_file(result.path)
            samples, sr = segment_to_float(seg)
            self._player = WavePlayer(samples, sr)
            self._player.on_finished(
                lambda: self.after(0, lambda: self._play_btn.configure(text="▶  Play")))

    def _play_pause(self):
        if self._player is None:
            return
        if self._player.playing:
            self._player.pause()
            self._play_btn.configure(text="▶  Play")
        else:
            self._player.play()
            self._play_btn.configure(text="⏸  Pause")

    def _open_editor(self):
        if self._last_result is None or not EDITOR_OK or WaveformEditor is None:
            return
        if self._player is not None and self._player.playing:
            self._player.pause()
            self._play_btn.configure(text="▶  Play")
        player = self._player
        if player is None and WavePlayer is not None and segment_to_float is not None:
            # No sounddevice: create a data-only player so the editor can load audio.
            # Playback inside the editor will raise RuntimeError and be caught there.
            seg = AudioSegment.from_file(self._last_result.path)
            samples, sr = segment_to_float(seg)
            player = WavePlayer(samples, sr)
        if player is None:
            messagebox.showinfo(
                "TightTalk",
                "Install sounddevice to enable the waveform editor.\n"
                "    pip install sounddevice")
            return
        WaveformEditor(self, self._last_result, player)

    def destroy(self):
        if self._player is not None:
            self._player.close()
        super().destroy()

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
