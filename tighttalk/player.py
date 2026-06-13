#!/usr/bin/env python3
"""
TightTalk — sample-accurate WAV playback via sounddevice (PortAudio).

Design (per round-2 consensus + debate amendments):
  • Lazy stream: opened on first play(), not at construction — no idle
    battery drain, no CoreAudio claim before the user asks for sound.
  • Pause = stop feeding frames (stream keeps running, emits silence) so
    resume is instant. After IDLE_CLOSE_S seconds paused, the stream is
    closed and transparently reopened on next play().
  • Device-switch resilience: a PortAudioError during play triggers ONE
    reopen attempt (handles AirPods connecting mid-session). If that also
    fails, the error surfaces to the caller.
  • Position = sample counter, exact. Poll .position_sec from tkinter
    via after() for playhead animation — no clock drift.

No tkinter import here; no editor import here. Pure audio transport.
"""

from __future__ import annotations

import threading

import numpy as np

try:
    import sounddevice as sd
    SOUNDDEVICE_OK = True
except (ImportError, OSError):   # OSError: PortAudio dylib missing
    SOUNDDEVICE_OK = False


IDLE_CLOSE_S = 60.0   # close the OS stream after this long paused


class WavePlayer:
    """Play/pause/seek/region playback over a float32 numpy array.

    samples: shape (n,) mono or (n, channels), float32 in [-1, 1]
    """

    def __init__(self, samples: np.ndarray, samplerate: int):
        if samples.ndim == 1:
            samples = samples[:, None]
        self.data = np.ascontiguousarray(samples, dtype=np.float32)
        self.sr = int(samplerate)
        self.nframes = len(self.data)

        self._cursor = 0              # next frame to emit
        self._end = self.nframes      # region end (exclusive)
        self._paused = True
        self._lock = threading.Lock()
        self._stream = None
        self._idle_timer: threading.Timer | None = None
        self._on_finished = None      # optional callback when region ends

    # ── Stream lifecycle ───────────────────────────────────────────────────

    def _open_stream(self):
        self._stream = sd.OutputStream(
            samplerate=self.sr,
            channels=self.data.shape[1],
            dtype="float32",
            latency="low",
            callback=self._callback,
        )
        self._stream.start()

    def _ensure_stream(self):
        """Open the stream if needed; retry once on PortAudioError
        (covers output-device changes like AirPods connecting)."""
        if self._stream is not None:
            return
        try:
            self._open_stream()
        except sd.PortAudioError:
            sd._terminate()
            sd._initialize()          # re-enumerate devices
            self._open_stream()       # second failure propagates

    def _close_stream(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _schedule_idle_close(self):
        self._cancel_idle_close()
        self._idle_timer = threading.Timer(IDLE_CLOSE_S, self._idle_close)
        self._idle_timer.daemon = True
        self._idle_timer.start()

    def _cancel_idle_close(self):
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _idle_close(self):
        with self._lock:
            if self._paused:
                self._close_stream()

    # ── Audio callback ─────────────────────────────────────────────────────

    def _callback(self, out, frames, time_info, status):
        with self._lock:
            if self._paused or self._cursor >= self._end:
                out[:] = 0
                if not self._paused and self._cursor >= self._end:
                    self._paused = True          # region finished
                    cb = self._on_finished
                    if cb is not None:
                        # fire outside the lock via a thread — the callback
                        # runs on the PortAudio thread and must not block
                        threading.Thread(target=cb, daemon=True).start()
                return
            n = min(frames, self._end - self._cursor)
            out[:n] = self.data[self._cursor:self._cursor + n]
            out[n:] = 0
            self._cursor += n

    # ── Transport API ──────────────────────────────────────────────────────

    def play(self, start_sec: float | None = None, end_sec: float | None = None):
        """Start/resume playback. Optional region [start_sec, end_sec)."""
        if not SOUNDDEVICE_OK:
            raise RuntimeError("sounddevice not available")
        self._cancel_idle_close()
        self._ensure_stream()
        with self._lock:
            if start_sec is not None:
                self._cursor = max(0, min(self.nframes, int(start_sec * self.sr)))
            if self._cursor >= self.nframes:
                self._cursor = 0
            self._end = (self.nframes if end_sec is None
                         else max(self._cursor, min(self.nframes, int(end_sec * self.sr))))
            self._paused = False

    def pause(self):
        with self._lock:
            self._paused = True
        self._schedule_idle_close()

    def toggle(self):
        if self.playing:
            self.pause()
        else:
            self.play()

    def seek(self, sec: float):
        with self._lock:
            self._cursor = max(0, min(self.nframes, int(sec * self.sr)))

    def stop(self):
        with self._lock:
            self._paused = True
            self._cursor = 0
            self._end = self.nframes
        self._schedule_idle_close()

    def set_samples(self, samples: np.ndarray):
        """Swap the audio buffer (after an edit). Clamps cursor/region."""
        if samples.ndim == 1:
            samples = samples[:, None]
        with self._lock:
            self.data = np.ascontiguousarray(samples, dtype=np.float32)
            self.nframes = len(self.data)
            self._cursor = min(self._cursor, self.nframes)
            self._end = self.nframes

    def on_finished(self, callback):
        """Register a callback fired when a region/track finishes."""
        self._on_finished = callback

    @property
    def position_sec(self) -> float:
        with self._lock:
            return self._cursor / self.sr

    @property
    def playing(self) -> bool:
        with self._lock:
            return not self._paused

    def close(self):
        self._cancel_idle_close()
        with self._lock:
            self._paused = True
        self._close_stream()


# ── Conversion helpers (pydub ↔ numpy) ─────────────────────────────────────────

def segment_to_float(seg) -> tuple[np.ndarray, int]:
    """pydub AudioSegment → (float32 array in [-1,1], sample_rate)."""
    arr = np.array(seg.get_array_of_samples(), dtype=np.float32)
    if seg.channels > 1:
        arr = arr.reshape(-1, seg.channels)
    return arr / float(1 << (8 * seg.sample_width - 1)), seg.frame_rate


def float_to_int16(x: np.ndarray) -> np.ndarray:
    """float32 [-1,1] → int16 PCM."""
    return (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)
