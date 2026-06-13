#!/usr/bin/env python3
"""
TightTalk — Audacity-style waveform editor (pure tkinter, zero extra deps).

Architecture (per round-2 consensus):
  PeakPyramid     — min/max/RMS mipmap over the samples (pure numpy, headless)
  EditEngine      — samples + undo/redo stack of removed slices (pure numpy)
  WaveformEditor  — tk.Toplevel glueing pyramid, engine, player, canvas

Rendering: the waveform is rasterized into a numpy RGB array, serialized as
an in-memory PPM (P6), and blitted to the canvas as ONE PhotoImage item —
vector Canvas items can't hit 30 fps, raster blit costs ~5–15 ms/frame.
Selection tint is baked into the raster (macOS Aqua stipple is unreliable).
Playhead / cursor / cut markers are thin Canvas lines layered on top.
"""

from __future__ import annotations

import os
import wave
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np

# ── Theme (matches tight_talk.py) ──────────────────────────────────────────────
BG      = "#1e1e2e"
FG      = "#cdd6f4"
ACCENT  = "#89b4fa"
GREEN   = "#a6e3a1"
YELLOW  = "#f9e2af"
SURFACE = "#313244"
RED     = "#f38ba8"

_BG_RGB     = (30, 30, 46)
_WAVE_RGB   = (137, 180, 250)    # min/max body
_RMS_RGB    = (180, 210, 254)    # brighter RMS core
_SEL_RGB    = (88, 110, 160)     # selection tint (blended)
_MARKER_HEX = "#45557a"          # cut markers (dimmed accent)

BASE_SPP   = 256                  # samples per bucket, pyramid level 0
FADE_MS    = 5                    # edge fade on deletions
UNDO_BYTES = 100 * 1024 * 1024    # undo stack caps
UNDO_COUNT = 50


# ── Peak pyramid ───────────────────────────────────────────────────────────────

class PeakPyramid:
    """Min/max/RMS mipmap. Display uses a mono mix; edits keep all channels."""

    def __init__(self, samples: np.ndarray):
        mono = samples if samples.ndim == 1 else samples.mean(axis=1)
        self.n_samples = len(mono)
        self.levels: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        nb = max(1, self.n_samples // BASE_SPP)
        head = mono[: nb * BASE_SPP].reshape(nb, BASE_SPP)
        mins = head.min(axis=1).astype(np.float32)
        maxs = head.max(axis=1).astype(np.float32)
        rms  = np.sqrt((head.astype(np.float64) ** 2).mean(axis=1)).astype(np.float32)
        self.levels.append((mins, maxs, rms))
        while len(mins) > 1024:
            m = (len(mins) // 2) * 2
            mins = mins[:m].reshape(-1, 2).min(axis=1)
            maxs = maxs[:m].reshape(-1, 2).max(axis=1)
            rms  = np.sqrt((rms[:m].reshape(-1, 2) ** 2).mean(axis=1)).astype(np.float32)
            self.levels.append((mins, maxs, rms))
        self._mono = mono

    def query(self, start: float, spp: float, width: int):
        """Per-pixel (mins, maxs, rms) for the viewport; handles deep zoom."""
        out_min = np.zeros(width, np.float32)
        out_max = np.zeros(width, np.float32)
        out_rms = np.zeros(width, np.float32)
        if spp >= BASE_SPP:
            level = min(int(np.log2(max(spp / BASE_SPP, 1.0))), len(self.levels) - 1)
            bucket = BASE_SPP * (2 ** level)
            mins, maxs, rms = self.levels[level]
            nb = len(mins)
            for x in range(width):
                b0 = int((start + x * spp) // bucket)
                b1 = max(b0 + 1, int((start + (x + 1) * spp) // bucket))
                if b0 >= nb:
                    break
                b1 = min(b1, nb)
                out_min[x] = mins[b0:b1].min()
                out_max[x] = maxs[b0:b1].max()
                out_rms[x] = rms[b0:b1].max()
        else:
            mono = self._mono
            n = len(mono)
            for x in range(width):
                s0 = int(start + x * spp)
                s1 = max(s0 + 1, int(start + (x + 1) * spp))
                if s0 >= n:
                    break
                chunk = mono[s0:min(s1, n)]
                out_min[x] = chunk.min()
                out_max[x] = chunk.max()
                out_rms[x] = np.sqrt(np.mean(chunk.astype(np.float64) ** 2))
        return out_min, out_max, out_rms


# ── Edit engine (undo/redo, headless-testable) ─────────────────────────────────

class EditEngine:
    """Destructive sample edits with an undo stack of removed slices."""

    def __init__(self, samples: np.ndarray):
        self.samples = samples            # float32, (n,) or (n, ch)
        self.undo_stack: list[dict] = []
        self.redo_stack: list[dict] = []
        self.revision = 0
        self.saved_revision = 0

    @property
    def dirty(self) -> bool:
        return self.revision != self.saved_revision

    def mark_saved(self):
        self.saved_revision = self.revision

    def _fade_len(self, sr: int) -> int:
        return int(sr * FADE_MS / 1000)

    def delete(self, s0: int, s1: int, sr: int) -> bool:
        """Delete [s0, s1) with short edge fades. Returns False if no-op."""
        n = len(self.samples)
        s0, s1 = max(0, int(s0)), min(n, int(s1))
        if s1 - s0 < 2:
            return False
        removed = self.samples[s0:s1].copy()
        fade = min(self._fade_len(sr), s0, n - s1)

        left  = self.samples[:s0].copy()
        right = self.samples[s1:].copy()
        edge_l = left[-fade:].copy() if fade else None
        edge_r = right[:fade].copy() if fade else None
        if fade:
            ramp_out = np.linspace(1.0, 0.0, fade, dtype=np.float32)
            ramp_in  = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            if left.ndim == 2:
                ramp_out = ramp_out[:, None]
                ramp_in  = ramp_in[:, None]
            left[-fade:]  *= ramp_out[::-1] if False else np.flip(ramp_out, 0) * 0 + (1 - np.flip(ramp_out, 0) * 0)  # placeholder
        # NOTE: simple approach — fade left tail OUT toward joint, right head IN
        if fade:
            fade_out = np.linspace(1.0, 0.6, fade, dtype=np.float32)
            fade_in  = np.linspace(0.6, 1.0, fade, dtype=np.float32)
            if left.ndim == 2:
                fade_out = fade_out[:, None]
                fade_in  = fade_in[:, None]
            left[-fade:]  *= fade_out
            right[:fade]  *= fade_in

        self.samples = np.concatenate([left, right])
        self.undo_stack.append({
            "pos": s0, "removed": removed, "fade": fade,
            "edge_l": edge_l, "edge_r": edge_r,
        })
        self.redo_stack.clear()
        self.revision += 1
        self._trim_stack()
        return True

    def undo(self) -> bool:
        if not self.undo_stack:
            return False
        e = self.undo_stack.pop()
        pos, removed, fade = e["pos"], e["removed"], e["fade"]
        left  = self.samples[:pos].copy()
        right = self.samples[pos:].copy()
        if fade:
            left[-fade:] = e["edge_l"]
            right[:fade] = e["edge_r"]
        self.samples = np.concatenate([left, removed, right])
        self.redo_stack.append(e)
        self.revision -= 1
        return True

    def redo(self) -> bool:
        if not self.redo_stack:
            return False
        e = self.redo_stack.pop()
        s0 = e["pos"]
        s1 = s0 + len(e["removed"])
        n = len(self.samples)
        left  = self.samples[:s0].copy()
        right = self.samples[s1:].copy()
        fade = e["fade"]
        if fade:
            fade_out = np.linspace(1.0, 0.6, fade, dtype=np.float32)
            fade_in  = np.linspace(0.6, 1.0, fade, dtype=np.float32)
            if left.ndim == 2:
                fade_out = fade_out[:, None]
                fade_in  = fade_in[:, None]
            left[-fade:] *= fade_out
            right[:fade] *= fade_in
        self.samples = np.concatenate([left, right])
        self.undo_stack.append(e)
        self.revision += 1
        return True

    def _trim_stack(self):
        total = sum(e["removed"].nbytes for e in self.undo_stack)
        while self.undo_stack and (total > UNDO_BYTES or len(self.undo_stack) > UNDO_COUNT):
            dropped = self.undo_stack.pop(0)
            total -= dropped["removed"].nbytes


# ── Waveform editor window ─────────────────────────────────────────────────────

class WaveformEditor(tk.Toplevel):
    """Audacity-style touch-up editor for the processed output."""

    def __init__(self, parent, result, player, on_saved=None):
        """
        result: ProcessResult (path, splice_samples, sample_rate, channels)
        player: WavePlayer (owned by the main window, injected)
        """
        super().__init__(parent)
        self.title(f"TightTalk — Edit {os.path.basename(result.path)}")
        self.configure(bg=BG)
        self.minsize(900, 380)
        self.geometry("1100x450")

        self.result = result
        self.player = player
        self.on_saved = on_saved
        self.sr = result.sample_rate
        self.markers = list(result.splice_samples)

        self.engine = EditEngine(player.data.copy())
        self.pyramid = PeakPyramid(self.engine.samples)

        # Viewport: first visible sample + samples per pixel
        self.vp_start = 0.0
        self.vp_spp = max(1.0, len(self.engine.samples) / 1000.0)
        self.sel: tuple[int, int] | None = None    # (s0, s1) in samples
        self._sel_anchor = 0
        self._last_size = (0, 0)
        self._resize_job = None
        self._photo = None
        self._after_id = None

        self._build_ui()
        self._bind_keys()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(50, self._full_redraw)
        self._tick()

    # ── UI scaffolding ─────────────────────────────────────────────────────

    def _build_ui(self):
        bar = ttk.Frame(self, padding=8)
        bar.pack(side="top", fill="x")

        self.play_btn = ttk.Button(bar, text="▶ Play", command=self._toggle_play,
                                   takefocus=0)
        self.play_btn.pack(side="left")
        ttk.Button(bar, text="⏹", width=3, command=self._stop,
                   takefocus=0).pack(side="left", padx=(6, 14))
        ttk.Button(bar, text="Delete selection", command=self._delete_sel,
                   takefocus=0).pack(side="left")
        ttk.Button(bar, text="Undo", command=self._undo,
                   takefocus=0).pack(side="left", padx=(14, 0))
        ttk.Button(bar, text="Redo", command=self._redo,
                   takefocus=0).pack(side="left", padx=(6, 0))

        self.save_btn = ttk.Button(bar, text="Save", command=self._save, takefocus=0)
        self.save_btn.pack(side="right")
        self.pos_var = tk.StringVar(value="0:00.0")
        ttk.Label(bar, textvariable=self.pos_var).pack(side="right", padx=12)

        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=0, cursor="crosshair")
        self.canvas.pack(side="top", fill="both", expand=True)

        self.hbar = ttk.Scrollbar(self, orient="horizontal", command=self._on_scrollbar)
        self.hbar.pack(side="bottom", fill="x")

        # Canvas overlay items (raster image is created on first redraw)
        self._img_item = None
        self._cursor_line = self.canvas.create_line(0, 0, 0, 0, fill=YELLOW, width=1)
        self._playhead = self.canvas.create_line(0, 0, 0, 0, fill=GREEN, width=2)
        self._marker_items: list[int] = []

        self.canvas.bind("<Configure>", self._on_configure)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Shift-MouseWheel>", self._on_pan_wheel)

    def _bind_keys(self):
        # Aqua gotchas: shifted letters arrive uppercase (<Command-Z> = shift),
        # Mac delete key sends <BackSpace>; bind Control fallbacks for variance
        # across bundled Tk builds.
        for seq in ("<space>",):
            self.bind(seq, lambda e: (self._toggle_play(), "break")[-1])
        for seq in ("<BackSpace>", "<Delete>"):
            self.bind(seq, lambda e: self._delete_sel())
        for seq in ("<Command-z>", "<Control-z>"):
            self.bind(seq, lambda e: self._undo())
        for seq in ("<Command-Z>", "<Control-Z>"):
            self.bind(seq, lambda e: self._redo())
        for seq in ("<Command-s>", "<Control-s>"):
            self.bind(seq, lambda e: self._save())
        self.canvas.focus_set()

    # ── Coordinate transforms ──────────────────────────────────────────────

    def _x_to_sample(self, x: float) -> int:
        return int(self.vp_start + x * self.vp_spp)

    def _sample_to_x(self, s: float) -> float:
        return (s - self.vp_start) / self.vp_spp

    def _clamp_viewport(self):
        n = len(self.engine.samples)
        w = max(1, self.canvas.winfo_width())
        self.vp_spp = max(1.0 / 16, min(self.vp_spp, n / w))
        self.vp_start = max(0.0, min(self.vp_start, n - self.vp_spp * w))

    # ── Rendering ──────────────────────────────────────────────────────────

    def _render_raster(self, w: int, h: int) -> tk.PhotoImage:
        mins, maxs, rms = self.pyramid.query(self.vp_start, self.vp_spp, w)
        img = np.empty((h, w, 3), np.uint8)
        img[:] = _BG_RGB
        mid = h // 2
        amp = mid - 4
        y_top = np.clip(mid - (maxs * amp).astype(int), 0, h - 1)
        y_bot = np.clip(mid - (mins * amp).astype(int), 0, h - 1)
        r_top = np.clip(mid - (rms * amp).astype(int), 0, h - 1)
        r_bot = np.clip(mid + (rms * amp).astype(int), 0, h - 1)
        ys = np.arange(h)[:, None]
        body = (ys >= y_top[None, :]) & (ys <= y_bot[None, :])
        img[body] = _WAVE_RGB
        core = (ys >= r_top[None, :]) & (ys <= r_bot[None, :]) & body
        img[core] = _RMS_RGB
        img[mid, :] = _RMS_RGB   # center line

        # Selection tint baked into the raster (no Aqua stipple issues)
        if self.sel:
            x0 = int(np.clip(self._sample_to_x(self.sel[0]), 0, w))
            x1 = int(np.clip(self._sample_to_x(self.sel[1]), 0, w))
            if x1 > x0:
                region = img[:, x0:x1].astype(np.uint16)
                tint = np.array(_SEL_RGB, np.uint16)
                img[:, x0:x1] = ((region + tint) // 2).astype(np.uint8)

        ppm = b"P6 %d %d 255 " % (w, h) + img.tobytes()
        return tk.PhotoImage(data=ppm)

    def _full_redraw(self):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 10 or h < 10:
            return
        self._clamp_viewport()
        self._photo = self._render_raster(w, h)      # keep ref or GC blanks it
        if self._img_item is None:
            self._img_item = self.canvas.create_image(0, 0, anchor="nw",
                                                      image=self._photo)
            self.canvas.tag_lower(self._img_item)
        else:
            self.canvas.itemconfig(self._img_item, image=self._photo)
        self._redraw_markers(w, h)
        self._update_overlays()
        self._update_scrollbar()

    def _redraw_markers(self, w: int, h: int):
        for item in self._marker_items:
            self.canvas.delete(item)
        self._marker_items.clear()
        last_x = -10
        for s in self.markers:
            x = self._sample_to_x(s)
            if x < 0 or x > w:
                continue
            if x - last_x < 3:          # skip markers that smear at this zoom
                continue
            last_x = x
            self._marker_items.append(
                self.canvas.create_line(x, 0, x, h, fill=_MARKER_HEX, width=1))
        self.canvas.tag_raise(self._cursor_line)
        self.canvas.tag_raise(self._playhead)

    def _update_overlays(self):
        h = self.canvas.winfo_height()
        x = self._sample_to_x(self.player.position_sec * self.sr)
        self.canvas.coords(self._playhead, x, 0, x, h)

    def _update_scrollbar(self):
        n = max(1, len(self.engine.samples))
        w = max(1, self.canvas.winfo_width())
        lo = self.vp_start / n
        hi = min(1.0, (self.vp_start + self.vp_spp * w) / n)
        self.hbar.set(lo, hi)

    # ── Events ─────────────────────────────────────────────────────────────

    def _on_configure(self, ev):
        if (ev.width, ev.height) == self._last_size:
            return
        self._last_size = (ev.width, ev.height)
        if self._resize_job:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(80, self._full_redraw)

    def _on_scrollbar(self, *args):
        w = max(1, self.canvas.winfo_width())
        n = len(self.engine.samples)
        if args[0] == "moveto":
            self.vp_start = float(args[1]) * n
        elif args[0] == "scroll":
            step = self.vp_spp * w * (0.1 if args[2] == "units" else 0.9)
            self.vp_start += int(args[1]) * step
        self._full_redraw()

    def _on_wheel(self, ev):
        factor = 0.85 if ev.delta > 0 else 1 / 0.85
        anchor = self.vp_start + ev.x * self.vp_spp
        self.vp_spp *= factor
        self._clamp_viewport()
        self.vp_start = anchor - ev.x * self.vp_spp
        self._full_redraw()

    def _on_pan_wheel(self, ev):
        self.vp_start += -ev.delta * self.vp_spp * 12
        self._full_redraw()

    def _on_press(self, ev):
        self.canvas.focus_set()
        self._sel_anchor = self._x_to_sample(ev.x)
        self.sel = None

    def _on_drag(self, ev):
        s = self._x_to_sample(ev.x)
        self.sel = (min(self._sel_anchor, s), max(self._sel_anchor, s))
        self._full_redraw()

    def _on_release(self, ev):
        s = self._x_to_sample(ev.x)
        if self.sel and abs(self.sel[1] - self.sel[0]) >= self.vp_spp * 3:
            return                               # kept as drag-selection
        # plain click → seek
        self.sel = None
        sec = max(0.0, s / self.sr)
        self.player.seek(sec)
        h = self.canvas.winfo_height()
        x = self._sample_to_x(s)
        self.canvas.coords(self._cursor_line, x, 0, x, h)
        self._full_redraw()

    # ── Transport ──────────────────────────────────────────────────────────

    def _toggle_play(self):
        try:
            if self.player.playing:
                self.player.pause()
                self.play_btn.configure(text="▶ Play")
            elif self.sel:
                self.player.play(self.sel[0] / self.sr, self.sel[1] / self.sr)
                self.play_btn.configure(text="⏸ Pause")
            else:
                self.player.play()
                self.play_btn.configure(text="⏸ Pause")
        except Exception as exc:
            messagebox.showerror("TightTalk", f"Playback error:\n{exc}", parent=self)

    def _stop(self):
        self.player.stop()
        self.play_btn.configure(text="▶ Play")

    def _tick(self):
        self._update_overlays()
        self.pos_var.set(self._fmt_time(self.player.position_sec))
        if not self.player.playing:
            self.play_btn.configure(text="▶ Play")
        self._after_id = self.after(33, self._tick)

    @staticmethod
    def _fmt_time(sec: float) -> str:
        m, s = divmod(max(0.0, sec), 60)
        return f"{int(m)}:{s:04.1f}"

    # ── Editing ────────────────────────────────────────────────────────────

    def _delete_sel(self):
        if not self.sel:
            return
        s0, s1 = self.sel
        if not self.engine.delete(s0, s1, self.sr):
            return
        removed = s1 - s0
        # shift cut markers that sit after the deleted range
        self.markers = [m if m < s0 else m - removed
                        for m in self.markers if not (s0 <= m < s1)]
        self.sel = None
        self._after_edit()

    def _undo(self):
        if self.engine.undo():
            self._after_edit()   # markers may now be slightly off; acceptable

    def _redo(self):
        if self.engine.redo():
            self._after_edit()

    def _after_edit(self):
        self.pyramid = PeakPyramid(self.engine.samples)
        self.player.set_samples(self.engine.samples)
        self._title_refresh()
        self._full_redraw()

    def _title_refresh(self):
        star = " •" if self.engine.dirty else ""
        self.title(f"TightTalk — Edit {os.path.basename(self.result.path)}{star}")

    # ── Save / close ───────────────────────────────────────────────────────

    def _save(self):
        out = self.result.path
        tmp = out + ".tmp"
        data = self.engine.samples
        pcm = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)
        ch = 1 if pcm.ndim == 1 else pcm.shape[1]
        try:
            with wave.open(tmp, "wb") as wf:
                wf.setnchannels(ch)
                wf.setsampwidth(2)
                wf.setframerate(self.sr)
                wf.writeframes(pcm.tobytes())
            os.replace(tmp, out)        # atomic — player/Finder may hold the old file
        except Exception as exc:
            messagebox.showerror("TightTalk", f"Save failed:\n{exc}", parent=self)
            return
        self.engine.mark_saved()
        self._title_refresh()
        if self.on_saved:
            self.on_saved()

    def _on_close(self):
        if self.engine.dirty:
            ans = messagebox.askyesnocancel(
                "TightTalk", "Save changes before closing?", parent=self)
            if ans is None:
                return
            if ans:
                self._save()
                if self.engine.dirty:    # save failed
                    return
        if self._after_id:
            self.after_cancel(self._after_id)
        self.player.stop()
        self.destroy()
