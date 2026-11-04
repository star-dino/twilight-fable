"""Recorded-sample layers for the Twilight Fable soundtrack (`tools/sound_samples.py`).

`tools/sound.py` imports this module. It reads the cache written by `tools/samples.py fetch`
(CC0 recordings only) and provides three deterministic, numpy-only layers:

  * `render_beds`   -- per-scene ambience beds, equal-power crossfaded at scene cuts
  * `render_music`  -- a small note sampler (nearest sampled pitch, resample-shift, velocity layer,
                       release tail) driven by a short per-scene score (night -> dawn)
  * `foley`         -- round-robin recorded foley one-shots with small pitch/gain jitter

If the cache (or a whole group) is missing, every function returns None and the caller falls back to
the synthesized layer, so a fresh clone still renders.
"""

from __future__ import annotations

import json
import os
import re
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

SR = 48000
REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE = Path(os.environ.get("STYLEHOP_SAMPLES_CACHE") or (REPO_ROOT.parent / "out" / "samples-cache" / "twilight-fable")).resolve()

_NOTE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def note_to_midi(name: str) -> int:
    m = re.fullmatch(r"([A-G])(#?)(-?\d)", name)
    if not m:
        raise ValueError(name)
    return 12 * (int(m.group(3)) + 1) + _NOTE[m.group(1)] + (1 if m.group(2) else 0)


# --------------------------------------------------------------------------------------------------
# bank
# --------------------------------------------------------------------------------------------------


class Bank:
    """Reads `index.json` + cached wavs. `usable` is False when nothing has been fetched."""

    def __init__(self, cache: Path | None = None):
        self.cache = Path(cache) if cache else CACHE
        self.rows: dict[str, dict] = {}
        self.groups: dict[str, list[dict]] = {}
        self._audio: dict[str, np.ndarray] = {}
        self._last: dict[str, int] = {}
        idx = self.cache / "index.json"
        if idx.is_file():
            try:
                self.rows = json.loads(idx.read_text(encoding="utf-8")).get("samples", {})
            except Exception:
                self.rows = {}
        for rid in sorted(self.rows):
            row = self.rows[rid]
            if not (self.cache / row["file"]).is_file():
                continue
            for g in row.get("uses", []):
                self.groups.setdefault(g, []).append(row)

    @property
    def usable(self) -> bool:
        return bool(self.groups)

    def has(self, group: str) -> bool:
        return bool(self.groups.get(group))

    def audio(self, row: dict) -> np.ndarray:
        a = self._audio.get(row["id"])
        if a is None:
            _sr, x = wavfile.read(str(self.cache / row["file"]))
            a = x.astype(np.float64) / 32768.0
            if a.ndim == 1:
                a = np.stack([a, a], axis=1)
            self._audio[row["id"]] = a
        return a

    def pick(self, group: str, rng: np.random.Generator) -> dict | None:
        """Round-robin-ish: random but never the same variant twice in a row."""
        rows = self.groups.get(group)
        if not rows:
            return None
        i = int(rng.integers(0, len(rows)))
        if len(rows) > 1 and self._last.get(group) == i:
            i = (i + 1 + int(rng.integers(0, len(rows) - 1))) % len(rows)
        self._last[group] = i
        return rows[i]


# --------------------------------------------------------------------------------------------------
# dsp helpers
# --------------------------------------------------------------------------------------------------


def shift(x: np.ndarray, semitones: float) -> np.ndarray:
    """Pitch-shift by resampling (length changes with pitch, like a sampler)."""
    if abs(semitones) < 0.01:
        return x
    ratio = Fraction(2.0 ** (semitones / 12.0)).limit_denominator(240)
    return signal.resample_poly(x, ratio.denominator, ratio.numerator, axis=0)


def filt(x: np.ndarray, lp: float | None = None, hp: float | None = None) -> np.ndarray:
    if hp:
        x = signal.sosfilt(signal.butter(2, hp / (0.5 * SR), "high", output="sos"), x, axis=0)
    if lp:
        x = signal.sosfilt(signal.butter(2, lp / (0.5 * SR), "low", output="sos"), x, axis=0)
    return x


def fade(x: np.ndarray, fi: float, fo: float) -> np.ndarray:
    x = x.copy()
    n = len(x)
    a, b = min(n // 2, int(fi * SR)), min(n // 2, int(fo * SR))
    if a > 1:
        x[:a] *= (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, a)))[:, None]
    if b > 1:
        x[-b:] *= (0.5 + 0.5 * np.cos(np.linspace(0, np.pi, b)))[:, None]
    return x


def add(buf: np.ndarray, at: float, x: np.ndarray, gain: float = 1.0) -> None:
    i = int(round(at * SR))
    if i >= len(buf) or len(x) == 0:
        return
    if i < 0:
        x, i = x[-i:], 0
    n = min(len(buf) - i, len(x))
    if n > 0:
        buf[i : i + n] += x[:n] * gain


def pan_stereo(x: np.ndarray, pan: float) -> np.ndarray:
    """Collapse to mono and place with an equal-power law."""
    m = x.mean(axis=1) if x.ndim == 2 else x
    ang = (float(np.clip(pan, -1, 1)) + 1.0) * (np.pi / 4.0)
    return np.stack([m * np.cos(ang), m * np.sin(ang)], axis=1)


# --------------------------------------------------------------------------------------------------
# beds
# --------------------------------------------------------------------------------------------------

# scene index -> [(group, level in dB)] ; night -> dusk -> hearth -> dawn
BEDS = {
    0: [("bed.night", 0.0), ("bed.stream", -7.0)],
    1: [("bed.dusk", 0.0), ("bed.night", -12.0)],
    2: [("bed.hearth", 0.0), ("bed.room", -4.0)],
    3: [("bed.dawn", 0.0), ("bed.dusk", -10.0)],
}
BED_LEVEL_DB = -6.0  # beds are normalised to -24 dBFS RMS; this sets the bed bus under foley and music
XFADE = 0.7


def _bed_layer(bank: Bank, group: str, n: int) -> np.ndarray | None:
    rows = bank.groups.get(group)
    if not rows:
        return None
    out = np.zeros((n, 2))
    for row in rows:
        a = bank.audio(row)
        if len(a) < n:  # loop with a short equal-power seam
            seam = min(len(a) // 4, int(0.5 * SR))
            reps = [a]
            while sum(len(r) for r in reps) - seam * (len(reps) - 1) < n:
                prev = reps[-1]
                ramp = np.linspace(0, np.pi / 2, seam)[:, None]
                head = reps[0][:seam]
                reps[-1] = np.concatenate([prev[:-seam], prev[-seam:] * np.cos(ramp) + head * np.sin(ramp)])
                reps.append(reps[0][seam:])
            a = np.concatenate(reps)
        seg = filt(a[:n], row.get("lp"), row.get("hp"))
        out += seg * float(row.get("gain", 1.0))
    return out


def render_beds(bank: Bank, scenes: list, dur: float) -> np.ndarray | None:
    if not any(bank.has(g) for lst in BEDS.values() for g, _ in lst):
        return None
    n_total = int(round(dur * SR))
    out = np.zeros((n_total, 2))
    scenes = scenes or [{"index": 0, "start": 0.0, "end": dur}]
    for pos, sc in enumerate(scenes):
        idx = int(sc.get("index", pos)) % 4
        s0 = float(sc.get("start", 0.0))
        s1 = float(sc.get("end", dur))
        half = XFADE / 2.0
        a0 = max(0.0, s0 - half) if pos > 0 else 0.0
        a1 = min(dur, s1 + half) if pos < len(scenes) - 1 else dur
        n = int(round((a1 - a0) * SR))
        if n <= 0:
            continue
        env = np.ones(n)
        fi = int(XFADE * SR) if pos > 0 else int(0.04 * SR)
        fo = int(XFADE * SR) if pos < len(scenes) - 1 else int(0.12 * SR)
        fi, fo = min(fi, n // 2), min(fo, n // 2)
        env[:fi] = np.sin(np.linspace(0, np.pi / 2, fi))
        env[n - fo :] *= np.cos(np.linspace(0, np.pi / 2, fo))
        layer = np.zeros((n, 2))
        for group, db in BEDS[idx]:
            part = _bed_layer(bank, group, n)
            if part is not None:
                layer += part * 10 ** (db / 20.0)
        add(out, a0, layer * env[:, None], 10 ** (BED_LEVEL_DB / 20.0))
    return out


# --------------------------------------------------------------------------------------------------
# music: sampler + score
# --------------------------------------------------------------------------------------------------

# base gain, default hold (s), release (s), attack fade (s)
INSTRUMENTS = {
    "harp": (0.62, 2.2, 0.55, 0.002),
    "glock": (0.50, 1.4, 0.45, 0.002),
    "kalimba": (0.62, 1.6, 0.40, 0.002),
    "pizz": (0.85, 0.7, 0.20, 0.002),
    "flute": (0.42, 1.8, 0.45, 0.10),
}


def sample_note(bank: Bank, inst: str, midi: float, hold: float | None, vel: float, rng: np.random.Generator) -> np.ndarray | None:
    rows = bank.groups.get(f"inst.{inst}")
    if not rows:
        return None
    base, def_hold, rel, att = INSTRUMENTS[inst]
    best = min(rows, key=lambda r: (abs(note_to_midi(r["note"]) - midi), note_to_midi(r["note"]) > midi))
    src = note_to_midi(best["note"])
    x = bank.audio(best)
    x = shift(x, midi - src)
    hold = def_hold if hold is None else hold
    n = min(len(x), int((hold + rel) * SR))
    x = x[:n].copy()
    r = min(n // 2, int(rel * SR))
    if r > 1:
        x[-r:] *= (0.5 + 0.5 * np.cos(np.linspace(0, np.pi, r)))[:, None]
    x = fade(x, att, 0.0)
    vel = float(np.clip(vel, 0.05, 1.0))
    x = filt(x, lp=2400.0 + 5200.0 * vel)  # softer = darker, like a real touch
    return x * (base * vel**1.25 * (1.0 + 0.06 * (rng.random() - 0.5)))


# Per-scene motif. (beat, midi, velocity[, hold in beats]).  Night (A minor, unresolved question) ->
# door (F, wonder; flute enters as the door opens) -> hearth (C, warm roll as the lamp rises) ->
# dawn (C, the night's rising question is answered on the sunrise).
SCORE = {
    0: dict(bpm=58, pan=-0.1,
            pizz=[(0, 45, .62), (2, 52, .5)],
            harp=[(0.5, 57, .34), (1.0, 60, .36), (1.5, 64, .38)],
            glock=[(1.5, 76, .34), (2.25, 79, .30)]),
    1: dict(bpm=72, pan=0.0,
            pizz=[(0, 41, .62), (2, 48, .52), (4, 53, .45)],
            harp=[(0.5 + 0.5 * i, m, .34 + 0.03 * i) for i, m in enumerate([53, 57, 60, 65, 69, 72])],
            kalimba=[(2.0, 69, .46), (2.5, 72, .42), (3.5, 67, .38), (4.25, 65, .34)],
            flute=[(2.2, 69, .40, 2.6)]),
    2: dict(bpm=66, pan=0.1,
            pizz=[(0, 48, .60), (2, 43, .52), (3.5, 48, .44)],
            harp=[(0.55 + 0.12 * i, m, .40) for i, m in enumerate([48, 55, 60, 64, 67])] + [(2.0, 64, .36), (2.5, 67, .36), (3.0, 72, .34)],
            flute=[(0.9, 64, .40, 1.6), (2.6, 67, .38, 1.3)],
            glock=[(3.2, 84, .26)]),
    3: dict(bpm=76, pan=0.0,
            pizz=[(0, 48, .56), (2, 55, .46)],
            harp=[(0.4 * i, m, .42) for i, m in enumerate([60, 64, 67, 72, 76, 79])],
            flute=[(0.3, 67, .38, 3.4)],
            glock=[(1.3, 76, .38), (2.0, 79, .38), (2.9, 84, .46)]),
}


def render_music(bank: Bank, scenes: list, dur: float, rng: np.random.Generator) -> np.ndarray | None:
    if not any(bank.has(f"inst.{i}") for i in INSTRUMENTS):
        return None
    out = np.zeros((int(round(dur * SR)), 2))
    scenes = scenes or [{"index": 0, "start": 0.0, "end": dur}]
    for pos, sc in enumerate(scenes):
        spec = SCORE[int(sc.get("index", pos)) % 4]
        s0, s1 = float(sc.get("start", 0.0)), float(sc.get("end", dur))
        beat = 60.0 / spec["bpm"]
        for inst in INSTRUMENTS:
            for ev in spec.get(inst, []):
                b, midi, vel = ev[0], ev[1], ev[2]
                t = s0 + b * beat
                if t >= s1 - 0.05 or t >= dur - 0.05:
                    continue
                hold = ev[3] * beat if len(ev) > 3 else None
                x = sample_note(bank, inst, midi, hold, vel, rng)
                if x is None:
                    continue
                spread = {"harp": 0.0, "pizz": -0.15, "glock": 0.2, "kalimba": 0.15, "flute": -0.05}[inst]
                add(out, t, pan_stereo(x, spec["pan"] + spread + 0.08 * (rng.random() - 0.5)))
    return out


# --------------------------------------------------------------------------------------------------
# foley
# --------------------------------------------------------------------------------------------------

# action -> [(group, gain, pitch ratio)]  (all layers play together)
FOLEY = {
    "paw_step": [("foley.paw_step", 0.38, 1.0)],
    "foot_step": [("foley.paw_step", 0.38, 1.0)],
    "critter_hop": [("foley.paw_step", 0.34, 1.15), ("foley.rustle", 0.18, 1.0)],
    "ear_perk": [("foley.rustle", 0.30, 1.25)],
    "water_ripple": [("foley.water_ripple", 0.42, 1.0)],
    "wave_surge": [("foley.water_ripple", 0.55, 0.85)],
    "drink_sip": [("foley.water_ripple", 0.40, 1.1)],
    "spirit_shimmer": [("foley.spirit_shimmer", 0.38, 1.0)],
    "wood_creak": [("foley.wood_creak", 0.55, 1.0)],
    "layer_float": [("foley.spirit_shimmer", 0.20, 0.9)],
    "stroke_draw": [("foley.spirit_shimmer", 0.34, 1.0)],
    "yawn": [("foley.rustle", 0.20, 0.85)],
}


def foley(bank: Bank, action: str, intensity: float, pitch: float, rng: np.random.Generator) -> np.ndarray | None:
    """Recorded foley for one cue as a mono-ish (n, 2) array, or None when the action has no recorded group."""
    layers = FOLEY.get(action)
    if not layers or not all(bank.has(g) for g, _, _ in layers):
        return None
    parts = []
    for group, gain, ratio in layers:
        row = bank.pick(group, rng)
        x = bank.audio(row)
        semis = 12.0 * np.log2(max(0.4, ratio * pitch)) + rng.uniform(-0.45, 0.45)
        x = shift(x, semis)
        x = filt(x, row.get("lp"), row.get("hp"))
        x = fade(x, 0.003, min(0.08, 0.3 * len(x) / SR))
        parts.append(x * gain * float(row.get("gain", 1.0)) * 10 ** (rng.uniform(-1.5, 1.5) / 20.0))
    n = max(len(p) for p in parts)
    out = np.zeros((n, 2))
    for p in parts:
        out[: len(p)] += p
    return out * float(np.clip(intensity, 0.0, 1.2))
