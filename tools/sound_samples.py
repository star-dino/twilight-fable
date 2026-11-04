"""Recorded-sample layers for the Twilight Fable soundtrack (`tools/sound_samples.py`).

`tools/sound.py` imports this module. It reads the cache written by `tools/samples.py fetch`
(CC0 recordings only) and provides three deterministic, numpy-only layers:

  * `render_beds`   -- per-scene ambience beds, equal-power crossfaded at scene cuts
  * `render_music`  -- a multi-instrument note sampler (pianissimo upright piano, concert harp,
                       soft-mallet vibraphone, bowed/pizzicato cello, non-vibrato flute, soft glock)
                       driven by cohesive per-scene harmonic motifs
  * `foley`         -- clean studio & field CC0 foley one-shots with gentle musical halos

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
        x = signal.sosfilt(signal.butter(2, np.clip(hp / (0.5 * SR), 0.001, 0.95), "high", output="sos"), x, axis=0)
    if lp:
        x = signal.sosfilt(signal.butter(2, np.clip(lp / (0.5 * SR), 0.002, 0.96), "low", output="sos"), x, axis=0)
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
    """Place stereo/mono signal with an equal-power law while preserving subtle stereo air."""
    ang = (float(np.clip(pan, -1, 1)) + 1.0) * (np.pi / 4.0)
    if x.ndim == 2:
        mid = 0.5 * (x[:, 0] + x[:, 1])
        side = 0.35 * (x[:, 0] - x[:, 1])
        return np.stack([mid * np.cos(ang) + side, mid * np.sin(ang) - side], axis=1)
    return np.stack([x * np.cos(ang), x * np.sin(ang)], axis=1)


# --------------------------------------------------------------------------------------------------
# beds
# --------------------------------------------------------------------------------------------------

# scene index -> [(group, level in dB)] ; night -> dusk -> hearth -> dawn
BEDS = {
    0: [("bed.night", 0.0), ("bed.stream", -8.0)],
    1: [("bed.dusk", 0.0), ("bed.night", -10.0)],
    2: [("bed.hearth", 0.0), ("bed.room", -4.0)],
    3: [("bed.dawn", 0.0), ("bed.dusk", -10.0)],
}
BED_LEVEL_DB = -9.5   # beds are normalised to -24 dBFS RMS; stays gently audible beneath character & score until a reveal
XFADE = 0.85
BED_STACK_DB = -7.0  # a declared bed list without explicit dB: first group at 0, each further group this much lower


def parse_beds(spec) -> list | None:
    """`data-bed="lake,frogs:-8"` -> [("bed.lake", 0.0), ("bed.frogs", -8.0)]; None when undeclared/empty."""
    if not spec or not isinstance(spec, str):
        return None
    out = []
    for tok in (t.strip() for t in spec.split(",")):
        if not tok:
            continue
        name, _, db = tok.partition(":")
        name = name.strip()
        if not name.startswith("bed."):
            name = "bed." + name
        try:
            level = float(db) if db.strip() else (0.0 if not out else BED_STACK_DB * len(out))
        except ValueError:
            level = 0.0
        out.append((name, level))
    return out or None


def _bed_layer(bank: Bank, group: str, n: int) -> np.ndarray | None:
    rows = bank.groups.get(group)
    if not rows:
        return None
    out = np.zeros((n, 2))
    for row in rows:
        a = bank.audio(row)
        if len(a) < n:  # loop with a smooth equal-power seam
            seam = min(len(a) // 4, int(0.8 * SR))
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
    return out / max(1.0, np.sqrt(len(rows)))


def render_beds(bank: Bank, scenes: list, dur: float, level_db: float | None = None) -> np.ndarray | None:
    level_db = BED_LEVEL_DB if level_db is None else level_db
    declared = [parse_beds(sc.get("bed")) for sc in (scenes or [])]
    wanted = [g for lst in BEDS.values() for g, _ in lst] + [g for d in declared if d for g, _ in d]
    if not any(bank.has(g) for g in wanted):
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
        fi = int(XFADE * SR) if pos > 0 else int(0.18 * SR)
        fo = int(XFADE * SR) if pos < len(scenes) - 1 else int(0.45 * SR)
        fi, fo = min(fi, n // 2), min(fo, n // 2)
        env[:fi] = np.sin(np.linspace(0, np.pi / 2, fi))
        env[n - fo :] *= np.cos(np.linspace(0, np.pi / 2, fo))
        layer = np.zeros((n, 2))
        for group, db in (declared[pos] or BEDS[idx]):
            part = _bed_layer(bank, group, n)
            if part is not None:
                layer += part * 10 ** (db / 20.0)
        add(out, a0, layer * env[:, None], 10 ** (level_db / 20.0))
    return out


# --------------------------------------------------------------------------------------------------
# music: sampler + score
# --------------------------------------------------------------------------------------------------

# (base gain, default hold s, release s, attack fade s, base_lp Hz, vel_lp Hz)
INSTRUMENTS = {
    "piano": (0.50, 2.3, 0.90, 0.008, 1700.0, 2400.0),
    "harp":  (0.44, 2.4, 0.85, 0.005, 2000.0, 2800.0),
    "vibes": (0.38, 2.1, 0.85, 0.006, 1850.0, 2500.0),
    "cello": (0.34, 3.0, 1.15, 0.380,  720.0,  720.0),
    "pizz":  (0.38, 0.75, 0.32, 0.004, 980.0, 1300.0),
    "flute": (0.30, 2.0, 0.80, 0.220, 1850.0, 2000.0),
    "glock": (0.20, 1.4, 0.60, 0.005, 2100.0, 2200.0),
}

INST_SPREAD = {
    "piano": -0.06,
    "harp":   0.08,
    "vibes":  0.14,
    "cello": -0.18,
    "pizz":  -0.14,
    "flute":  0.04,
    "glock":  0.18,
}


def sample_note(bank: Bank, inst: str, midi: float, hold: float | None, vel: float, rng: np.random.Generator) -> np.ndarray | None:
    rows = bank.groups.get(f"inst.{inst}")
    if not rows:
        return None
    base, def_hold, rel, att, lp0, lp_vel = INSTRUMENTS[inst]
    best = min(rows, key=lambda r: (abs(note_to_midi(r["note"]) - midi), note_to_midi(r["note"]) > midi))
    src = note_to_midi(best["note"])
    x = bank.audio(best)
    x = shift(x, midi - src)
    hold = def_hold if hold is None else hold
    n = min(len(x), int((hold + rel) * SR))
    if n < 32:
        return None
    x = x[:n].copy()
    r = min(max(2, int(0.65 * n)), int(rel * SR))
    if r > 1:
        x[-r:] *= (0.5 + 0.5 * np.cos(np.linspace(0, np.pi, r)))[:, None]
    x = fade(x, att, 0.0)
    vel = float(np.clip(vel, 0.05, 1.0))
    hp = 45.0 if inst in ("cello", "pizz", "piano", "harp") else 180.0
    x = filt(x, lp=lp0 + lp_vel * vel, hp=hp)
    return x * (base * (vel ** 1.18) * (1.0 + 0.04 * (rng.random() - 0.5)))


# Default 4-scene arc ("The Warm Window" & undeclared scenes): Am9 -> Fmaj7 -> Cmaj9 -> Cmaj9 (night -> dawn)
SCORE = {
    0: dict(
        bpm=58, pan=-0.05,
        cello=[(0.05, 45, 0.48, 2.6)],
        pizz=[(0.0, 45, 0.46), (2.0, 52, 0.38)],
        piano=[(0.0, 57, 0.44), (1.0, 60, 0.42), (2.0, 64, 0.44)],
        harp=[(0.5, 60, 0.38), (1.5, 64, 0.40), (2.4, 67, 0.38)],
        vibes=[(1.5, 71, 0.36)],
    ),
    1: dict(
        bpm=68, pan=0.0,
        cello=[(0.05, 41, 0.50, 3.2)],
        pizz=[(0.0, 41, 0.46), (2.5, 48, 0.40)],
        piano=[(0.0, 53, 0.44), (1.2, 57, 0.42), (2.4, 60, 0.46), (3.6, 64, 0.44)],
        harp=[(0.6, 57, 0.38), (1.8, 60, 0.40), (2.9, 65, 0.42), (3.8, 69, 0.40)],
        flute=[(2.1, 69, 0.44, 2.0)],
    ),
    2: dict(
        bpm=64, pan=0.05,
        cello=[(0.05, 48, 0.52, 2.8)],
        piano=[(0.0, 48, 0.46), (1.0, 55, 0.44), (2.0, 60, 0.46), (2.9, 64, 0.44)],
        harp=[(0.4, 55, 0.40), (1.4, 60, 0.42), (2.3, 64, 0.42), (3.1, 67, 0.40)],
        flute=[(0.8, 64, 0.44, 1.8)],
        vibes=[(1.7, 72, 0.38)],
    ),
    3: dict(
        bpm=64, pan=0.0,
        cello=[(0.05, 48, 0.50, 2.6)],
        harp=[(0.14 * i, m, 0.40) for i, m in enumerate([48, 55, 60, 64, 67, 72])],
        flute=[(0.35, 67, 0.44, 2.0)],
        piano=[(1.2, 64, 0.42), (2.0, 60, 0.44)],
        vibes=[(1.9, 72, 0.40)],
    ),
}

MOODS = {"night": SCORE[0], "door": SCORE[1], "hearth": SCORE[2], "dawn": SCORE[3]}

# Cohesive nocturnal lake suite (Dadd9 -> Bm9/Gmaj9 -> Gadd9/Asus4 -> Dmaj9):
# warm low cello foundation + intimate pianissimo piano & harp + round soft-mallet vibraphone & gentle flute.
MOODS.update({
    "lake-night": dict(
        bpm=60, pan=-0.06,
        cello=[(0.05, 38, 0.50, 3.3)],
        pizz=[(0.0, 38, 0.44), (2.5, 45, 0.38)],
        piano=[(0.0, 50, 0.46), (1.1, 57, 0.42), (2.1, 64, 0.44), (3.1, 66, 0.46)],
        harp=[(0.55, 57, 0.38), (1.6, 62, 0.40), (2.6, 66, 0.42), (3.65, 69, 0.40)],
        vibes=[(1.85, 74, 0.38), (3.25, 69, 0.42)],
    ),
    "mirror": dict(
        bpm=60, pan=0.0,
        cello=[(0.05, 47, 0.48, 2.0), (2.35, 43, 0.52, 2.1)],
        piano=[(0.0, 54, 0.42), (1.15, 59, 0.42), (2.35, 55, 0.46), (3.4, 62, 0.45)],
        harp=[(0.55, 57, 0.38), (1.65, 62, 0.40), (2.65, 66, 0.44), (3.20, 69, 0.46), (3.85, 74, 0.42)],
        flute=[(0.75, 66, 0.42, 1.5), (2.70, 69, 0.46, 1.5)],
        vibes=[(2.65, 74, 0.40), (3.20, 78, 0.44)],
    ),
    "play": dict(
        bpm=76, pan=0.05,
        cello=[(0.08, 43, 0.46, 2.2), (2.9, 45, 0.48, 2.1)],
        pizz=[(0.0, 43, 0.46), (1.5, 50, 0.40), (2.9, 45, 0.44), (4.4, 52, 0.38)],
        harp=[(0.0, 55, 0.42), (0.65, 59, 0.40), (1.3, 62, 0.44), (2.6, 66, 0.48), (3.4, 69, 0.46), (4.5, 64, 0.42)],
        piano=[(0.65, 62, 0.42), (2.6, 67, 0.46), (3.9, 69, 0.44), (4.9, 73, 0.42)],
        vibes=[(1.85, 71, 0.38), (3.6, 74, 0.42), (4.5, 76, 0.42)],
    ),
    "stars": dict(
        bpm=56, pan=0.0,
        cello=[(0.05, 38, 0.54, 3.5), (0.20, 45, 0.40, 3.3)],
        harp=[(0.15 * i, m, 0.40 + 0.015 * i) for i, m in enumerate([50, 57, 62, 66, 69, 73, 74])] + [(2.25, 69, 0.38), (3.05, 66, 0.36)],
        flute=[(0.45, 69, 0.45, 2.2)],
        piano=[(1.35, 66, 0.44), (2.25, 64, 0.42), (3.05, 62, 0.46)],
        vibes=[(1.30, 74, 0.42), (3.05, 69, 0.40)],
    ),
})

MUSIC_VEL = 0.88


def score_for(sc: dict, pos: int) -> dict:
    """`data-mood` -> a named motif (MOODS, or night/door/hearth/dawn); otherwise the scene index default."""
    mood = (sc.get("mood") or "").strip().lower()
    if mood in MOODS:
        return MOODS[mood]
    return SCORE[int(sc.get("index", pos)) % 4]


def render_music(bank: Bank, scenes: list, dur: float, rng: np.random.Generator) -> np.ndarray | None:
    if not any(bank.has(f"inst.{i}") for i in INSTRUMENTS):
        return None
    n_total = int(round(dur * SR))
    out = np.zeros((n_total, 2))
    scenes = scenes or [{"index": 0, "start": 0.0, "end": dur}]
    for pos, sc in enumerate(scenes):
        spec = score_for(sc, pos)
        s0, s1 = float(sc.get("start", 0.0)), float(sc.get("end", dur))
        beat = 60.0 / spec["bpm"]
        for inst in INSTRUMENTS:
            for ev in spec.get(inst, []):
                b, midi, vel = ev[0], ev[1], ev[2] * MUSIC_VEL
                t = s0 + b * beat
                if t >= s1 - 0.06 or t >= dur - 0.12:
                    continue
                hold = ev[3] * beat if len(ev) > 3 else None
                x = sample_note(bank, inst, midi, hold, vel, rng)
                if x is None:
                    continue
                spread = INST_SPREAD.get(inst, 0.0)
                add(out, t, pan_stereo(x, spec["pan"] + spread + 0.05 * (rng.random() - 0.5)))
    return fade(out, 0.03, 0.55)


# --------------------------------------------------------------------------------------------------
# mix motion: reveal swell + foley sidechain (control-rate dB curves)
# --------------------------------------------------------------------------------------------------

REVEAL_DB = 8.0        # default swell above the low base
REVEAL_LEAD = 0.3      # swell starts this much before the marked moment
REVEAL_ATTACK = 0.9
REVEAL_HOLD = 1.3
REVEAL_RELEASE = 2.0
MUSIC_REVEAL_DUCK_DB = -1.5
FOLEY_DUCK_DB = 1.5    # subtle dip under foley transients without choppy pumping
FOLEY_DUCK = (0.04, 0.08, 0.38)  # attack, hold, release (s)
CTRL_RATE = 200


def reveal_events(scenes: list, cues: list) -> list:
    """[(absolute t, swell dB)] from scene `reveal="t_local[:dB]"` and cues of kind `reveal` (dB = 10 * intensity)."""
    ev = []
    for sc in scenes or []:
        raw = sc.get("reveal")
        if not raw:
            continue
        parts = str(raw).split(":")
        try:
            t = float(sc.get("start", 0.0)) + float(parts[0])
            db = float(parts[1]) if len(parts) > 1 and parts[1].strip() else REVEAL_DB
        except ValueError:
            continue
        ev.append((t, min(db, 10.0)))
    for c in cues or []:
        if str(c.get("kind")) == "reveal":
            db = float(c["gain_db"]) if c.get("gain_db") is not None else 10.0 * float(c.get("intensity", 0.75))
            ev.append((float(c.get("t", 0.0)), min(db, 10.0)))
    return sorted(ev)


def _env(t: np.ndarray, t0: float, att: float, hold: float, rel: float) -> np.ndarray:
    """0 -> 1 (raised cosine over att) -> hold -> 0 (raised cosine over rel)."""
    up = np.clip((t - t0) / max(att, 1e-6), 0.0, 1.0)
    dn = np.clip((t - (t0 + att + hold)) / max(rel, 1e-6), 0.0, 1.0)
    return (0.5 - 0.5 * np.cos(np.pi * up)) * (0.5 + 0.5 * np.cos(np.pi * dn))


def mix_motion(scenes: list, cues: list, dur: float, hits: list) -> dict:
    """Control-rate curves: bed dB (reveal swell + foley duck), music dB (gentle duck during the swell), and pieces."""
    n = int(dur * CTRL_RATE) + 2
    t = np.arange(n) / CTRL_RATE
    swell = np.zeros(n)
    amount = np.zeros(n)
    for T, db in reveal_events(scenes, cues):
        e = _env(t, T - REVEAL_LEAD, REVEAL_ATTACK, REVEAL_HOLD, REVEAL_RELEASE)
        swell = np.maximum(swell, e * db)
        amount = np.maximum(amount, e)
    duck = np.zeros(n)
    for ht, inten in hits:
        depth = FOLEY_DUCK_DB * float(np.clip(inten, 0.0, 1.0))
        a, h, r = FOLEY_DUCK
        duck = np.minimum(duck, -depth * _env(t, ht - a, a, h, r))
    return {"t": t, "swell_db": swell, "duck_db": duck, "bed_db": swell + duck, "music_db": MUSIC_REVEAL_DUCK_DB * amount}


def apply_db(x: np.ndarray, t_ctrl: np.ndarray, db: np.ndarray) -> np.ndarray:
    g = 10.0 ** (np.interp(np.arange(len(x)) / SR, t_ctrl, db) / 20.0)
    return x * g[:, None]


# --------------------------------------------------------------------------------------------------
# foley
# --------------------------------------------------------------------------------------------------

# action -> [(group, gain, pitch ratio, max_dur_s, fade_out_s[, delay_s])]
FOLEY = {
    "paw_step":       [("foley.paw_step", 0.28, 1.00, 0.14, 0.05)],
    "foot_step":      [("foley.paw_step", 0.28, 1.00, 0.14, 0.05)],
    "critter_hop":    [("foley.paw_step", 0.30, 1.18, 0.14, 0.05)],
    "water_ripple":   [("foley.water_ripple", 0.38, 1.00, 0.28, 0.08)],
    "wave_surge":     [("foley.water_ripple", 0.44, 0.86, 0.34, 0.10)],
    "drink_sip":      [("foley.water_ripple", 0.32, 1.12, 0.26, 0.08)],
    "spirit_shimmer": [("foley.spirit_shimmer", 0.22, 1.00, 1.10, 0.38)],
    "layer_float":    [("foley.spirit_shimmer", 0.15, 1.00, 0.95, 0.35)],
    "stroke_draw":    [("foley.spirit_shimmer", 0.18, 1.00, 0.90, 0.32)],
    "wood_creak":     [("foley.wood_latch", 0.24, 1.00, 0.14, 0.05, 0.0), ("foley.wood_creak", 0.28, 0.92, 0.40, 0.16, 0.045)],
}

# Pentatonic {D, E, G, A, B} studio-instrument grace notes for expressive character foley
# (consonant across both the D-major/B-minor lake suite and the A-minor/C-major woodland suite)
FOLEY_GRACE = {
    "ear_perk":       [("harp", 69, 0.28, 0.42, 0.0), ("vibes", 81, 0.18, 0.38, 0.03)],
    "critter_hop":    [("harp", 62, 0.26, 0.36, 0.0), ("harp", 69, 0.28, 0.42, 0.06)],
    "water_ripple":   [("vibes", 74, 0.15, 0.55, 0.03)],
    "spirit_shimmer": [("vibes", 81, 0.20, 0.85, 0.0), ("harp", 74, 0.20, 0.90, 0.07)],
    "wood_creak":     [("harp", 57, 0.22, 0.65, 0.05)],
    "yawn":           [("flute", 64, 0.24, 0.55, 0.0), ("harp", 57, 0.20, 0.65, 0.10)],
}


def foley(bank: Bank, action: str, intensity: float, pitch: float, rng: np.random.Generator) -> np.ndarray | None:
    """Recorded foley for one cue as a stereo (n, 2) array, or None when the action has no recorded group."""
    layers = FOLEY.get(action, [])
    graces = FOLEY_GRACE.get(action, [])
    if not layers and not graces:
        return None
    if layers and not all(bank.has(spec[0]) for spec in layers):
        return None

    parts: list[tuple[float, np.ndarray]] = []
    for spec in layers:
        group, gain, ratio, max_dur, fo_s = spec[:5]
        delay_s = spec[5] if len(spec) > 5 else 0.0
        row = bank.pick(group, rng)
        if row is None:
            continue
        x = bank.audio(row)
        jitter = 0.10 if group == "foley.spirit_shimmer" else 0.30
        semis = 12.0 * np.log2(max(0.4, ratio * pitch)) + rng.uniform(-jitter, jitter)
        x = shift(x, semis)
        n_lim = min(len(x), int(round(max_dur * SR)))
        x = x[:n_lim].copy()
        x = filt(x, row.get("lp"), row.get("hp"))
        x = fade(x, 0.004, min(fo_s, 0.45 * len(x) / SR))
        parts.append((delay_s, x * gain * float(row.get("gain", 1.0)) * (10 ** (rng.uniform(-0.8, 0.8) / 20.0))))

    semi_off = int(round(12.0 * np.log2(max(0.5, min(1.6, pitch)))))
    choice = int(rng.choice([0, 2, 5])) if action == "ear_perk" else (int(rng.choice([0, 2, 7])) if action == "water_ripple" else 0)
    for inst, midi, vel, hold, delay_s in graces:
        note = sample_note(bank, inst, midi + semi_off + choice, hold, vel, rng)
        if note is not None:
            parts.append((delay_s, note * 0.50))

    if not parts:
        return None
    n = max(int(round(d * SR)) + len(p) for d, p in parts)
    out = np.zeros((n, 2))
    for d, p in parts:
        add(out, d, p)
    return out * float(np.clip(intensity, 0.0, 1.15))
