#!/usr/bin/env python3
"""
Twilight Fable cue-driven procedural sound & score synthesizer (`tools/sound.py`).

Pure Python (NumPy + SciPy + stdlib `wave`) synthesizer that turns a deterministic
DOM & runtime motion cue sheet (`window.CUES` -> `cues.json`) into a mastered 48kHz 16-bit
stereo WAV soundtrack for the `twilight` preset: Celesta bell-tine modal synth, warm
felt-piano chords and cello, a nocturnal-forest hall shimmer, soft noise-free foley
(paw step, ear perk, critter hop, water ripple, spirit shimmer, wood creak) and a
warm high-roll-off master bus.

Usage:
  python3 tools/sound.py --cues cues.json --out <work-dir>/mix.wav [--preset twilight]
  python3 tools/sound.py --cues cues.json --voice voice.wav --out mix.wav [--duck-db -10] [--voice-gain 0]
"""

import argparse
import json
from pathlib import Path
import sys
import wave
import zlib
import numpy as np
from scipy import signal

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))
try:
    from audio_grains import ensure_audio_grains
except Exception:
    ensure_audio_grains = None

SR = 48000

PRESETS = {
    "twilight": {
        "chords": [
            [261.63, 329.63, 392.00, 523.25, 587.33],  # Scene 1: Cadd9 (C4 - E4 - G4 - C5 - D5, pure open consonant)
            [220.00, 261.63, 329.63, 392.00, 523.25],  # Scene 2: Am7 (A3 - C4 - E4 - G4 - C5)
            [174.61, 261.63, 349.23, 440.00, 523.25],  # Scene 3: Fadd9 (F3 - C4 - F4 - A4 - C5)
            [261.63, 329.63, 392.00, 523.25, 659.25],  # Scene 4: Cmaj (C4 - E4 - G4 - C5 - E5, warm home resolution)
        ],
        "pad_gain": 0.11,
        "chime_freqs": [392.00, 523.25, 587.33, 659.25],
        "tap_freq": 261.63,
        "tick_freq": 523.25,
    },
}

PRESET_ALIASES = {
    "twilight-fable": "twilight",
}


def resolve_preset(name: str) -> str:
    key = PRESET_ALIASES.get(name, name) if name else "twilight"
    if key not in PRESETS:
        print(f"[sound] unknown preset {name!r}; this repo only ships 'twilight'", file=sys.stderr)
        return "twilight"
    return key


# =============================================================================
# 0. HYBRID GRAIN BANK LOADER & CORE PHYSICAL MODELING TOOLKIT
# =============================================================================

_GRAIN_CACHE = {}


def load_grain(name: str) -> np.ndarray:
    """Loads a 48kHz WAV grain from `assets/audio/<name>.wav` (auto-building if missing)."""
    key = name if name.endswith(".wav") else f"{name}.wav"
    if key in _GRAIN_CACHE:
        return _GRAIN_CACHE[key]
    audio_dir = REPO_ROOT / "assets" / "audio"
    fpath = audio_dir / key
    if not fpath.exists() and ensure_audio_grains is not None:
        ensure_audio_grains(audio_dir)
    if not fpath.exists():
        return np.zeros(int(0.2 * SR), dtype=np.float64)
    with wave.open(str(fpath), "rb") as wf:
        nch = wf.getnchannels()
        nframes = wf.getnframes()
        raw = np.frombuffer(wf.readframes(nframes), dtype=np.int16).astype(np.float64) / 32768.0
        if nch > 1:
            raw = raw.reshape(-1, nch)
    _GRAIN_CACHE[key] = raw
    return raw


def slice_grain(
    name: str,
    dur: float,
    pitch=1.0,
    rng: np.random.Generator = None,
    random_offset: bool = True,
    gain: float = 1.0,
) -> np.ndarray:
    """
    Slices a mono segment from a pre-baked acoustic grain in `assets/audio/`, applies
    fractional pitch-shifting (so every event has unique formant/pitch variation), and
    applies smooth raised-cosine anti-click fades.
    Accepts `dur` either in seconds (float <= 60) or in sample count (int > 60), and
    allows the 3rd positional arg to be either `pitch` (float) or `rng` (Generator).
    """
    if isinstance(pitch, np.random.Generator):
        rng = pitch
        pitch = 1.0
    if rng is None:
        rng = np.random.default_rng(101)
    raw = load_grain(name)
    if raw.ndim == 2:
        raw = raw.mean(axis=1)
    if isinstance(dur, (int, np.integer)) and dur > 60:
        n_out = max(16, int(dur))
    else:
        n_out = max(16, int(round(float(dur) * SR)))
    pitch_eff = float(np.clip(float(pitch) * rng.uniform(0.95, 1.05), 0.55, 1.95))
    n_src = max(16, int(round(n_out * pitch_eff)))

    if len(raw) <= n_src:
        reps = int(np.ceil(n_src / max(1, len(raw)))) + 1
        src = np.tile(raw, reps)[:n_src]
    else:
        max_off = len(raw) - n_src
        off = int(rng.integers(0, max_off + 1)) if random_offset and max_off > 0 else 0
        src = raw[off : off + n_src]

    # Fast linear interpolation resampling for pitch shift
    x_old = np.linspace(0.0, 1.0, len(src))
    x_new = np.linspace(0.0, 1.0, n_out)
    out = np.interp(x_new, x_old, src)
    return fade_in_out(out, 0.002, 0.015) * float(gain)


def fade_in_out(sig: np.ndarray, fade_in_s: float = 0.002, fade_out_s: float = 0.015) -> np.ndarray:
    """Smooth raised-cosine fade-in and fade-out so event boundaries never click."""
    n = len(sig)
    if n < 4:
        return sig
    out = sig.copy()
    fi = min(int(fade_in_s * SR), n // 3)
    fo = min(int(fade_out_s * SR), n // 3)
    if fi > 1:
        w_in = 0.5 * (1.0 - np.cos(np.pi * np.arange(fi) / fi))
        out[:fi] *= w_in[:, None] if out.ndim == 2 else w_in
    if fo > 1:
        w_out = 0.5 * (1.0 + np.cos(np.pi * np.arange(fo) / fo))
        out[-fo:] *= w_out[:, None] if out.ndim == 2 else w_out
    return out


def generate_pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """True 1/f pink noise via FFT spectral shaping (never harsh white noise)."""
    if n < 8:
        return np.zeros(max(1, n), dtype=np.float64)
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0 / SR)
    scale = 1.0 / np.sqrt(np.maximum(freqs, 20.0))
    scale[0] = 0.0
    pink = np.fft.irfft(spec * scale, n=n)
    return pink / (np.max(np.abs(pink)) + 1e-12)


def time_varying_bandpass(
    sig: np.ndarray,
    f_curve,
    q: float = 2.6,
    btype: str = "bandpass",
    block: int = 128,
) -> np.ndarray:
    """Zipper-free block-wise biquad filter sweep carrying exact state `zi` across blocks."""
    n = len(sig)
    if np.ndim(f_curve) == 0:
        f_arr = np.full(n, float(f_curve), dtype=np.float64)
    else:
        f_arr = np.asarray(f_curve, dtype=np.float64)
    out = np.zeros(n, dtype=np.float64)
    nyq = 0.5 * SR
    zi = None
    for i in range(0, n, block):
        j = min(n, i + block)
        fc = float(np.clip(f_arr[min(len(f_arr) - 1, (i + j) // 2)], 35.0, nyq * 0.92))
        if btype == "bandpass":
            bw = max(fc / max(q, 0.5), 28.0)
            low = np.clip((fc - 0.5 * bw) / nyq, 0.001, 0.94)
            high = np.clip((fc + 0.5 * bw) / nyq, low + 0.002, 0.96)
            sos = signal.butter(2, [low, high], btype="bandpass", output="sos")
        elif btype == "low":
            sos = signal.butter(2, np.clip(fc / nyq, 0.002, 0.96), btype="low", output="sos")
        else:
            sos = signal.butter(2, np.clip(fc / nyq, 0.002, 0.96), btype="high", output="sos")
        if zi is None:
            zi = np.zeros((sos.shape[0], 2), dtype=np.float64)
        out[i:j], zi = signal.sosfilt(sos, sig[i:j], zi=zi)
    return out


def modal_strike(
    f0: float,
    dur: float,
    material: str = "wood",
    pitch_drop=0.0,
    rng: np.random.Generator = None,
    amp: float = 1.0,
) -> np.ndarray:
    """Physical modal resonator + contact transient for wood, marimba, ceramic, glass, bell, brass, timpani, metal."""
    if isinstance(pitch_drop, np.random.Generator):
        rng = pitch_drop
        pitch_drop = 0.0
    if rng is None:
        rng = np.random.default_rng(101)
    n = max(16, int(round(dur * SR)))
    t = np.arange(n, dtype=np.float64) / SR

    if material == "wood":
        ratios, amps, decays = [1.0, 2.42, 4.15, 6.32], [1.0, 0.45, 0.22, 0.08], [0.055, 0.028, 0.014, 0.007]
        click_freq, click_dur, click_amp = 1400.0, 0.0035, 0.42
    elif material == "marimba":
        ratios, amps, decays = [1.0, 3.98, 9.25, 14.8], [1.0, 0.38, 0.14, 0.04], [0.18, 0.058, 0.018, 0.006]
        click_freq, click_dur, click_amp = 1800.0, 0.003, 0.30
    elif material == "ceramic":
        ratios, amps, decays = (
            [1.0, 1.58, 2.37, 3.44],
            [0.9, 0.55, 0.32, 0.14],
            [0.085, 0.055, 0.032, 0.018],
        )
        click_freq, click_dur, click_amp = 2200.0, 0.0025, 0.28
    elif material == "glass":
        ratios, amps, decays = [1.0, 2.0, 3.0], [1.0, 0.28, 0.08], [0.38, 0.18, 0.075]
        click_freq, click_dur, click_amp = 1800.0, 0.0025, 0.14
    elif material == "bell":
        ratios = [0.5, 1.0, 1.501, 2.0, 2.44]
        amps = [0.32, 1.0, 0.36, 0.40, 0.14]
        decays = [0.65, 0.52, 0.34, 0.24, 0.12]
        click_freq, click_dur, click_amp = 2100.0, 0.0025, 0.18
    elif material == "brass":
        ratios, amps, decays = (
            [1.0, 1.498, 2.002, 2.76, 3.85, 5.12],
            [1.0, 0.55, 0.48, 0.28, 0.15, 0.08],
            [0.34, 0.25, 0.19, 0.11, 0.065, 0.035],
        )
        click_freq, click_dur, click_amp = 2600.0, 0.0025, 0.40
    elif material == "timpani":
        ratios, amps, decays = (
            [1.0, 1.504, 1.742, 2.00, 2.295],
            [1.0, 0.62, 0.38, 0.26, 0.14],
            [0.52, 0.34, 0.22, 0.16, 0.10],
        )
        click_freq, click_dur, click_amp = 420.0, 0.008, 0.35
    else:  # metal
        ratios, amps, decays = (
            [1.0, 1.38, 1.92, 2.78, 4.05, 5.88],
            [0.9, 0.65, 0.5, 0.35, 0.22, 0.12],
            [0.09, 0.06, 0.045, 0.03, 0.018, 0.01],
        )
        click_freq, click_dur, click_amp = 2900.0, 0.002, 0.55

    out = np.zeros(n, dtype=np.float64)
    if abs(float(pitch_drop)) > 1e-4:
        f_env = f0 * (1.0 + float(pitch_drop) * np.exp(-t / 0.035))
        phase_base = 2.0 * np.pi * np.cumsum(f_env) / SR
    else:
        phase_base = 2.0 * np.pi * f0 * t

    att = 1.0 - np.exp(-t / 0.0008)
    for r, a, d in zip(ratios, amps, decays):
        if f0 * r >= 0.46 * SR:
            continue
        det = 1.0 + rng.uniform(-0.003, 0.003)
        out += a * att * np.exp(-t / d) * np.sin(r * det * phase_base)

    c_len = min(int(click_dur * SR * 4), n)
    if c_len > 8:
        t_c = np.arange(c_len) / SR
        burst = rng.standard_normal(c_len) * np.exp(-t_c / click_dur)
        nyq = 0.5 * SR
        bw_low = np.clip(click_freq * 0.5 / nyq, 0.01, 0.85)
        bw_high = np.clip(click_freq * 1.6 / nyq, bw_low + 0.02, 0.95)
        transient = signal.sosfilt(signal.butter(2, [bw_low, bw_high], btype="bandpass", output="sos"), burst)
        mallet = np.sin(np.pi * np.clip(t_c / (click_dur * 1.5), 0.0, 1.0)) * np.exp(-t_c / click_dur)
        out[:c_len] += click_amp * (0.65 * transient + 0.35 * mallet)

    mx = float(np.max(np.abs(out))) + 1e-12
    return fade_in_out(out / mx, fade_in_s=0.0004, fade_out_s=0.015) * float(amp)


def minnaert_bubble(f0: float, dur: float = 0.034, sweep_mult: float = 2.15, amp: float = 1.0) -> np.ndarray:
    """Minnaert physical bubble resonance sweeping upward as bubble neck pinches off."""
    n = max(10, int(round(dur * SR)))
    t = np.arange(n, dtype=np.float64) / SR
    tau = dur * 0.75
    f_inst = f0 * (1.0 + (sweep_mult - 1.0) * np.power(np.clip(t / tau, 0.0, 1.2), 1.7))
    phase = 2.0 * np.pi * np.cumsum(f_inst) / SR
    env = (1.0 - np.exp(-t / 0.0022)) * np.exp(-t / (dur * 0.33))
    return fade_in_out(env * np.sin(phase), 0.0004, 0.004) * float(amp)


def pan_stereo_haas(mono: np.ndarray, pan: float = 0.0, haas_ms: float = 0.16) -> np.ndarray:
    """Equal-power stereo panner with subtle Haas micro-delay on the contralateral channel."""
    pan = float(np.clip(pan, -1.0, 1.0))
    angle = (pan + 1.0) * (np.pi / 4.0)
    g_l, g_r = np.cos(angle), np.sin(angle)
    n = len(mono)
    out = np.zeros((n, 2), dtype=np.float64)
    delay_samples = int(abs(pan) * (haas_ms * 0.001 * SR))
    if 0 < delay_samples < n:
        if pan < 0:
            out[:, 0] = mono * g_l
            out[delay_samples:, 1] = mono[:-delay_samples] * g_r
        else:
            out[delay_samples:, 0] = mono[:-delay_samples] * g_l
            out[:, 1] = mono * g_r
    else:
        out[:, 0] = mono * g_l
        out[:, 1] = mono * g_r
    return out


def add_studio_tail(
    stereo: np.ndarray,
    wet: float = 0.14,
    rt60: float = 0.26,
    damp_hz: float = 3800.0,
    room_size: float = None,
) -> np.ndarray:
    """Finite 10-tap early reflection + air-damped room tail so foley events sit in a real acoustic space."""
    if room_size is not None:
        rt60 = float(room_size)
    tail_len = int(rt60 * 1.25 * SR)
    n_in = len(stereo)
    total_len = n_in + tail_len
    padded = np.zeros((total_len, 2), dtype=np.float64)
    padded[:n_in] = stereo
    taps_ms = [6.7, 11.3, 17.9, 24.1, 32.3, 43.7, 56.9, 73.1, 94.7, 121.3]
    refl = np.zeros_like(padded)
    for idx, t_ms in enumerate(taps_ms):
        d = int(t_ms * 0.001 * SR)
        if d >= total_len:
            continue
        decay_gain = (10.0 ** (-3.0 * (t_ms * 0.001) / max(rt60, 0.05))) * (0.85 ** idx)
        if idx % 2 == 0:
            refl[d:, 0] += padded[:-d, 0] * decay_gain
            refl[d:, 1] += padded[:-d, 1] * decay_gain * 0.90
        else:
            refl[d:, 0] += padded[:-d, 1] * decay_gain * 0.88
            refl[d:, 1] += padded[:-d, 0] * decay_gain
    sos_lp = signal.butter(2, float(np.clip(damp_hz, 250.0, 14000.0)) / (0.5 * SR), btype="low", output="sos")
    sos_hp = signal.butter(1, 110.0 / (0.5 * SR), btype="high", output="sos")
    refl = signal.sosfilt(sos_lp, refl, axis=0)
    refl = signal.sosfilt(sos_hp, refl, axis=0)
    return fade_in_out(padded + wet * refl, fade_in_s=0.0004, fade_out_s=0.025)


def mix_into(stereo: np.ndarray, start_idx: int, sig: np.ndarray, pan: float = 0.0):
    """Mixes a mono or stereo signal into stereo buffer at `start_idx`."""
    if start_idx >= stereo.shape[0] or len(sig) == 0:
        return
    if start_idx < 0:
        sig = sig[-start_idx:]
        start_idx = 0
    end_idx = min(stereo.shape[0], start_idx + len(sig))
    n = end_idx - start_idx
    if n <= 0:
        return
    if sig.ndim == 2:
        stereo[start_idx:end_idx, 0] += sig[:n, 0]
        stereo[start_idx:end_idx, 1] += sig[:n, 1]
    else:
        panned = pan_stereo_haas(sig[:n], pan=pan)
        stereo[start_idx:end_idx] += panned


def _derive_topic_signature(cue_data: dict, base_seed: int = 77) -> dict:
    """
    Computes a deterministic musical variation signature from the video's scenes & cues so
    different topics within the same style have distinct key centers, chord progressions,
    melodic contours, and tempo breathing instead of identical background loops.
    """
    scenes = cue_data.get("scenes") or []
    cues = cue_data.get("cues") or []
    fingerprint = "|".join(
        [f"{s.get('id','')}:{s.get('title','')}:{s.get('look','')}:{s.get('dur',0)}" for s in scenes]
        + [f"{c.get('kind','')}:{c.get('label','')}" for c in cues[:16]]
    )
    h = zlib.crc32(fingerprint.encode("utf-8")) & 0xFFFFFFFF
    semitones = [-3, -2, 0, 2, 3, 4, 5][h % 7]
    pitch_mult = float(2.0 ** (semitones / 12.0))
    prog_orders = [
        [0, 1, 2, 3],
        [0, 3, 1, 2],
        [1, 3, 0, 2],
        [0, 2, 3, 1],
    ]
    arp_patterns = [
        [0, 2, 1, 3, 2, 4, 1, 2],
        [0, 1, 3, 2, 4, 3, 1, 2],
        [2, 0, 3, 1, 4, 2, 3, 0],
        [0, 3, 2, 4, 1, 3, 2, 0],
    ]
    tempo_scale = 0.94 + 0.03 * ((h >> 8) % 5)
    prog_order = prog_orders[(h >> 3) % len(prog_orders)]
    return {
        "hash": h,
        "seed": (int(base_seed) ^ int(h)) & 0xFFFFFFFF,
        "semitones": semitones,
        "pitch_mult": pitch_mult,
        "pitch_ratio": pitch_mult,
        "prog_order": prog_order,
        "chord_rot": prog_order[0],
        "arp_pattern": arp_patterns[(h >> 5) % len(arp_patterns)],
        "tempo_scale": tempo_scale,
    }


# =============================================================================
# 1. ACOUSTIC BODY & SPACE IMPULSE RESPONSE (IR) CONVOLUTION ENGINE
# =============================================================================


# =============================================================================
# 2. MEDIUM-SPECIFIC PHYSICAL MODELING & SCORE ENGINES
# =============================================================================


def _synth_felt_piano_voice(freq: float, dur: float, rng: np.random.Generator, vel: float = 0.85) -> np.ndarray:
    """Warm, velvety nocturnal felt-piano note (pure harmonic overtones, soft 18ms attack, zero noise click)."""
    n = max(16, int(round(dur * SR)))
    t = np.arange(n, dtype=np.float64) / SR
    attack = np.sin(0.5 * np.pi * np.clip(t / 0.018, 0.0, 1.0))
    decay = 0.72 * np.exp(-t / (dur * 0.55)) + 0.28 * np.exp(-t / (dur * 1.15))
    env = attack * decay
    sig = (
        0.76 * np.sin(2.0 * np.pi * freq * t)
        + 0.19 * np.sin(2.0 * np.pi * (freq * 2.0) * t + 0.2) * np.exp(-t / 0.38)
        + 0.05 * np.sin(2.0 * np.pi * (freq * 3.0) * t + 0.5) * np.exp(-t / 0.18)
    )
    sos_lp = signal.butter(2, min(1600.0, freq * 3.6) / (0.5 * SR), btype="low", output="sos")
    out = signal.sosfilt(sos_lp, sig * env) * 0.20 * vel
    return fade_in_out(out, 0.008, 0.04)


def _synth_celesta_bell(freq: float, dur: float, vel: float = 0.85) -> np.ndarray:
    """Warm, round music-box / soft celesta tone (zero metallic inharmonic partials, soft 14ms attack)."""
    n = max(16, int(round(dur * SR)))
    t = np.arange(n, dtype=np.float64) / SR
    attack = np.sin(0.5 * np.pi * np.clip(t / 0.014, 0.0, 1.0))
    env = attack * (0.75 * np.exp(-t / 0.58) + 0.25 * np.exp(-t / 1.25))
    tine = (
        0.82 * np.sin(2.0 * np.pi * freq * t)
        + 0.15 * np.sin(2.0 * np.pi * (freq * 2.0) * t + 0.15) * np.exp(-t / 0.26)
        + 0.03 * np.sin(2.0 * np.pi * (freq * 3.0) * t + 0.4) * np.exp(-t / 0.12)
    )
    sos_lp = signal.butter(2, 1850.0 / (0.5 * SR), btype="low", output="sos")
    return fade_in_out(signal.sosfilt(sos_lp, tine * env) * 0.15 * vel, 0.006, 0.04)


def synth_twilight_celesta_bed(
    dur: float,
    scenes: list,
    cfg: dict,
    rng: np.random.Generator,
    cues: list = None,
    sig: dict = None,
) -> np.ndarray:
    """
    Synthesizes a warm, soothing, Studio-Ghibli style nocturnal lake score (`twilight-fable`):
      - All instruments stay strictly in the warm middle register (65 Hz - 988 Hz), with zero metallic/high-frequency harshness.
      - Soft Felt-Piano Arpeggios + Warm Bowed Cello (`arco` & `pizz`) + Round Music-Box Celesta Melody.
    """
    sig = sig or _derive_topic_signature({"scenes": scenes, "cues": cues})
    n_samples = int(round(dur * SR))
    chords = cfg["chords"]
    stereo = synth_ambient_bed(dur, scenes, cfg, rng) * 0.82

    if not scenes:
        scenes = [{"start": 0.0, "dur": dur, "end": dur, "look": "twilight"}]

    # 1. Warm Cello Foundation (low-passed arco & soft wooden pizzicato)
    for idx, sc in enumerate(scenes):
        start_t = float(sc.get("start", 0.0))
        sc_dur = float(sc.get("dur", max(1.0, dur - start_t)))
        chord = chords[idx % len(chords)]

        if idx == 0:
            for b_i in range(int(sc_dur / 1.5)):
                bt = start_t + 0.2 + b_i * 1.5
                if bt < dur - 0.35:
                    f_b = chord[0] * 0.5 if (b_i % 2 == 0) else chord[2] * 0.5
                    pizz = synth_cello_voice(f_b, 0.95, "pizz", rng) * 0.58
                    mix_into(stereo, int(round(bt * SR)), pizz, pan=-0.18)
        elif idx == 1:
            c1 = synth_cello_voice(chord[0] * 0.5, min(sc_dur * 0.52, 2.5), "arco", rng) * 0.62
            c2 = synth_cello_voice(chord[2] * 0.5, min(sc_dur * 0.52, 2.5), "arco", rng) * 0.58
            mix_into(stereo, int(round((start_t + 0.12) * SR)), c1, pan=-0.20)
            mix_into(stereo, int(round((start_t + sc_dur * 0.46) * SR)), c2, pan=-0.15)
        elif idx == 2:
            for b_i in range(int(sc_dur / 1.15)):
                bt = start_t + 0.12 + b_i * 1.15
                if bt < dur - 0.35:
                    f_b = chord[b_i % 2] * 0.5
                    pizz = synth_cello_voice(f_b, 0.80, "pizz", rng) * 0.55
                    mix_into(stereo, int(round(bt * SR)), pizz, pan=-0.18)
            leap_arco = synth_cello_voice(chord[2] * 0.5, min(2.3, sc_dur * 0.52), "arco", rng) * 0.58
            mix_into(stereo, int(round((start_t + sc_dur * 0.40) * SR)), leap_arco, pan=0.16)
        else:
            c1 = synth_cello_voice(chord[0] * 0.5, min(sc_dur * 0.52, 2.4), "arco", rng) * 0.52
            c2 = synth_cello_voice(chord[2] * 0.5, min(sc_dur * 0.52, 2.4), "arco", rng) * 0.48
            mix_into(stereo, int(round((start_t + 0.12) * SR)), c1, pan=-0.18)
            mix_into(stereo, int(round((start_t + sc_dur * 0.46) * SR)), c2, pan=0.16)

    # 2. Unhurried, Breathing Felt-Piano Arpeggios & Warm Middle-Register Celesta Melody
    step_sec = 0.44
    arp_cycle = [0, 2, 3, 4, 2, 1, 3, 2]
    melody_degrees = [2, 3, 4, 2, 1, 2, 3, 1, 4, 3, 2, 0, 1, 2, 3, 4]
    total_steps = int(np.ceil(dur / step_sec))

    for s in range(total_steps):
        t_s = s * step_sec
        if t_s >= dur - 1.6:
            break
        sc_idx = 0
        for idx_s, sc in enumerate(scenes):
            if float(sc.get("start", 0.0)) <= t_s < float(sc.get("end", dur)):
                sc_idx = idx_s
                break

        if sc_idx == 0 and (s % 4 == 3):
            continue

        chord = chords[sc_idx % len(chords)]
        arp_deg = arp_cycle[s % len(arp_cycle)] % len(chord)
        # Keep piano strictly in warm middle register (196 Hz - 587 Hz)
        f_arp = chord[arp_deg] * (0.5 if arp_deg >= 3 and (s % 2 == 0) else 1.0)

        vel_arp = 0.62 if sc_idx == 0 else (0.76 if sc_idx in (1, 2) else 0.72)
        piano_note = _synth_felt_piano_voice(f_arp, 1.15, rng, vel=vel_arp)
        pan_arp = np.sin(s * 0.6 + sc_idx) * 0.28
        mix_into(stereo, int(round(t_s * SR)), piano_note, pan=pan_arp)

        # Lyrical Celesta melody strictly capped below 988 Hz (B5)
        play_melody = (s % 2 == 0) if sc_idx in (1, 2, 3) else (s % 3 == 0)
        if play_melody:
            m_deg = melody_degrees[(s // 2 + sc_idx * 2) % len(melody_degrees)] % len(chord)
            f_mel = chord[m_deg] * (1.0 if chord[m_deg] > 420.0 else 2.0)
            f_mel = min(f_mel, 987.77)
            vel_mel = 0.68 if sc_idx == 0 else (0.82 if sc_idx in (1, 3) else 0.78)
            cel = _synth_celesta_bell(f_mel, 1.25, vel=vel_mel)
            mix_into(stereo, int(round((t_s + 0.015) * SR)), cel, pan=-pan_arp * 0.8)

    # Gentle 3-note final resolution arpeggio after regular steps finish (no stacking)
    if dur > 2.2:
        final_t = max(0.0, dur - 1.55)
        final_chord = chords[(len(scenes) - 1) % len(chords)]
        for v_i, nf in enumerate(final_chord[:3]):
            p_note = _synth_felt_piano_voice(nf, 1.45, rng, vel=0.52)
            idx_sample = int(round((final_t + v_i * 0.22) * SR))
            mix_into(stereo, idx_sample, p_note, pan=-0.18 + 0.18 * v_i)

    # Silky, noise-free stereo hall shimmer (decorrelated smooth early reflections, zero comb harshness)
    reverb_l = np.zeros(n_samples, dtype=np.float64)
    reverb_r = np.zeros(n_samples, dtype=np.float64)
    for delay_ms, gain_l, gain_r in [(31.0, 0.14, 0.11), (53.0, 0.10, 0.13), (89.0, 0.08, 0.07), (142.0, 0.05, 0.06), (218.0, 0.035, 0.035)]:
        d = int(delay_ms * 0.001 * SR)
        if d < n_samples:
            reverb_l[d:] += stereo[:-d, 1] * gain_l
            reverb_r[d:] += stereo[:-d, 0] * gain_r
    sos_rev = signal.butter(2, 1400.0 / (0.5 * SR), btype="low", output="sos")
    stereo[:, 0] = stereo[:, 0] * 0.84 + signal.sosfilt(sos_rev, reverb_l)
    stereo[:, 1] = stereo[:, 1] * 0.84 + signal.sosfilt(sos_rev, reverb_r)
    return stereo


def synth_upright_bass_note(freq: float, dur: float, rng: np.random.Generator) -> np.ndarray:
    """Synthesizes a woody pizzicato bass note with fingerboard slap (the cello `pizz` voice)."""
    n = int(round(dur * SR))
    t = np.arange(n, dtype=np.float64) / SR
    env = np.clip(t / 0.006, 0.0, 1.0) * np.exp(-t / 0.24)
    body = (
        0.72 * np.sin(2.0 * np.pi * freq * t)
        + 0.42 * np.sin(4.0 * np.pi * freq * t + 0.3) * np.exp(-t / 0.14)
        + 0.18 * np.sin(6.0 * np.pi * freq * t) * np.exp(-t / 0.08)
    )
    # Woody finger slap at attack
    slap = rng.standard_normal(n) * np.exp(-t / 0.008) * 0.12
    sos = signal.butter(2, [650.0 / (0.5 * SR), 2200.0 / (0.5 * SR)], btype="band", output="sos")
    return body * env * 0.38 + signal.sosfilt(sos, slap)


def synth_cello_voice(freq: float, dur: float, mode: str, rng: np.random.Generator) -> np.ndarray:
    """Synthesizes a warm chamber Cello note (`arco` bowed sustain or `pizz` wooden pluck)."""
    n = max(16, int(round(dur * SR)))
    t = np.arange(n, dtype=np.float64) / SR
    if mode == "pizz":
        return synth_upright_bass_note(freq, dur, rng) * 0.88
    # Bowed `arco` with warm wooden body formants (no harsh high-frequency rosin hiss)
    attack = np.clip(t / min(0.28, dur * 0.25), 0.0, 1.0)
    release = np.clip((dur - t) / min(0.35, dur * 0.3), 0.0, 1.0)
    env = (np.sin(0.5 * np.pi * attack) ** 1.3) * np.sin(0.5 * np.pi * release)
    vib = 0.0042 * np.sin(2.0 * np.pi * 5.1 * t) * np.clip((t - 0.22) / 0.25, 0.0, 1.0)
    phase = 2.0 * np.pi * np.cumsum(freq * (1.0 + vib)) / SR
    out = np.zeros(n, dtype=np.float64)
    for k in range(1, 8):
        out += (0.52 / (k ** 1.18)) * np.sin(k * phase + 0.2 * k)
    sos_body = signal.butter(2, [95.0 / (0.5 * SR), 1050.0 / (0.5 * SR)], btype="band", output="sos")
    return signal.sosfilt(sos_body, out * env) * 0.30


def synth_ambient_bed(dur: float, scenes: list, cfg: dict, rng: np.random.Generator) -> np.ndarray:
    """Synthesizes a warm evolving stereo harmonic pad across scenes (the twilight bed sits on top of it)."""
    n_samples = int(round(dur * SR))
    stereo = np.zeros((n_samples, 2), dtype=np.float64)

    if not scenes:
        scenes = [{"start": 0.0, "dur": dur, "end": dur}]

    chords = cfg["chords"]
    for idx, sc in enumerate(scenes):
        start_t = float(sc.get("start", 0.0))
        sc_dur = float(sc.get("dur", max(1.0, dur - start_t)))
        pad_len = min(dur - start_t, sc_dur + 0.45)
        if pad_len <= 0.05:
            continue
        n = int(round(pad_len * SR))
        t = np.arange(n, dtype=np.float64) / SR
        chord = chords[idx % len(chords)]

        attack = np.clip(t / 0.45, 0.0, 1.0)
        release = np.clip((pad_len - t) / 0.55, 0.0, 1.0)
        env = np.sin(0.5 * np.pi * attack) * np.sin(0.5 * np.pi * release)

        left = np.zeros(n, dtype=np.float64)
        right = np.zeros(n, dtype=np.float64)
        for v_i, freq in enumerate(chord):
            detune = 1.0 + (v_i - 2) * 0.0018
            lfo = 1.0 + 0.12 * np.sin(2.0 * np.pi * (0.28 + 0.07 * v_i) * t)
            sig_l = np.sin(2.0 * np.pi * freq * t) + 0.25 * np.sin(4.0 * np.pi * freq * t)
            sig_r = np.sin(2.0 * np.pi * (freq * detune) * t + 0.4) + 0.25 * np.sin(4.0 * np.pi * (freq * detune) * t)
            weight = 0.45 / (1.0 + 0.35 * v_i)
            left += sig_l * lfo * weight
            right += sig_r * lfo * weight

        sos = signal.butter(2, 1600.0 / (0.5 * SR), btype="low", output="sos")
        left = signal.sosfilt(sos, left) * env * cfg["pad_gain"]
        right = signal.sosfilt(sos, right) * env * cfg["pad_gain"]

        s0 = int(round(start_t * SR))
        s1 = min(n_samples, s0 + n)
        stereo[s0:s1, 0] += left[: s1 - s0]
        stereo[s0:s1, 1] += right[: s1 - s0]

    return stereo


# =============================================================================
# 3. FRAME-ACCURATE SEMANTIC FOLEY & SCENE TRANSITION CUES
# =============================================================================


def synth_scene_cut(cfg: dict, intensity: float, idx: int, rng: np.random.Generator) -> np.ndarray:
    """Scene transition: a single warm felt-piano bell."""
    dur = 0.75
    f0 = cfg["chime_freqs"][idx % len(cfg["chime_freqs"])]
    warm_bell = _synth_felt_piano_voice(f0, dur, rng, vel=0.55)
    return warm_bell * (0.45 * intensity)


def synth_card_enter(cfg: dict, intensity: float, idx: int, rng: np.random.Generator) -> np.ndarray:
    """Soft wooden tap with a foliage-rustle grain for card/element entrances."""
    f0 = cfg["tap_freq"] * (1.0 + ((idx * 3) % 5) * 0.055)
    dur = 0.26
    n = int(round(dur * SR))
    strike = modal_strike(f0 * 0.85, dur, "wood", rng, amp=0.16)
    grain = slice_grain("foliage_paw_grain", n, rng, gain=0.28)
    return (strike[:n] + grain) * intensity


def synth_layer_float(cfg: dict, intensity: float, idx: int, rng: np.random.Generator) -> np.ndarray:
    """Harmonic lift shimmer: a round celesta bell."""
    dur = 0.46
    n = int(round(dur * SR))
    f1 = cfg["chime_freqs"][(idx + 1) % len(cfg["chime_freqs"])]
    warm_tone = _synth_celesta_bell(min(f1, 659.25), dur, vel=0.50)
    return warm_tone[:n] * (0.45 * intensity)


def synth_count_tick(cfg: dict, intensity: float, idx: int) -> np.ndarray:
    """Micro tick: a small water droplet."""
    f0 = cfg["tick_freq"] * (1.0 + (idx % 4) * 0.04)
    return minnaert_bubble(f0 * 0.72, 0.09, amp=0.22 * intensity)


def synth_stroke_draw(intensity: float, rng: np.random.Generator) -> np.ndarray:
    """Stroke / draw: a soft foliage rustle grain."""
    n = int(round(0.38 * SR))
    return slice_grain("foliage_paw_grain", n, rng, gain=0.52 * intensity)


def synth_runtime_foley(
    kind: str,
    intensity: float,
    rng: np.random.Generator,
    pitch: float = 1.0,
) -> np.ndarray:
    """Soft semantic foley for the twilight preset (paw step, ear perk, critter hop, water, shimmer, creak)."""
    p_scale = float(np.clip(pitch * rng.uniform(0.95, 1.05), 0.65, 1.55))

    if kind == "wave_surge":
        dur = 0.88
        n = int(round(dur * SR))
        t = np.arange(n, dtype=np.float64) / SR
        noise = generate_pink_noise(n, rng)
        f_sweep = 160.0 + 980.0 * np.sin(np.pi * t / dur) ** 1.4
        roar = time_varying_bandpass(noise, f_sweep, 1.4) * np.sin(np.pi * t / dur) * 0.28
        bubbles = slice_grain("stream_bubble_grain", n, rng, gain=0.52)
        return (roar + bubbles) * intensity

    if kind == "drink_sip":
        dur = 0.38
        n = int(round(dur * SR))
        b1 = minnaert_bubble(420.0 * p_scale, 0.14, amp=0.22)
        b2 = minnaert_bubble(560.0 * p_scale, 0.14, amp=0.18)
        clink = modal_strike(780.0 * p_scale, dur, "ceramic", rng, amp=0.16)
        out = clink[:n].copy()
        out[: len(b1)] += b1
        s2 = min(n - len(b2), int(0.08 * SR))
        if s2 > 0:
            out[s2 : s2 + len(b2)] += b2
        return out * intensity

    # --- TWILIGHT-FABLE FOLEY (100% Noise-Free, Silky Musical Foley) ---
    if kind == "paw_step":
        dur = 0.16
        n = int(round(dur * SR))
        t = np.arange(n, dtype=np.float64) / SR
        # Ultra-soft low velvet cushion thud (62Hz -> 40Hz) with smooth sine attack, zero noise
        f_pad = (40.0 + 22.0 * np.exp(-t / 0.025)) * p_scale
        pad = np.sin(2.0 * np.pi * np.cumsum(f_pad) / SR) * np.sin(0.5 * np.pi * np.clip(t / 0.012, 0.0, 1.0)) * np.exp(-t / 0.042)
        return fade_in_out(pad * 0.08 * intensity, 0.008, 0.03)

    if kind == "ear_perk":
        dur = 0.45
        return _synth_felt_piano_voice(392.00, dur, rng, vel=0.28) * (0.28 * intensity)

    if kind == "critter_hop":
        dur = 0.55
        n = int(round(dur * SR))
        n1 = _synth_felt_piano_voice(293.66, dur, rng, vel=0.32)
        n2 = _synth_felt_piano_voice(440.00, dur, rng, vel=0.28)
        out = n1[:n].copy()
        s2 = int(0.07 * SR)
        if s2 < n:
            out[s2:] += n2[: n - s2]
        return fade_in_out(out * (0.30 * intensity), 0.01, 0.05)

    if kind == "water_ripple":
        dur = 0.65
        n = int(round(dur * SR))
        t = np.arange(n, dtype=np.float64) / SR
        # Pure, round water droplet (smooth sine glide 330Hz -> 440Hz, soft 12ms attack, zero noise)
        f_drop = (330.0 + 110.0 * (1.0 - np.exp(-t / 0.045))) * p_scale
        env_d = np.sin(0.5 * np.pi * np.clip(t / 0.012, 0.0, 1.0)) * np.exp(-t / 0.085)
        drop = np.sin(2.0 * np.pi * np.cumsum(f_drop) / SR) * env_d * 0.11
        warm_echo = _synth_celesta_bell(523.25, dur, vel=0.26)
        out = drop + warm_echo[:n] * 0.45
        return fade_in_out(out * (0.32 * intensity), 0.01, 0.05)

    if kind == "spirit_shimmer":
        dur = 0.85
        n = int(round(dur * SR))
        n1 = _synth_felt_piano_voice(392.00, dur, rng, vel=0.34)
        n2 = _synth_celesta_bell(587.33, dur, vel=0.28)
        out = n1[:n].copy()
        s2 = int(0.11 * SR)
        if s2 < n:
            out[s2:] += n2[: n - s2]
        return fade_in_out(out * (0.32 * intensity), 0.015, 0.06)

    if kind == "wood_creak":
        dur = 0.48
        n = int(round(dur * SR))
        t = np.arange(n, dtype=np.float64) / SR
        # Stick-slip wooden door / branch creak
        rate = 28.0 + 18.0 * np.sin(np.pi * t / dur)
        pulses = np.maximum(0.0, np.sin(2.0 * np.pi * np.cumsum(rate) / SR)) ** 6.0
        exc = pulses * generate_pink_noise(n, rng)
        creak = time_varying_bandpass(exc, (360.0 + 140.0 * np.sin(2.0 * np.pi * 2.5 * t)) * p_scale, 5.5) * 0.45
        body = modal_strike(210.0 * p_scale, dur, "wood", rng, amp=0.22)
        return (creak + body[:n]) * intensity

    if kind == "foot_step":
        return synth_runtime_foley("paw_step", intensity, rng, pitch=pitch)

    return synth_stroke_draw(intensity, rng)


FOLEY_ACTIONS = {
    "paw_step",
    "ear_perk",
    "critter_hop",
    "water_ripple",
    "spirit_shimmer",
    "wood_creak",
    "foot_step",
    "wave_surge",
    "drink_sip",
    "brush_sweep",
}


def _resolve_cue_semantic_action(kind: str, label: str) -> str:
    """Maps (kind, label) from HTML data-cues, [data-sfx], and runtime cues to a twilight synthesis primitive."""
    k = (kind or "").strip().lower()
    lbl = (label or "").strip().lower()

    if k in FOLEY_ACTIONS:
        return k
    if lbl in FOLEY_ACTIONS:
        return lbl

    # Direct [data-sfx] aliases
    if k in ("whoosh", "swipe"):
        return "ear_perk"
    if k == "chime":
        return "spirit_shimmer"
    if k == "thud":
        return "foot_step"

    for token in (lbl, k):
        if not token:
            continue
        if any(w in token for w in ("paw", "pad_step", "forest_step", "moss_step", "trot")):
            return "paw_step"
        if any(w in token for w in ("ear", "perk", "whisker", "blink", "sniff", "head_tilt")):
            return "ear_perk"
        if any(w in token for w in ("creak", "door", "trunk", "bark", "hollow", "branch")):
            return "wood_creak"
        if any(w in token for w in ("ripple", "pond", "stream", "brook", "water", "reflection", "lotus")):
            return "water_ripple"
        if any(w in token for w in ("spirit", "lantern", "firefly", "spore", "glow", "wisp", "beacon", "starlight")):
            return "spirit_shimmer"
        if any(w in token for w in ("fawn", "deer", "fox", "cat", "bunny", "rabbit", "squirrel", "critter", "leap", "bound")):
            return "critter_hop"
        if any(w in token for w in ("musicbox", "carousel", "windup", "chime", "bell", "ding", "sparkle")):
            return "spirit_shimmer"
        if any(w in token for w in ("boing", "hop", "bounce", "spring", "jump", "glove")):
            return "critter_hop"
        if "wave" in token or "surge" in token:
            return "wave_surge"
        if "drink" in token or "wine" in token or "sip" in token:
            return "drink_sip"
        if "step" in token or "walk" in token or "stride" in token:
            return "paw_step"
        if any(w in token for w in ("xingcao", "verse", "colophon", "brush", "stroke", "inscription")):
            return "stroke_draw"

    if k in ("scene_cut", "cut"):
        return "scene_cut"
    if k in ("layer_float", "float", "mist"):
        return "layer_float"
    if k in ("count_tick", "tick"):
        return "count_tick"
    if k in ("stroke_draw", "stroke"):
        return "stroke_draw"
    if k in ("impact", "hit", "slam"):
        return "wood_creak"
    return "card_enter"


def master_bus(stereo: np.ndarray, ceiling_db: float = -1.2) -> np.ndarray:
    """Warm tonal shaping, DC block, dynamic-range-preserving leveling, and true-peak limiting."""
    # High-pass to remove DC/rumble
    sos_hp = signal.butter(2, 28.0 / (0.5 * SR), btype="high", output="sos")
    out = signal.sosfilt(sos_hp, stereo, axis=0)

    # Warm picture-book nocturnal tone: roll off harsh highs above 2.6 kHz
    sos_lp = signal.butter(2, 2600.0 / (0.5 * SR), btype="low", output="sos")
    out = signal.sosfilt(sos_lp, out, axis=0)

    # Measure RMS only on active (non-silent) regions so dramatic silence holds are never pumped up
    mono_abs = np.max(np.abs(out), axis=1)
    active_mask = mono_abs > 1.5e-3
    if np.any(active_mask):
        active_rms = np.sqrt(np.mean(out[active_mask] ** 2) + 1e-12)
    else:
        active_rms = np.sqrt(np.mean(out ** 2) + 1e-12)

    # Preserve wide crest factor for a gentle score while keeping healthy loudness
    target_rms = 0.095
    max_gain = 1.8
    gain = float(np.clip(target_rms / active_rms, 0.55, max_gain))
    out = out * gain

    # True-peak ceiling at ceiling_db (-1.2 dBTP)
    ceiling = 10.0 ** (ceiling_db / 20.0)
    peak = float(np.max(np.abs(out)))
    if peak > ceiling:
        out *= ceiling / peak
    elif peak > 1e-5:
        out *= min(1.22, ceiling / peak)

    # Gentle 40ms fade-in and 120ms fade-out
    n = out.shape[0]
    fi = min(n, int(0.04 * SR))
    fo = min(n, int(0.12 * SR))
    if fi > 0:
        out[:fi] *= np.linspace(0.0, 1.0, fi)[:, None]
    if fo > 0:
        out[-fo:] *= np.linspace(1.0, 0.0, fo)[:, None]

    return out


def _load_voice(path: str, n: int) -> np.ndarray:
    """Read a mono/stereo voice WAV, resample to 48 kHz, and fit it to `n` samples."""
    try:
        import soundfile as sf
    except ImportError:
        raise SystemExit("sound.py: --voice needs soundfile (pip install soundfile)")
    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception as e:
        raise SystemExit(f"sound.py: cannot read voice '{path}': {e}")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.reshape(-1)
    if sr != SR and audio.size:
        audio = signal.resample_poly(audio, SR, sr).astype(np.float32)
    if audio.size < n:
        audio = np.pad(audio, (0, n - audio.size))
    else:
        audio = audio[:n]
    return audio


def voice_envelope(voice: np.ndarray, attack: float = 0.02, release: float = 0.25) -> np.ndarray:
    """10 ms RMS envelope (0..1) with a fast attack and a slow release for sidechain ducking."""
    hop = max(1, int(0.01 * SR))
    frames = voice.size // hop
    if frames <= 0:
        return np.zeros(voice.size, dtype=np.float32)
    rms = np.sqrt(np.mean(voice[: frames * hop].reshape(frames, hop) ** 2, axis=1) + 1e-12)
    ref = float(np.percentile(rms, 95))
    if ref < 1e-5:
        return np.zeros(voice.size, dtype=np.float32)
    target = np.clip((rms / ref - 0.04) / 0.96, 0.0, 1.0)
    ca = 1.0 - np.exp(-hop / (max(float(attack), 1e-3) * SR))
    cr = 1.0 - np.exp(-hop / (max(float(release), 1e-3) * SR))
    env = np.empty_like(target)
    y = 0.0
    for i, x in enumerate(target):
        y += (x - y) * (ca if x > y else cr)
        env[i] = y
    env = np.repeat(env, hop)
    if env.size < voice.size:
        env = np.pad(env, (0, voice.size - env.size), constant_values=float(env[-1]) if env.size else 0.0)
    return env[: voice.size].astype(np.float32)


def mix_voice_into(
    stereo: np.ndarray,
    voice_path: str,
    duck_db: float = -10.0,
    voice_gain_db: float = 0.0,
    attack: float = 0.02,
    release: float = 0.25,
    ceiling_db: float = -1.2,
) -> np.ndarray:
    """
    Carve a Mid-channel spectral vocal pocket (220 Hz - 3800 Hz) under the voice with 18 ms lookahead,
    preserving <220 Hz bass fundamentals and stereo Side ambience, then blend the voice with a subtle
    decorrelated early-reflection halo and soft-knee true-peak limiting.
    """
    n = stereo.shape[0]
    voice = _load_voice(voice_path, n)
    if not np.any(voice):
        return stereo
    duck_db = min(0.0, float(duck_db))
    bg = np.asarray(stereo, dtype=np.float64)
    v = np.asarray(voice, dtype=np.float64)

    if duck_db < 0.0:
        lookahead = int(0.018 * SR)
        v_det = np.pad(v[lookahead:], (0, min(lookahead, v.size))) if lookahead < v.size else v
        env = voice_envelope(v_det.astype(np.float32), attack=attack, release=max(release, 0.28)).astype(np.float64)

        mid = 0.5 * (bg[:, 0] + bg[:, 1])
        side = 0.5 * (bg[:, 0] - bg[:, 1])
        sos_formant = signal.butter(2, [220.0 / (0.5 * SR), 3800.0 / (0.5 * SR)], btype="band", output="sos")
        mid_formant = signal.sosfilt(sos_formant, mid)
        mid_outer = mid - mid_formant

        floor_mid = 10.0 ** (duck_db / 20.0)
        floor_outer = 10.0 ** ((duck_db * 0.35) / 20.0)
        floor_side = 10.0 ** ((duck_db * 0.22) / 20.0)

        mid_ducked = mid_formant * (1.0 - (1.0 - floor_mid) * env) + mid_outer * (1.0 - (1.0 - floor_outer) * env)
        side_ducked = side * (1.0 - (1.0 - floor_side) * env)
        bg = np.column_stack((mid_ducked + side_ducked, mid_ducked - side_ducked))

    v_scaled = v * (10.0 ** (float(voice_gain_db) / 20.0))
    sos_halo = signal.butter(2, [200.0 / (0.5 * SR), 2800.0 / (0.5 * SR)], btype="band", output="sos")
    v_halo = signal.sosfilt(sos_halo, v_scaled) * 0.18
    d_l, d_r = int(0.013 * SR), int(0.021 * SR)
    halo_l = np.pad(v_halo[:-d_l], (d_l, 0)) if d_l < n else np.zeros_like(v_scaled)
    halo_r = -np.pad(v_halo[:-d_r], (d_r, 0)) if d_r < n else np.zeros_like(v_scaled)

    out = np.column_stack((bg[:, 0] + v_scaled + halo_l, bg[:, 1] + v_scaled + halo_r))
    ceiling = 10.0 ** (float(ceiling_db) / 20.0)
    knee = ceiling * 0.78
    mag = np.abs(out)
    over = mag > knee
    if np.any(over):
        sign = np.sign(out[over])
        excess = (mag[over] - knee) / max(ceiling - knee, 1e-6)
        out[over] = sign * (knee + (ceiling - knee) * np.tanh(excess))
    peak = float(np.max(np.abs(out)))
    if peak > ceiling:
        out = out * (ceiling / peak)
    return out.astype(np.float32)


def synthesize_from_cues(cue_data: dict, preset: str = "twilight", seed: int = 77) -> np.ndarray:
    preset = resolve_preset(preset)
    cfg = PRESETS[preset]
    dur = float(cue_data.get("dur", 6.0))
    scenes = cue_data.get("scenes", [])
    cues = cue_data.get("cues", [])

    # Derive a topic-unique acoustic signature so different stories get distinct keys, tempos & chord rotations
    sig = _derive_topic_signature(cue_data, base_seed=seed)
    rng = np.random.default_rng(sig["seed"])

    stereo = synth_twilight_celesta_bed(dur, scenes, cfg, rng, cues=cues, sig=sig)

    room_size = 0.34
    room_wet = 0.14

    last_action_t = {}
    for idx, cue in enumerate(cues):
        t = float(cue.get("t", 0.0))
        kind = str(cue.get("kind", "card_enter"))
        label = str(cue.get("label", ""))
        intensity = float(cue.get("intensity", 0.75))
        pitch = float(cue.get("pitch", 1.0))
        start_sample = int(round(t * SR))

        # Per-event deterministic RNG seeded by CRC32 so every cue instance has organic micro-variation
        cue_crc = zlib.crc32(f"{preset}:{idx}:{kind}:{label}:{int(round(t * 1000))}".encode("utf-8")) & 0xFFFFFFFF
        cue_rng = np.random.default_rng(cue_crc)

        if "pan" in cue and cue["pan"] is not None:
            pan = float(np.clip(float(cue["pan"]), -0.75, 0.75))
        else:
            pan = float(np.clip(((cue_crc % 100) - 50) / 125.0, -0.40, 0.40))

        action = _resolve_cue_semantic_action(kind, label)
        if action != "scene_cut":
            min_gap = 0.65 if action in ("paw_step", "critter_hop", "foot_step") else 0.40
            if action in last_action_t and (t - last_action_t[action]) < min_gap:
                continue
            last_action_t[action] = t

        if action == "scene_cut":
            mono = synth_scene_cut(cfg, intensity, idx, cue_rng)
            pan = 0.0
        elif action == "layer_float":
            mono = synth_layer_float(cfg, intensity, idx, cue_rng)
        elif action == "count_tick":
            mono = synth_count_tick(cfg, intensity, idx)
        elif action == "stroke_draw":
            mono = synth_stroke_draw(intensity, cue_rng)
        elif action in FOLEY_ACTIONS:
            mono = synth_runtime_foley(action, intensity, cue_rng, pitch=pitch)
        else:
            mono = synth_card_enter(cfg, intensity, idx, cue_rng)

        cue_st = pan_stereo_haas(mono, pan=pan)
        cue_st = add_studio_tail(cue_st, room_size=room_size, wet=room_wet)
        n_c = cue_st.shape[0]
        if start_sample < stereo.shape[0] and n_c > 0:
            s0 = max(0, start_sample)
            c0 = max(0, -start_sample)
            length = min(stereo.shape[0] - s0, n_c - c0)
            if length > 0:
                stereo[s0 : s0 + length] += cue_st[c0 : c0 + length]

    return master_bus(stereo, ceiling_db=-1.2)


def write_wav(path: Path, stereo: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = np.clip(stereo * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm16.tobytes())


def default_cue_sheet() -> dict:
    """Fallback 4-scene cue sheet when --cues is omitted in standalone CLI invocations."""
    return {
        "dur": 14.0,
        "scenes": [
            {"index": 0, "id": "scene-01", "start": 0.0, "dur": 3.5, "end": 3.5},
            {"index": 1, "id": "scene-02", "start": 3.5, "dur": 3.7, "end": 7.2},
            {"index": 2, "id": "scene-03", "start": 7.2, "dur": 3.6, "end": 10.8},
            {"index": 3, "id": "scene-04", "start": 10.8, "dur": 3.2, "end": 14.0},
        ],
        "cues": [
            {"t": 0.0, "kind": "scene_cut", "intensity": 0.8},
            {"t": 0.18, "kind": "card_enter", "intensity": 0.78},
            {"t": 0.42, "kind": "stroke_draw", "intensity": 0.65},
            {"t": 1.1, "kind": "layer_float", "intensity": 0.85},
            {"t": 1.6, "kind": "water_ripple", "intensity": 0.72},
            {"t": 3.5, "kind": "scene_cut", "intensity": 1.0},
            {"t": 3.72, "kind": "card_enter", "intensity": 0.75},
            {"t": 3.95, "kind": "count_tick", "intensity": 0.55},
            {"t": 4.5, "kind": "critter_hop", "intensity": 0.85},
            {"t": 7.2, "kind": "scene_cut", "intensity": 1.0},
            {"t": 7.45, "kind": "card_enter", "intensity": 0.75},
            {"t": 7.8, "kind": "spirit_shimmer", "intensity": 0.8},
            {"t": 8.3, "kind": "layer_float", "intensity": 0.85},
            {"t": 10.8, "kind": "scene_cut", "intensity": 1.0},
            {"t": 11.05, "kind": "card_enter", "intensity": 0.75},
            {"t": 12.0, "kind": "layer_float", "intensity": 0.85},
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Cue-driven procedural sound & physical modeling score synthesizer for StyleHop.")
    parser.add_argument("--cues", required=False, default=None, help="Path to cues.json extracted from window.CUES")
    parser.add_argument("--out", required=True, help="Output 48kHz stereo WAV path")
    parser.add_argument("--preset", default="twilight", help="Sonic palette preset (only `twilight` / `twilight-fable`)")
    parser.add_argument("--voice", default=None, help="48 kHz mono/stereo voice WAV to mix over the score")
    parser.add_argument("--duck-db", type=float, default=-10.0, help="Music gain reduction under speech (0 disables ducking)")
    parser.add_argument("--voice-gain", type=float, default=0.0, help="Voice gain in dB before the final limiter")
    parser.add_argument("--duck-attack", type=float, default=0.02, help="Sidechain attack in seconds")
    parser.add_argument("--duck-release", type=float, default=0.25, help="Sidechain release in seconds")
    args = parser.parse_args()

    if args.cues:
        cue_path = Path(args.cues)
        cue_data = json.loads(cue_path.read_text(encoding="utf-8"))
    else:
        cue_data = default_cue_sheet()

    preset = resolve_preset(cue_data.get("soundPreset") or args.preset)

    stereo = synthesize_from_cues(cue_data, preset=preset)
    if args.voice:
        voice_path = Path(args.voice)
        if not voice_path.is_file():
            raise SystemExit(f"sound.py: voice file not found: {voice_path}")
        stereo = mix_voice_into(
            stereo,
            str(voice_path),
            duck_db=args.duck_db,
            voice_gain_db=args.voice_gain,
            attack=args.duck_attack,
            release=args.duck_release,
        )
    out_path = Path(args.out)
    write_wav(out_path, stereo)
    peak_db = 20.0 * np.log10(max(1e-9, float(np.max(np.abs(stereo)))))
    rms_db = 20.0 * np.log10(max(1e-9, float(np.sqrt(np.mean(stereo ** 2)))))
    print(
        json.dumps(
            {
                "out": str(out_path),
                "preset": preset,
                "dur": float(cue_data.get("dur", 0)),
                "cues": len(cue_data.get("cues", [])),
                "voice": str(Path(args.voice)) if args.voice else None,
                "peak_dbtp": round(peak_db, 2),
                "rms_dbfs": round(rms_db, 2),
            }
        )
    )


if __name__ == "__main__":
    main()
