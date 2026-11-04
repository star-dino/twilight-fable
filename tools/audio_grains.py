#!/usr/bin/env python3
"""
Twilight Fable acoustic grain generator (`tools/audio_grains.py`).

Generates two deterministic 48kHz 16-bit WAV micro-samples in `assets/audio/` (git-ignored,
auto-built on demand by `tools/sound.py`):

  - `foliage_paw_grain.wav`   moss/grass/leaf rustle and soft paw-pad compression
  - `stream_bubble_grain.wav` brook turbulence plus a Minnaert bubble pinch-off cascade

Usage:
  python3 tools/audio_grains.py [--out-dir assets/audio]
"""

import argparse
from pathlib import Path
import wave
import numpy as np
from scipy import signal

SR = 48000


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0 / SR)
    scale = 1.0 / np.sqrt(np.maximum(freqs, 20.0))
    scale[0] = 0.0
    out = np.fft.irfft(spec * scale, n=n)
    return out / (np.max(np.abs(out)) + 1e-12)


def bp(sig: np.ndarray, lo: float, hi: float, order: int = 2) -> np.ndarray:
    nyq = 0.5 * SR
    lo_n = float(np.clip(lo / nyq, 1e-4, 0.96))
    hi_n = float(np.clip(hi / nyq, lo_n + 1e-3, 0.99))
    sos = signal.butter(order, [lo_n, hi_n], btype="bandpass", output="sos")
    return signal.sosfilt(sos, sig)


def lp(sig: np.ndarray, fc: float, order: int = 2) -> np.ndarray:
    sos = signal.butter(order, float(np.clip(fc / (0.5 * SR), 1e-4, 0.98)), btype="low", output="sos")
    return signal.sosfilt(sos, sig)


def fade_norm(sig: np.ndarray, fi_s: float = 0.002, fo_s: float = 0.015, peak: float = 0.92) -> np.ndarray:
    n = len(sig)
    out = sig.copy()
    fi = min(int(fi_s * SR), n // 3)
    fo = min(int(fo_s * SR), n // 3)
    if fi > 1:
        w = 0.5 * (1.0 - np.cos(np.pi * np.arange(fi) / fi))
        out[:fi] *= w[:, None] if out.ndim == 2 else w
    if fo > 1:
        w = 0.5 * (1.0 + np.cos(np.pi * np.arange(fo) / fo))
        out[-fo:] *= w[:, None] if out.ndim == 2 else w
    mx = float(np.max(np.abs(out))) + 1e-12
    return out * (peak / mx)


def minnaert_bubble(f0: float, dur: float = 0.034, mult: float = 2.15) -> np.ndarray:
    n = max(12, int(round(dur * SR)))
    t = np.arange(n, dtype=np.float64) / SR
    tau = dur * 0.74
    f_inst = f0 * (1.0 + (mult - 1.0) * np.power(np.clip(t / tau, 0.0, 1.2), 1.7))
    phase = 2.0 * np.pi * np.cumsum(f_inst) / SR
    env = (1.0 - np.exp(-t / 0.002)) * np.exp(-t / (dur * 0.34))
    return env * np.sin(phase)


def add_at(buf: np.ndarray, sig: np.ndarray, t_sec: float, gain: float = 1.0):
    i0 = int(round(t_sec * SR))
    if i0 < 0 or i0 >= len(buf):
        return
    i1 = min(len(buf), i0 + len(sig))
    buf[i0:i1] += sig[: i1 - i0] * gain


# =============================================================================
# GRAIN RECIPES
# =============================================================================


def gen_foliage_paw_grain(seed: int = 1004) -> np.ndarray:
    """Warm, soft velvet paw pad on damp moss (low-passed, zero harsh high-frequency hiss)."""
    rng = np.random.default_rng(seed)
    dur = 0.80
    n = int(dur * SR)
    out = np.zeros(n, dtype=np.float64)

    for idx, t0 in enumerate([0.02, 0.28, 0.54]):
        gn = int(0.22 * SR)
        gt = np.arange(gn, dtype=np.float64) / SR
        f_pad = 48.0 + 32.0 * np.exp(-gt / 0.022)
        pad = np.sin(2.0 * np.pi * np.cumsum(f_pad) / SR) * np.exp(-gt / 0.045) * (1.0 - np.exp(-gt / 0.004)) * 0.55
        swish = bp(pink_noise(gn, rng), 220.0, 1100.0) * np.exp(-((gt - 0.05) / 0.045) ** 2) * 0.18
        add_at(out, lp(pad + swish, 1300.0), t0, 1.0 - 0.08 * idx)

    return fade_norm(out, 0.005, 0.03, peak=0.48)


def gen_stream_bubble_grain(seed: int = 1005) -> np.ndarray:
    """Gentle, warm nocturnal lake ripple & soft low-mid water droplets (1.35s)."""
    rng = np.random.default_rng(seed)
    dur = 1.35
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float64) / SR

    lfo1 = np.sin(2.0 * np.pi * 1.4 * t + 0.3) * 0.5 + 0.5
    stream_lo = bp(pink_noise(n, rng), 180.0, 720.0) * (0.45 + 0.35 * lfo1) * 0.30
    out = stream_lo

    for at in np.sort(rng.uniform(0.05, dur - 0.08, 9)):
        f0 = float(rng.uniform(280.0, 620.0))
        bdur = float(rng.uniform(0.032, 0.058))
        bub = minnaert_bubble(f0, bdur, 1.42)
        add_at(out, bub, float(at), float(rng.uniform(0.20, 0.48)))

    return fade_norm(lp(out, 1500.0), 0.02, 0.04, peak=0.52)


def save_wav(path: Path, data: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = np.clip(np.round(data * 32767.0), -32768, 32767).astype(np.int16)
    nch = 2 if data.ndim == 2 else 1
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(nch)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm16.tobytes())


RECIPES = {
    "foliage_paw_grain.wav": gen_foliage_paw_grain,
    "stream_bubble_grain.wav": gen_stream_bubble_grain,
}


def ensure_audio_grains(out_dir: Path = None) -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    target_dir = out_dir if out_dir is not None else (repo_root / "assets" / "audio")
    target_dir.mkdir(parents=True, exist_ok=True)
    for fname, fn in RECIPES.items():
        fpath = target_dir / fname
        if not fpath.exists():
            sig = fn()
            save_wav(fpath, sig)
    return target_dir


def main():
    parser = argparse.ArgumentParser(description="Generate hybrid physical audio grains in assets/audio/.")
    parser.add_argument("--out-dir", type=str, default="assets/audio", help="Output directory")
    parser.add_argument("--force", action="store_true", help="Overwrite existing grain files")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    for fname, fn in RECIPES.items():
        dest = out_dir / fname
        if args.force or not dest.exists():
            sig = fn()
            save_wav(dest, sig)
        sz = dest.stat().st_size
        total_bytes += sz
        print(f"[audio_grains] wrote {dest.relative_to(repo_root)} ({sz / 1024:.1f} KB)")
    print(f"[audio_grains] complete: {len(RECIPES)} files, {total_bytes / 1024:.1f} KB total")


if __name__ == "__main__":
    main()
