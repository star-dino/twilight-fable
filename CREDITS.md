# Credits & Third-Party Notices (`CREDITS.md`)

## 1. Author & AI Generation Disclosure

- **Author / Maintainer**: **Star Dinosaur (星星恐龙)**
- **License**: [MIT License](LICENSE)
- **AI Generation Statement**: The procedural animation scripts, level simulations, shaders, sound design recipes, and skill documentation in this repository were generated and iteratively refined using **Gemini 4**.

---

## 2. Upstream Open-Source Toolchains (MIT License)

This repository was split from `stardino-stylehop` at commit `bdf4dfb`; it inherits the following notices. It builds upon the architecture, procedural audio synthesis, and sound mastering toolchains of two parent MIT-licensed projects:

### A. `lemo-opuscar`
- **Repository**: [https://github.com/lemomo-ai/lemo-opuscar](https://github.com/lemomo-ai/lemo-opuscar)
- **Copyright**: `Copyright (c) 2026 LemoLab`
- **License**: MIT License
- **License text**: full MIT notice reproduced in [LICENSE-THIRD-PARTY](LICENSE-THIRD-PARTY) because this repository contained **adapted** code, not just imports (the files below lived in `showcase/` of the parent repository `stardino-stylehop` and are not part of this repository):
  - `showcase/corelib/audio_dsp.py` — `quick_click` / `quick_thump` / `quick_whoosh` are ports of upstream `core/audio/sfx.py` (`click` / `thump` / `whoosh`), refactored onto local helper functions but sharing the same durations, partials and envelopes.
  - `showcase/worlds/*/audio_*.py` — synthesis recipes (instrument layering, filter settings, mastering targets) derived from upstream style demos.
- **Distribution**: the library itself is a **runtime dependency only**, cloned separately via `LEMO_OPUSCAR_ROOT`; no upstream source tree is vendored here.

### B. `earshot`
- **Repository**: [https://github.com/Leonxlnx/earshot](https://github.com/Leonxlnx/earshot)
- **Copyright**: `Copyright (c) 2026 Leon Lin`
- **License**: MIT License (full text reproduced in [LICENSE-THIRD-PARTY](LICENSE-THIRD-PARTY))
- **Usage (inherited from `stardino-stylehop`)**:
  - Acoustically fitted modal/spectral Foley bank (`earshot.forge.bank`) and EBU R128 loudness mastering & sidechain ducking (`master.py`).
- **Distribution**: referenced as a **runtime dependency only** — cloned separately via `EARSHOT_ROOT`; no earshot source is vendored in this repository.

### C. Optional voice engines (`tools/voice.py`)
- `tools/voice.py` follows the `lines.json` CLI contract and trimming/default behaviour of the upstream MIT `lemo-opuscar` voice tools (`core/tts/tts.py`, `core/tts/tts_zh.py`), reimplemented for StyleHop's scene anchors, cache, timing manifest and SRT output.
- **Kokoro-82M** ([hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)): Apache-2.0 model weights; `kokoro-onnx` runtime is a separate optional dependency. Local/offline.
- **edge-tts** ([rany2/edge-tts](https://github.com/rany2/edge-tts)): LGPLv3 tool, not vendored; it calls Microsoft's online Edge speech service, whose terms must be checked separately before commercial use.
- **soundfile** ([libsndfile](https://github.com/libsndfile/libsndfile)): BSD-3-Clause; used only when `--voice` reads/writes WAV.
- ChatTTS, F5-TTS, Spark-TTS, Fish Speech and similar CC-BY-NC / CC-BY-NC-SA weights are **not** supported for deliverable output and are not bundled.
