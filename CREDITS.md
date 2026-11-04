# Credits & Third-Party Notices (`CREDITS.md`)

## 1. Author & AI Generation Disclosure

- **Author / Maintainer**: **Star Dinosaur (星星恐龙)**
- **License**: [MIT License](LICENSE)
- **AI Generation Statement**: The procedural animation scripts, level simulations, shaders, sound design recipes, and skill documentation in this repository were generated and iteratively refined with AI coding assistants: **Gemini** (the original `stardino-stylehop` work) and **Anthropic Claude** (the standalone split, the recorded-sound base and its tooling; commits carry a `Co-Authored-By` trailer).

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
- **Distribution**: no upstream source tree is vendored in this repository, and no file here reads `LEMO_OPUSCAR_ROOT` or imports the library; the notice is kept because code and recipes adapted from it existed in the parent repository this one was split from.

### B. `earshot`
- **Repository**: [https://github.com/Leonxlnx/earshot](https://github.com/Leonxlnx/earshot)
- **Copyright**: `Copyright (c) 2026 Leon Lin`
- **License**: MIT License (full text reproduced in [LICENSE-THIRD-PARTY](LICENSE-THIRD-PARTY))
- **Usage (inherited from `stardino-stylehop`)**:
  - Acoustically fitted modal/spectral Foley bank (`earshot.forge.bank`) and EBU R128 loudness mastering & sidechain ducking (`master.py`).
- **Distribution**: no earshot source is vendored in this repository, and no file here reads `EARSHOT_ROOT` or imports it; the notice is kept because API-compatible mastering code and recipes derived from it existed in the parent repository.

### C. Optional voice engines (`tools/voice.py`)
- `tools/voice.py` follows the `lines.json` CLI contract and trimming/default behaviour of the upstream MIT `lemo-opuscar` voice tools (`core/tts/tts.py`, `core/tts/tts_zh.py`), reimplemented for StyleHop's scene anchors, cache, timing manifest and SRT output.
- **Kokoro-82M** ([hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)): Apache-2.0 model weights; `kokoro-onnx` runtime is a separate optional dependency. Local/offline.
- **edge-tts** ([rany2/edge-tts](https://github.com/rany2/edge-tts)): LGPLv3 tool, not vendored; it calls Microsoft's online Edge speech service, whose terms must be checked separately before commercial use.
- **soundfile** ([libsndfile](https://github.com/libsndfile/libsndfile)): BSD-3-Clause; used only when `--voice` reads/writes WAV.
- ChatTTS, F5-TTS, Spark-TTS, Fish Speech and similar CC-BY-NC / CC-BY-NC-SA weights are **not** supported for deliverable output and are not bundled.

---

## 3. Recorded sounds (CC0 1.0)

The recorded soundtrack layers (ambience beds, foley, and the VCSL / VSCO-2-CE instrument samples) are all **CC0 1.0** public-domain dedications. No audio is committed to this repository: `npm run init` (`tools/samples.py fetch`) downloads each file from the pinned URL in `tools/samples.manifest.json` and checks its sha256. The Freesound recordings are also mirrored, unmodified, as assets of the GitHub release `sounds-v1` for availability (the VCSL / VSCO-2-CE files are fetched from their pinned upstream commits). Every source (author, title, page) is listed in [`SOUND_CREDITS.md`](SOUND_CREDITS.md). `tools/samples.py` refuses any entry whose licence is not CC0-1.0. CC0 requires no attribution; it is given anyway for traceability.
