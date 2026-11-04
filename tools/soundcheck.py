#!/usr/bin/env python3
"""Sound regression check for Twilight Fable (`tools/soundcheck.py`, `npm run soundcheck`).

Re-renders the stored cue sheets in `styles/twilight-fable/sound_cues/` through `tools/sound.py` (recorded CC0 layers
from `npm run init`), measures the mix and compares it with `styles/twilight-fable/sound_baseline.json`:

  * per-layer pre-master RMS (music / beds / foley, dBFS)
  * bed level around a reveal (before / at / after, dBFS) when the cue sheet has one
  * integrated loudness (LUFS) and true peak (dBTP) after the same two-pass linear `loudnorm` that `render.mjs`
    puts on the video (target -19 LUFS / -1.5 dBTP, no voice)

Exit codes: 0 ok, 1 drift beyond tolerance, 2 cannot run (no fetched samples, missing ffmpeg / baseline).

  python3 tools/soundcheck.py              compare every case with the baseline
  python3 tools/soundcheck.py --only mirror-lake-cat
  python3 tools/soundcheck.py --accept     overwrite the baseline with the current numbers (intentional change)
  python3 tools/soundcheck.py --add NAME CUES.json   register a new case (copies the cue sheet, then accepts it)

A cue sheet is `SOUND_DUMP_CUES=<file> node tools/render.mjs ... --video ...` (what `render.mjs` hands to `sound.py`).
The WAV sha256 is recorded for information; only the numbers above decide pass / fail, so a different
numpy / scipy build that is not bit-identical does not fail the check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

STYLE_DIR = REPO / "styles" / "twilight-fable"
BASELINE = STYLE_DIR / "sound_baseline.json"
CUES_DIR = STYLE_DIR / "sound_cues"

TOLERANCE = {"layer_db": 1.0, "reveal_db": 1.0, "lufs": 1.0, "true_peak_db": 1.0}
TARGET_I, TARGET_TP, TARGET_LRA = -19.0, -1.5, 20.0  # same as render.mjs without a voice track


def _loudnorm(wav: Path, extra: str = "") -> dict:
    af = f"loudnorm=I={TARGET_I}:TP={TARGET_TP}:LRA={TARGET_LRA}{extra}:print_format=json"
    p = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(wav), "-af", af, "-f", "null", "-"],
                       capture_output=True, text=True)
    m = re.search(r"\{\s*\"input_i\"[\s\S]*?\}", p.stderr)
    if not m:
        raise SystemExit("soundcheck: ffmpeg loudnorm gave no statistics:\n" + p.stderr[-400:])
    return json.loads(m.group(0))


def measure_loudness(wav: Path, tmp: Path) -> dict:
    """Two-pass linear loudnorm exactly as render.mjs builds it; returns LUFS / dBTP of the normalised result."""
    st = _loudnorm(wav)
    extra = (f":measured_I={st['input_i']}:measured_TP={st['input_tp']}:measured_LRA={st['input_lra']}"
             f":measured_thresh={st['input_thresh']}:offset={st['target_offset']}:linear=true")
    out = tmp / "norm.wav"
    af = f"loudnorm=I={TARGET_I}:TP={TARGET_TP}:LRA={TARGET_LRA}{extra}"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav), "-af", af,
                    "-ar", "48000", "-c:a", "pcm_s16le", str(out)], check=True)
    after = _loudnorm(out)
    return {"lufs_pre": round(float(st["input_i"]), 2), "true_peak_pre": round(float(st["input_tp"]), 2),
            "lufs": round(float(after["input_i"]), 2), "true_peak": round(float(after["input_tp"]), 2)}


def render_case(cues_path: Path) -> dict:
    import sound  # tools/sound.py

    cue_data = json.loads(cues_path.read_text(encoding="utf-8"))
    preset = sound.resolve_preset(cue_data.get("soundPreset") or "twilight")
    report: dict = {}
    stereo = sound.synthesize_from_cues(cue_data, preset=preset, samples="on", report=report)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        wav = tmp / "mix.wav"
        sound.write_wav(wav, stereo)
        sha = hashlib.sha256(wav.read_bytes()).hexdigest()
        loud = measure_loudness(wav, tmp)
    m = {
        "dur": cue_data.get("dur"),
        "layers_rms_dbfs": report["layers_rms_dbfs"],
        "reveals": report.get("reveals", []),
        "loudness": loud,
        "wav_sha256": sha,
    }
    ar = report.get("around_reveal")
    if ar:
        m["around_reveal"] = {k: ar[k] for k in ("before_rms_dbfs", "at_rms_dbfs", "after_rms_dbfs")}
    return m


def compare(name: str, base: dict, cur: dict, tol: dict) -> list[str]:
    bad = []

    def chk(label, b, c, t):
        if b is None or c is None:
            if b != c:
                bad.append(f"{name}: {label} baseline={b} current={c}")
            return
        if abs(c - b) > t:
            bad.append(f"{name}: {label} {b:+.2f} -> {c:+.2f} (delta {c - b:+.2f}, tol {t})")

    for k, b in base["layers_rms_dbfs"].items():
        chk(f"layer {k} RMS dBFS", b, cur["layers_rms_dbfs"].get(k), tol["layer_db"])
    chk("LUFS", base["loudness"]["lufs"], cur["loudness"]["lufs"], tol["lufs"])
    chk("true peak dBTP", base["loudness"]["true_peak"], cur["loudness"]["true_peak"], tol["true_peak_db"])
    if "around_reveal" in base or "around_reveal" in cur:
        b, c = base.get("around_reveal") or {}, cur.get("around_reveal") or {}
        if not b or not c:
            bad.append(f"{name}: reveal presence changed (baseline={bool(b)}, current={bool(c)})")
        else:
            for k in ("before_rms_dbfs", "at_rms_dbfs", "after_rms_dbfs"):
                chk(f"reveal bed {k.split('_')[0]}", b[k], c[k], tol["reveal_db"])
    if len(base.get("reveals", [])) != len(cur.get("reveals", [])):
        bad.append(f"{name}: reveal count {len(base.get('reveals', []))} -> {len(cur.get('reveals', []))}")
    return bad


def fmt(m: dict) -> str:
    L = m["layers_rms_dbfs"]
    s = f"music {L['music']:+.2f}  beds {L['beds']:+.2f}  foley {L['foley']:+.2f} dBFS | {m['loudness']['lufs']:+.2f} LUFS  {m['loudness']['true_peak']:+.2f} dBTP"
    ar = m.get("around_reveal")
    if ar:
        s += f" | reveal bed {ar['before_rms_dbfs']:+.1f} -> {ar['at_rms_dbfs']:+.1f} -> {ar['after_rms_dbfs']:+.1f} dBFS"
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--accept", action="store_true", help="write the current numbers to sound_baseline.json")
    ap.add_argument("--only", default=None, help="limit to one case name")
    ap.add_argument("--add", nargs=2, metavar=("NAME", "CUES_JSON"), help="register a new case from a cue sheet")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        print("soundcheck: ffmpeg is required", file=sys.stderr)
        return 2
    from sound_samples import Bank
    if not Bank().usable:
        print("soundcheck: no fetched samples; run `npm run init` first", file=sys.stderr)
        return 2

    base = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.is_file() else {"version": 1, "tolerance": TOLERANCE, "cases": {}}
    if args.add:
        name, src = args.add
        CUES_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, CUES_DIR / f"{name}.cues.json")
        base["cases"].setdefault(name, {"cues": f"sound_cues/{name}.cues.json"})
        args.accept, args.only = True, name
    if not base["cases"]:
        print("soundcheck: baseline has no cases (use --add NAME CUES.json)", file=sys.stderr)
        return 2
    tol = {**TOLERANCE, **base.get("tolerance", {})}

    failures: list[str] = []
    for name, case in base["cases"].items():
        if args.only and name != args.only:
            continue
        cues_path = STYLE_DIR / case["cues"]
        if not cues_path.is_file():
            print(f"soundcheck: missing cue sheet {cues_path}", file=sys.stderr)
            return 2
        cur = render_case(cues_path)
        print(f"{name:22s} {fmt(cur)}")
        if args.accept:
            base["cases"][name] = {"cues": case["cues"], **cur}
            continue
        if "layers_rms_dbfs" not in case:
            failures.append(f"{name}: no baseline numbers (run --accept)")
            continue
        bad = compare(name, case, cur, tol)
        failures += bad
        same = cur["wav_sha256"] == case.get("wav_sha256")
        print(f"{'':22s} {'DRIFT' if bad else 'ok'}  (wav sha256 {'identical' if same else 'differs, numbers within tolerance' if not bad else 'differs'})")

    if args.accept:
        base["tolerance"] = tol
        base["note"] = ("Per-layer numbers are pre-master dBFS from sound.py --report; loudness is after the render.mjs two-pass "
                        "linear loudnorm (-19 LUFS / -1.5 dBTP target). Regenerate with `python3 tools/soundcheck.py --accept`.")
        BASELINE.write_text(json.dumps(base, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"soundcheck: baseline updated -> {BASELINE.relative_to(REPO)}")
        return 0
    if failures:
        print("\nsoundcheck: DRIFT", file=sys.stderr)
        for f in failures:
            print("  " + f, file=sys.stderr)
        print("If the change is intentional, listen to it, then run `python3 tools/soundcheck.py --accept`.", file=sys.stderr)
        return 1
    print("soundcheck: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
