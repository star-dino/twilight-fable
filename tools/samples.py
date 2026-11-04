#!/usr/bin/env python3
"""Recorded-sample fetcher for Twilight Fable (`tools/samples.py`, `npm run init`).

The repository never contains audio (`.gitignore` drops *.wav/*.ogg/...). The soundtrack is built from
CC0 recordings that are downloaded on demand from pinned links listed in `tools/samples.manifest.json`:

  * `sounds`       -- Freesound CC0 recordings (ambience beds, foley). `url` is the public HQ preview
                      (cdn.freesound.org, no API key needed); `mirror` is an optional second URL
                      (e.g. a GitHub Release asset on your own account) that is tried when `url` fails,
                      or first with `--prefer-mirror`.
  * `instruments`  -- note-by-note CC0 instrument samples from VCSL / VSCO-2-CE, pinned to a git commit.

Commands:
  fetch   [--only ID] [--prefer-mirror]   download, sha256-verify, decode to 48 kHz stereo, cut, normalise
  verify                                  re-check cached raw files against their hashes
  list                                    what is in the manifest / cache
  pin                                     fill in missing sha256 values (maintainer tool)
  credits                                 regenerate SOUND_CREDITS.md from the manifest
  bundle <dir> [--mirror-base URL]        copy the Freesound raw files + manifest into <dir>, ready to upload as
                                          release assets; with --mirror-base the written manifest points `mirror`
                                          at <URL>/<file>

Everything lands in `../out/samples-cache/twilight-fable/` (override: $STYLEHOP_SAMPLES_CACHE):

  raw/<id>.<ext>        verified downloads, untouched
  wav/<id>[__n].wav     48 kHz stereo s16 windows / cuts, what `sound.py` reads
  index.json            id -> file + mixing metadata

`tools/sound.py` falls back to pure synthesis when the cache is missing, so a fresh clone still renders.
Only CC0-1.0 entries are accepted; anything else aborts the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from scipy.io import wavfile

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = Path(__file__).resolve().parent / "samples.manifest.json"
CACHE = Path(os.environ.get("STYLEHOP_SAMPLES_CACHE") or (REPO_ROOT.parent / "out" / "samples-cache" / "twilight-fable")).resolve()
RAW = CACHE / "raw"
WAV = CACHE / "wav"
INDEX = CACHE / "index.json"
CREDITS = REPO_ROOT / "SOUND_CREDITS.md"
SR = 48000
ALLOWED_LICENSE = "CC0-1.0"


# --------------------------------------------------------------------------------------------------
# manifest helpers
# --------------------------------------------------------------------------------------------------


def load_manifest() -> dict:
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for s in m.get("sounds", []):
        if s.get("license") != ALLOWED_LICENSE:
            raise SystemExit(f"samples.py: {s['id']} is {s.get('license')!r}; only {ALLOWED_LICENSE} is allowed")
    for ins in m.get("instruments", []):
        if ins.get("license") != ALLOWED_LICENSE:
            raise SystemExit(f"samples.py: instrument {ins['id']} is {ins.get('license')!r}; only {ALLOWED_LICENSE} is allowed")
    return m


def save_manifest(m: dict) -> None:
    MANIFEST.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def iter_items(m: dict):
    """Yield (id, entry, source urls, ext, extra mixing metadata) for every downloadable file."""
    for s in m.get("sounds", []):
        ext = Path(urllib.parse.urlparse(s["url"]).path).suffix or ".ogg"
        yield s["id"], s, s, ext, {}
    for ins in m.get("instruments", []):
        base = ins["base"].rstrip("/")
        for f in ins["files"]:
            url = base + "/" + urllib.parse.quote(f["path"])
            yield f["id"], f, {"url": url, "mirror": f.get("mirror")}, ".wav", {
                "instrument": ins["id"], "uses": ["inst." + ins["id"]], "kind": "inst", "gain": 1.0}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    ok = False
    if shutil.which("curl"):
        rc = subprocess.run(
            ["curl", "-sSL", "--fail", "--retry", "2", "--max-time", "300", "-o", str(tmp), url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        ok = rc == 0 and tmp.is_file() and tmp.stat().st_size > 0
    if not ok:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "stylehop-samples/1"})
            with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as fh:
                shutil.copyfileobj(resp, fh)
            ok = tmp.stat().st_size > 0
        except Exception:
            ok = False
    if ok:
        tmp.replace(dest)
    elif tmp.exists():
        tmp.unlink()
    return ok


def fetch_raw(sid: str, ext: str, src: dict, prefer_mirror: bool, want_sha: str | None) -> Path | None:
    dest = RAW / f"{sid}{ext}"
    if dest.is_file() and (not want_sha or sha256_file(dest) == want_sha):
        return dest
    urls = [u for u in (src.get("url"), src.get("mirror")) if u]
    if prefer_mirror:
        urls.reverse()
    for u in urls:
        if download(u, dest):
            got = sha256_file(dest)
            if want_sha and got != want_sha:
                print(f"[sha] {sid}: {u} hash mismatch (got {got[:12]}.., want {want_sha[:12]}..)", file=sys.stderr)
                dest.unlink()
                continue
            return dest
    return None


# --------------------------------------------------------------------------------------------------
# decode / cut / normalise
# --------------------------------------------------------------------------------------------------


def decode_stereo(src: Path) -> np.ndarray:
    """Any audio file -> float32 (n, 2) at 48 kHz via ffmpeg."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(src), "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"],
        stdout=subprocess.PIPE,
        check=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.float32).reshape(-1, 2).copy()


def shape_edges(x: np.ndarray, fade_in: float, fade_out: float) -> np.ndarray:
    n = len(x)
    fi, fo = min(n // 2, int(fade_in * SR)), min(n // 2, int(fade_out * SR))
    if fi > 1:
        x[:fi] *= (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fi)))[:, None]
    if fo > 1:
        x[-fo:] *= (0.5 + 0.5 * np.cos(np.linspace(0, np.pi, fo)))[:, None]
    return x


def normalise(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "rms":  # ambience: steady level, -24 dBFS RMS, peak-safe
        rms = float(np.sqrt(np.mean(x**2)) + 1e-9)
        g = (10 ** (-24 / 20)) / rms
        g = min(g, 0.95 / (float(np.max(np.abs(x))) + 1e-9))
        return x * g
    peak = float(np.max(np.abs(x)) + 1e-9)  # one-shots: peak 0.9
    return x * (0.9 / peak)


def write_wav(path: Path, x: np.ndarray) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), SR, np.clip(x * 32767.0, -32768, 32767).astype(np.int16))
    return len(x) / SR


def make_wavs(sid: str, entry: dict, raw: Path) -> list[dict]:
    """Produce the cache wavs for one source; return index rows."""
    audio = decode_stereo(raw)
    mode = entry.get("normalize") or ("rms" if entry.get("kind") == "bed" else "peak")
    cuts = entry.get("cuts")
    rows = []
    meta = {k: entry[k] for k in ("uses", "gain", "lp", "hp", "note", "vel", "instrument", "kind", "pan") if k in entry}
    if cuts:
        for n, (t0, t1) in enumerate(cuts):
            seg = audio[int(t0 * SR) : int(t1 * SR)].copy()
            if len(seg) < 64:
                continue
            seg = normalise(shape_edges(seg, 0.004, 0.040), mode)
            wid = f"{sid}__{n}"
            dur = write_wav(WAV / f"{wid}.wav", seg)
            rows.append({"id": wid, "src": sid, "file": f"wav/{wid}.wav", "dur": round(dur, 3), **meta})
    else:
        if entry.get("window"):
            t0, t1 = entry["window"]
            audio = audio[int(t0 * SR) : int(t1 * SR)].copy()
        fade = (0.05, 0.05) if mode == "rms" else (0.004, 0.06)
        audio = normalise(shape_edges(audio, *fade), mode)
        dur = write_wav(WAV / f"{sid}.wav", audio)
        rows.append({"id": sid, "src": sid, "file": f"wav/{sid}.wav", "dur": round(dur, 3), **meta})
    return rows


# --------------------------------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------------------------------


def cmd_fetch(args) -> int:
    m = load_manifest()
    if not shutil.which("ffmpeg"):
        raise SystemExit("samples.py: ffmpeg is required")
    index = {"version": 2, "samples": {}}
    failed, total = [], 0
    for sid, entry, src, ext, extra in iter_items(m):
        if args.only and not (sid == args.only or sid.startswith(args.only)):
            continue
        total += 1
        raw = fetch_raw(sid, ext, src, args.prefer_mirror, entry.get("sha256") or None)
        if raw is None:
            failed.append(sid)
            print(f"[fail] {sid}", file=sys.stderr)
            continue
        for row in make_wavs(sid, {**entry, **extra}, raw):
            index["samples"][row["id"]] = row
    if args.only and INDEX.is_file():  # partial refresh: merge
        old = json.loads(INDEX.read_text(encoding="utf-8"))
        old["samples"].update(index["samples"])
        index = old
    index["generated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    INDEX.write_text(json.dumps(index, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"samples: {total - len(failed)}/{total} sources ready, {len(index['samples'])} cached files -> {CACHE}")
    if failed:
        print(f"samples: {len(failed)} failed ({', '.join(failed[:8])}...); sound.py will fall back for the missing groups", file=sys.stderr)
        return 1
    return 0


def cmd_verify(_args) -> int:
    m = load_manifest()
    bad = 0
    for sid, entry, _src, ext, _x in iter_items(m):
        p = RAW / f"{sid}{ext}"
        if not p.is_file():
            print(f"[missing] {sid}")
            bad += 1
        elif entry.get("sha256") and sha256_file(p) != entry["sha256"]:
            print(f"[hash] {sid}")
            bad += 1
    print("verify: ok" if not bad else f"verify: {bad} problem(s)")
    return 1 if bad else 0


def cmd_list(_args) -> int:
    m = load_manifest()
    have = json.loads(INDEX.read_text(encoding="utf-8"))["samples"] if INDEX.is_file() else {}
    for s in m.get("sounds", []):
        n = sum(1 for k in have if have[k].get("src") == s["id"])
        print(f"{s['id']:<14} {s['license']}  {s['author']:<22} {','.join(s.get('uses', [])):<28} cached={n}")
    for ins in m.get("instruments", []):
        n = sum(1 for f in ins["files"] if f["id"] in have)
        print(f"{ins['id']:<14} {ins['license']}  {ins['author']:<22} {len(ins['files'])} notes cached={n}")
    return 0


def cmd_pin(_args) -> int:
    m = load_manifest()
    changed = 0
    for sid, entry, src, ext, _x in iter_items(m):
        if entry.get("sha256"):
            continue
        raw = fetch_raw(sid, ext, src, False, None)
        if raw is None:
            print(f"[pin-fail] {sid}", file=sys.stderr)
            continue
        entry["sha256"] = sha256_file(raw)
        changed += 1
    save_manifest(m)
    print(f"pin: {changed} hash(es) written")
    return 0


def cmd_credits(_args) -> int:
    m = load_manifest()
    L = [
        "# Sound credits",
        "",
        "Every sound used by the recorded soundtrack is **CC0 1.0** (public domain dedication): no attribution is required,",
        "no NC/ND/BY terms apply, and the videos may be monetised. This list is generated from `tools/samples.manifest.json`",
        "(`python3 tools/samples.py credits`); it exists so each sound stays traceable to its source page.",
        "",
        "No audio is committed to the repository: `npm run init` downloads the files into `../out/samples-cache/twilight-fable/`.",
        "",
        "## Freesound recordings (CC0 1.0)",
        "",
        "| id | author | title | used for | source |",
        "| --- | --- | --- | --- | --- |",
    ]
    for s in m.get("sounds", []):
        L.append(f"| {s['id']} | {s['author']} | {s['title']} | {', '.join(s.get('uses', []))} | {s['source']} |")
    L += ["", "## Instrument samples (CC0 1.0)", ""]
    for ins in m.get("instruments", []):
        L.append(f"- **{ins['title']}** ({len(ins['files'])} files, `{ins['id']}`): {ins['author']} -- {ins['source']} @ `{ins['commit'][:10]}`")
    L.append("")
    CREDITS.write_text("\n".join(L), encoding="utf-8")
    print(f"credits: wrote {CREDITS.name}")
    return 0


def cmd_bundle(args) -> int:
    """Copy the Freesound raw files + manifest into a folder to upload as release assets."""
    m = load_manifest()
    out = Path(args.dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    made = 0
    for s in m.get("sounds", []):
        ext = Path(urllib.parse.urlparse(s["url"]).path).suffix or ".ogg"
        name = f"{s['id']}{ext}"
        raw = fetch_raw(s["id"], ext, s, False, s.get("sha256") or None)
        if raw is None:
            print(f"[bundle-fail] {s['id']}", file=sys.stderr)
            continue
        shutil.copyfile(raw, out / name)
        if args.mirror_base:
            s["mirror"] = args.mirror_base.rstrip("/") + "/" + name
        made += 1
    (out / "samples.manifest.json").write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"bundle: {made} file(s) + samples.manifest.json -> {out}")
    if not args.mirror_base:
        print("bundle: upload the files as release assets, then re-run with --mirror-base <release download URL>, "
              "and copy the written manifest over tools/samples.manifest.json")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--only")
    f.add_argument("--prefer-mirror", action="store_true")
    f.set_defaults(fn=cmd_fetch)
    sub.add_parser("verify").set_defaults(fn=cmd_verify)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    sub.add_parser("pin").set_defaults(fn=cmd_pin)
    sub.add_parser("credits").set_defaults(fn=cmd_credits)
    b = sub.add_parser("bundle")
    b.add_argument("dir")
    b.add_argument("--mirror-base")
    b.set_defaults(fn=cmd_bundle)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
