"""
Build the bundled samples of Entropy FM from their public sources. A developer runs it, never the provider.

    uv run python scripts/fetch_samples.py [--cache folder] [--only felt_piano strings ...]

It clones the sources (shallow, only the folders it needs), picks the notes and velocity layers that the
soundscapes use, trims them, converts them to mono OGG/Vorbis at 44.1 kHz with ffmpeg and writes them to
`entropy_fm/samples/<instrument>/` with an `instrument.json` that maps them (the zones of the sampler).
The root of every zone is measured from the sample itself, so a mislabelled file cannot detune a note.
Sustained instruments get a loop that is made seamless with a cross fade.

Only CC0 and public domain sources are used (see `entropy_fm/samples/LICENSES.md`):

- VSCO 2 Community Edition, Versilian Studios: CC0 1.0 (LICENSE file of the repository).
  https://github.com/sgossner/VSCO-2-CE
- VCSL, Versilian Community Sample Library: CC0 1.0 (LICENSE and README of the repository).
  https://github.com/sgossner/VCSL
- Splendid Grand Piano, released by AKAI: Public Domain (README of the repository).
  https://github.com/sfzinstruments/splendidgrandpiano
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

RATE = 44100
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "entropy_fm" / "samples"
VSCO_URL = "https://github.com/sgossner/VSCO-2-CE"
VCSL_URL = "https://github.com/sgossner/VCSL"
PIANO_URL = "https://github.com/sfzinstruments/splendidgrandpiano"
NOTE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
SOURCES = {"vsco": VSCO_URL, "vcsl": VCSL_URL, "piano": PIANO_URL}


def run(*cmd: str, cwd: Path | None = None) -> None:
    """Run a command and fail loudly."""
    subprocess.run(cmd, check=True, cwd=cwd)  # noqa: S603


def clone(url: str, folder: Path, *sparse: str) -> Path:
    """Clone some folders of a repository, shallow, unless it is there already."""
    if not (folder / ".git").exists():
        run(
            "git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", "-q", url, str(folder)
        )
    run("git", "sparse-checkout", "set", "--no-cone", *[f"/{path}" for path in sparse], cwd=folder)
    return folder


def decode(path: Path) -> np.ndarray:
    """Return a file as mono float samples at 44.1 kHz."""
    raw = subprocess.run(  # noqa: S603
        [  # noqa: S607
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(RATE),
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, dtype="<f4").astype(np.float64)


def write_ogg(path: Path, wave: np.ndarray, quality: int = 3) -> None:
    """Write mono float samples as OGG/Vorbis, 44.1 kHz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(wave, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    subprocess.run(  # noqa: S603
        [  # noqa: S607
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "s16le",
            "-ar",
            str(RATE),
            "-ac",
            "1",
            "-i",
            "-",
            "-c:a",
            "libvorbis",
            "-q:a",
            str(quality),
            "-ar",
            str(RATE),
            str(path),
        ],
        input=pcm,
        check=True,
    )


QUEUE: list[tuple[str, str, np.ndarray, str, int, float, float]] = []


def queue(  # noqa: PLR0913, PLR0917
    folder: str,
    name: str,
    wave: np.ndarray,
    group: str,
    quality: int = 3,
    start: float = 0.0,
    span: float = 0.4,
) -> None:
    """Hold a sample until the whole instrument is known, so that the keys of a layer can be made equally loud."""
    QUEUE.append((folder, name, wave, group, quality, start, span))


def _level(wave: np.ndarray, start: float, span: float) -> float:
    """Return the loudness of a sample: the peak of its attack when it decays, the RMS of its sustain when it holds."""
    piece = wave[int(start * RATE) : int((start + span) * RATE)]
    return float((np.abs(piece).max() if span <= 0.5 else np.sqrt(np.mean(piece**2))) + 1e-9)


def flush() -> None:
    """
    Write the queued samples, each layer of an instrument brought to its own median loudness.

    The loudness is the peak of the attack of a piano or the RMS of the sustain of a string; the samples of one layer of one
    instrument differ by a few dB from key to key (a recording of a real instrument), which this evens out, while the
    difference between the layers (soft and loud) is kept. No sample is scaled to a peak.
    """
    groups: dict[tuple[str, str], list[tuple[str, np.ndarray, int, float, float]]] = {}
    for folder, name, wave, group, quality, start, span in QUEUE:
        groups.setdefault((folder, group), []).append((name, wave, quality, start, span))
    for (folder, _group), items in groups.items():
        levels = [_level(wave, start, span) for _, wave, _, start, span in items]
        target = float(np.median(levels))
        for (name, wave, quality, _, _), level in zip(items, levels, strict=True):
            scaled = wave * min(max(target / level, 0.4), 2.5)
            peak = float(np.abs(scaled).max())
            if peak > 0.95:
                scaled = scaled * (0.95 / peak)
            write_ogg(OUT / folder / name, scaled, quality=quality)
    QUEUE.clear()


def trim(wave: np.ndarray, seconds: float, fade: float = 0.04) -> np.ndarray:
    """Cut a sample to its useful length with a fade out, and drop leading silence."""
    peak = float(np.abs(wave).max()) or 1.0
    start = int(np.argmax(np.abs(wave) > peak * 0.004))
    start = max(start - 8, 0)
    wave = wave[start : start + int(seconds * RATE)].copy()
    count = min(int(fade * RATE), len(wave))
    wave[-count:] *= np.linspace(1.0, 0.0, count)
    return wave


def measure_pitch(wave: np.ndarray) -> float | None:
    """Return the MIDI pitch of the loudest steady part of a sample, by autocorrelation (first strong peak)."""
    peak = int(np.argmax(np.abs(wave)))
    size = 16384
    start = min(peak + 2000, max(len(wave) - size, 0))
    frame = wave[start : start + size]
    if len(frame) < size:
        frame = np.pad(frame, (0, size - len(frame)))
    frame = frame * np.hanning(size)
    spectrum = np.abs(np.fft.rfft(frame, 2 * size)) ** 2
    corr = np.fft.irfft(spectrum)[:size]
    corr = corr / max(float(corr[0]), 1e-12)
    low, high = int(RATE / 2200.0), int(RATE / 25.0)
    window = corr[low:high]
    top = float(window.max())
    if top < 0.2:
        return None
    # the first local maximum that is nearly as strong as the strongest: the true period, not a multiple
    for lag in range(low + 1, high - 1):
        if corr[lag] >= 0.85 * top and corr[lag] >= corr[lag - 1] and corr[lag] >= corr[lag + 1]:
            before, mid, after = corr[lag - 1], corr[lag], corr[lag + 1]
            shift = 0.5 * (before - after) / (before - 2.0 * mid + after)
            return float(69.0 + 12.0 * np.log2(RATE / (lag + shift) / 440.0))
    return None


def note_number(name: str) -> int:
    """Return the MIDI number of a note name such as A#2 (C4 is 60)."""
    match = re.fullmatch(r"([A-G])([#b]?)(-?\d+)", name)
    assert match, name
    shift = {"#": 1, "b": -1, "": 0}[match.group(2)]
    return 12 * (int(match.group(3)) + 1) + NOTE_PC[match.group(1)] + shift


def checked_root(wave: np.ndarray, named: int) -> float:
    """
    Return the measured pitch of a sample, which is the truth: some sets name their octaves differently.

    The measured pitch has to be a whole number of octaves from the name (within 0.9 of a semitone, so that a
    sample that is a little out of tune is mapped as it is); otherwise the name is kept and a note is printed.
    """
    measured = measure_pitch(wave)
    if measured is not None:
        for octave in (-24, -12, 0, 12, 24):
            if abs(measured - (named + octave)) < 0.9:
                return round(measured, 2)
    print(f"  note: measured {measured} against named {named}, keeping the name")
    return float(named)


def make_loop(
    wave: np.ndarray, begin: float, end: float, cross: float = 1.2
) -> tuple[np.ndarray, int, int]:
    """
    Cut a sample near `end` and make [begin, end) loop without a seam, by a cross fade of the tail.

    The cut is the point within half a second of `end` where the last `cross` seconds are most like what comes before
    `begin` (normalised correlation), so that the tone is in phase at the join and the cross fade has no dip.
    """
    a, c = int(begin * RATE), int(cross * RATE)
    lo, hi = int(max(end - 0.5, begin + 2.0) * RATE), min(int((end + 0.5) * RATE), len(wave))
    target = wave[a - c : a]
    best, b = -2.0, min(int(end * RATE), len(wave))
    # slide the window of the tail over the candidate cuts, a thousandth of a second apart
    energy = float(np.sum(target**2)) or 1.0
    for cut in range(lo, hi, 44):
        piece = wave[cut - c : cut]
        denom = (float(np.sum(piece**2)) * energy) ** 0.5 or 1.0
        score = float(np.dot(piece, target)) / denom
        if score > best:
            best, b = score, cut
    body = wave[:b].copy()
    fade = np.linspace(0.0, 1.0, c)
    body[b - c : b] = body[b - c : b] * np.cos(fade * np.pi / 2) + body[a - c : a] * np.sin(
        fade * np.pi / 2
    )
    return body, a, b


def zones_with_layers(items: list[dict[str, object]]) -> list[dict[str, object]]:
    """Give every zone the keys it covers: from half way to its lower neighbour to half way to the upper."""
    by_layer: dict[tuple[int, int], list[dict[str, object]]] = {}
    for item in items:
        by_layer.setdefault((int(item["vlo"]), int(item["vhi"])), []).append(item)  # type: ignore[call-overload]
    for group in by_layer.values():
        group.sort(key=lambda z: float(z["root"]))  # type: ignore[arg-type]
        for index, zone in enumerate(group):
            root = float(zone["root"])  # type: ignore[arg-type]
            zone["lo"] = 0 if index == 0 else int((float(group[index - 1]["root"]) + root) // 2) + 1  # type: ignore[arg-type]
            zone["hi"] = (
                127
                if index == len(group) - 1
                else int((root + float(group[index + 1]["root"])) // 2)
            )  # type: ignore[arg-type]
    return items


def felt_piano(vsco: Path) -> dict[str, object]:
    """The upright piano of VSCO 2 CE at two velocities, C and G of every octave: a soft, felted piano."""  # noqa: D401
    folder = vsco / "Keys" / "Upright Nr1"
    items: list[dict[str, object]] = []
    for layer, (vlo, vhi) in {"pp": (1, 70), "mf": (71, 127)}.items():
        for octave in range(1, 7):
            for pc in ("C", "G"):
                path = folder / f"UR1_{pc}{octave}_{layer}_RR1.wav"
                if not path.exists():
                    continue
                wave = trim(decode(path), 7.0, fade=1.0)
                root = checked_root(wave, note_number(f"{pc}{octave}"))
                name = f"{layer}_{round(root)}.ogg"
                queue("felt_piano", name, wave * 0.85, layer)
                items.append({"file": name, "root": root, "vlo": vlo, "vhi": vhi})
    return {
        "name": "Upright piano, VSCO 2 CE",
        "zones": zones_with_layers(items),
        "release": 0.4,
        "voices": 16,
    }


def strings(vsco: Path) -> dict[str, object]:
    """The violin and cello sections of VSCO 2 CE (sustain, vibrato), each with a seamless loop."""  # noqa: D401
    items: list[dict[str, object]] = []
    sources = (
        ("violins", vsco / "Strings" / "Violin Section" / "susVib", "VlnEns_susVib_{n}_v1.wav"),
        ("cellos", vsco / "Strings" / "Cello Section" / "susvib", "susvib_{n}_v1_1.wav"),
    )
    for tag, folder, pattern in sources:
        for path in sorted(folder.glob("*.wav")):
            if not re.fullmatch(pattern.format(n=r"[A-G]#?\d").replace(".", r"\."), path.name):
                continue
            named = re.search(r"_([A-G]#?\d)_", path.name)
            assert named
            wave = decode(path)
            start = int(np.argmax(np.abs(wave) > np.abs(wave).max() * 0.01))
            wave = wave[max(start - 200, 0) :]
            wave = wave / max(float(np.abs(wave).max()), 1e-6) * 0.7
            root = checked_root(wave[: 4 * RATE], note_number(named.group(1)))
            end = min(len(wave) / RATE - 0.2, 9.5)
            body, loop_a, loop_b = make_loop(wave, 4.2, end)
            fade = min(int(0.2 * RATE), len(body))
            body[-fade:] *= np.linspace(1.0, 0.0, fade)
            name = f"{tag}_{round(root)}.ogg"
            queue("strings", name, body, tag, 3, 4.5, 2.0)
            items.append(
                {
                    "file": name,
                    "root": root,
                    "vlo": 1,
                    "vhi": 127,
                    "loop_start": loop_a,
                    "loop_end": loop_b,
                    "name": tag,
                }
            )
    return {
        "name": "Violin and cello sections, VSCO 2 CE",
        "zones": zones_with_layers_by_name(items),
        "attack": 0.9,
        "release": 1.6,
        "voices": 12,
    }


def zones_with_layers_by_name(items: list[dict[str, object]]) -> list[dict[str, object]]:
    """Like `zones_with_layers`, but the zones are mapped separately for each named group."""
    out: list[dict[str, object]] = []
    for group in sorted({str(item["name"]) for item in items}):
        out.extend(zones_with_layers([item for item in items if item["name"] == group]))
    return out


def electric_piano(vcsl: Path) -> dict[str, object]:
    """The FM electric piano that the VCSL sampled from a Yamaha TX81Z, three velocities."""  # noqa: D401
    folder = vcsl / "Electrophones" / "TX81Z" / "FM Piano"
    items: list[dict[str, object]] = []
    for layer, (vlo, vhi) in {"vl1": (1, 60), "vl2": (61, 100), "vl3": (101, 127)}.items():
        for path in sorted(folder.glob(f"FMPiano_*_{layer}.wav")):
            named = re.search(r"FMPiano_([A-G]#?)(\d)_", path.name)
            assert named
            number = note_number(f"{named.group(1)}{int(named.group(2))}")
            if not 36 <= number <= 84 or (number % 4 != 0 and number % 12 not in (0, 4, 8)):
                continue
            wave = trim(decode(path), 5.0, fade=0.6)
            root = checked_root(wave, number)
            name = f"{layer}_{round(root)}.ogg"
            queue("ep", name, wave * 0.85, layer)
            items.append({"file": name, "root": root, "vlo": vlo, "vhi": vhi})
    return {
        "name": "FM electric piano, VCSL (Yamaha TX81Z)",
        "zones": zones_with_layers(items),
        "release": 0.5,
        "voices": 12,
    }


def pluck(vcsl: Path) -> dict[str, object]:
    """A strumstick, plucked with the finger: a soft, guitar-like voice."""  # noqa: D401
    folder = vcsl / "Chordophones" / "Composite Chordophones" / "Strumstick" / "Finger"
    items: list[dict[str, object]] = []
    for path in sorted(folder.glob("*_vl2_rr1.wav")):
        named = re.search(r"_([A-G]#?)(\d)_vl", path.name)
        assert named
        wave = trim(decode(path), 3.0, fade=0.4)
        root = checked_root(wave, note_number(f"{named.group(1)}{named.group(2)}"))
        name = f"p_{len(items)}_{round(root)}.ogg"
        queue("pluck", name, wave * 0.85, "all")
        items.append({"file": name, "root": root, "vlo": 1, "vhi": 127})
    return {
        "name": "Strumstick, VCSL",
        "zones": zones_with_layers(items),
        "release": 0.3,
        "voices": 8,
    }


def drums(vcsl: Path) -> dict[str, object]:
    """One-shots of the VCSL: a soft bass drum, a snare with and without wires, a stick click, claves."""
    base = vcsl / "Membranophones" / "Struck Membranophones"
    plan = {
        "kick": (
            base / "Bass Drum 2",
            ["bassdrum_hit_pp1", "bassdrum_hit_mp1", "bassdrum_hit_mf1"],
            0.5,
        ),
        "snare": (
            base / "Snare Drum, Modern 2",
            ["Snare3M_HitSN_v2_rr1_Mid", "Snare3M_HitSN_v3_rr1_Mid", "Snare3M_HitSN_v4_rr1_Mid"],
            0.45,
        ),
        "rim": (
            base / "Snare Drum, Modern 2",
            ["Snare3M_Xstick_v2_rr1_Mid", "Snare3M_stick_v3_rr1_Mid"],
            0.25,
        ),
        "click": (
            vcsl / "Idiophones" / "Struck Idiophones" / "Claves",
            ["Claves1_Hit_v1_rr1_Mid", "Claves1_Hit_v2_rr1_Mid", "Claves2_Hit_v1_rr1_Mid"],
            0.2,
        ),
    }
    shots: dict[str, list[dict[str, object]]] = {}
    for group, (folder, names, seconds) in plan.items():
        for name in names:
            wave = trim(decode(folder / f"{name}.wav"), seconds, fade=0.05)
            wave = wave * (0.9 / max(float(np.abs(wave).max()), 1e-6))
            write_ogg(OUT / "drums" / f"{name}.ogg", wave, quality=4)
            shots.setdefault(group, []).append({"file": f"{name}.ogg", "name": name})
    return {"name": "Drum one-shots, VCSL", "shots": shots}


def _levelled(wave: np.ndarray, peak: float) -> np.ndarray:
    """Return a wave scaled so that its highest sample is `peak`."""
    return wave * (peak / max(float(np.abs(wave).max()), 1e-6))


def slit_drum(vcsl: Path) -> dict[str, object]:
    """
    Make the slit drum (log drum) of the VCSL: its two tongues, three velocities each, a short ring.

    There are only two pitches, a minor sixth apart (about 108 and 156 Hz); a note of the scale is the nearer
    tongue read a few semitones up or down by the sampler. The layers are brought to peaks of 0.45, 0.65 and 0.9,
    a gentler spread than the recording's.
    """
    folder = vcsl / "Idiophones" / "Struck Idiophones" / "Slit Drum"
    items: list[dict[str, object]] = []
    layers = {"v1": (1, 50, 0.45), "v2": (51, 92, 0.65), "v3": (93, 127, 0.9)}
    for tongue in ("Hi", "Lo"):
        for layer, (vlo, vhi, peak) in layers.items():
            wave = trim(decode(folder / f"LogDrum{tongue}_MedM_{layer}_rr1_Sum.wav"), 0.8, fade=0.3)
            root = checked_root(wave, 51 if tongue == "Hi" else 45)
            name = f"{tongue.lower()}_{layer}.ogg"
            write_ogg(OUT / "slit_drum" / name, _levelled(wave, peak), quality=3)
            items.append({"file": name, "root": root, "vlo": vlo, "vhi": vhi})
    return {
        "name": "Slit drum, VCSL",
        "zones": sorted(items, key=lambda i: (i["root"], i["vlo"])),  # type: ignore[arg-type,return-value]
        "release": 0.25,
        "voices": 6,
    }


def woodblock(vcsl: Path) -> dict[str, object]:
    """A woodblock of the VCSL (one pitch, about 1.37 kHz), three velocities, trimmed to its short ring."""  # noqa: D401
    folder = vcsl / "Idiophones" / "Struck Idiophones" / "Woodblock"
    items: list[dict[str, object]] = []
    layers = {
        "pp": ("wood_click_pp_rr1", 1, 50, 0.45),
        "mp": ("wood_click_mp", 51, 92, 0.65),
        "f": ("wood_click3_vl2", 93, 127, 0.9),
    }
    for layer, (source, vlo, vhi, peak) in layers.items():
        wave = trim(decode(folder / f"{source}.wav"), 0.35, fade=0.12)
        root = checked_root(wave, 89)
        name = f"block_{layer}.ogg"
        write_ogg(OUT / "woodblock" / name, _levelled(wave, peak), quality=3)
        items.append({"file": name, "root": root, "vlo": vlo, "vhi": vhi})
    return {"name": "Woodblock, VCSL", "zones": items, "release": 0.15, "voices": 6}


def grand_piano(piano: Path) -> dict[str, object]:
    """Splendid Grand Piano (public domain): two velocity layers, a note every three semitones."""
    folder = piano / "Samples"
    items: list[dict[str, object]] = []
    for layer, (vlo, vhi) in {"PP": (1, 72), "FF": (73, 127)}.items():
        have = {note_number(p.stem.split(" ", 1)[1]): p for p in folder.glob(f"{layer} *.flac")}
        chosen: list[int] = []
        for target in range(40, 97, 3):
            near = min(have, key=lambda n: (abs(n - target), n))
            if abs(near - target) <= 2 and near not in chosen:
                chosen.append(near)
        for root in sorted(chosen):
            wave = trim(decode(have[root]), 5.0, fade=0.8)
            heard = checked_root(wave, root)
            name = f"{layer.lower()}_{root}.ogg"
            queue("grand_piano", name, wave * 0.8, layer)
            items.append({"file": name, "root": heard, "vlo": vlo, "vhi": vhi})
    return {
        "name": "Splendid Grand Piano",
        "zones": zones_with_layers(items),
        "release": 0.5,
        "voices": 16,
    }


LICENSE_TEXT = """# Licenses of the bundled samples

Every sample that Entropy FM bundles is CC0 1.0 or in the public domain, taken from the sources named
here, whose own license files and READMEs were read at the original repository (not at a mirror).
Nothing is under a non-commercial, attribution-only or share-alike license. `scripts/fetch_samples.py`
rebuilds all of it; the files are trimmed, converted to mono OGG/Vorbis and (for strings) given a
seamless loop, which are the only changes. `entropy_fm/samples/<instrument>/instrument.json` maps them.

| folder | what it is | source | author | license |
|---|---|---|---|---|
| felt_piano/ | upright piano ("Upright Nr1"), soft and medium layers, C and G of each octave | VSCO 2 Community Edition, Keys/Upright Nr1, {vsco} | Versilian Studios (Sam Gossner, Simon Dalzell) | CC0 1.0, `LICENSE` of the repository |
| strings/ | violin section and cello section, sustained with vibrato, looped | VSCO 2 Community Edition, Strings, {vsco} | Versilian Studios | CC0 1.0, `LICENSE` of the repository |
| ep/ | FM electric piano sampled from a Yamaha TX81Z, three velocity layers | VCSL, Electrophones/TX81Z/FM Piano, {vcsl} | Versilian Studios (patches made with TX81Z Programmer, sampled from the hardware) | CC0 1.0, `LICENSE` and `README` of the repository |
| pluck/ | strumstick played with the finger | VCSL, Chordophones/Composite Chordophones/Strumstick, {vcsl} | Versilian Studios | CC0 1.0 |
| drums/ | soft bass drum, snare, cross stick, claves (one-shots) | VCSL, Membranophones/Struck Membranophones and Idiophones/Struck Idiophones, {vcsl} | Versilian Studios | CC0 1.0 |
| slit_drum/ | slit (log) drum, both tongues, three velocities each | VCSL, Idiophones/Struck Idiophones/Slit Drum, {vcsl} | Versilian Studios | CC0 1.0, `LICENSE` and `README` of the repository |
| woodblock/ | woodblock, three velocities | VCSL, Idiophones/Struck Idiophones/Woodblock, {vcsl} | Versilian Studios | CC0 1.0, `LICENSE` and `README` of the repository |
| grand_piano/ | Splendid Grand Piano, two velocity layers, a note every three semitones | {piano} | AKAI (Steinway samples released in the early 2000s), mapped by kinwie | Public Domain (README of the repository: "Public Domain samples by AKAI") |

Everything else in Entropy FM (oscillators, drums that are not listed, noise, vinyl and tape) is made with
numpy and is original work of this project (Apache-2.0).

## Looked at and not used, and why

- **Mellotron samples** (the `archiveorg-mellotron` set that smpldsnds converted from a Mellotron
  CD-ROM archive on archive.org): the mirror has no license statement and archive.org could not be
  reached from the build sandbox, so no license could be verified. Not bundled. The Mellotron-like
  strings and choir are synthesised instead.
- **jRhodes3d**, and the "Electric Piano 1" of the Discord GM bank that is made from it: the sources
  disagree (CC0 and CC-BY-NC). Not bundled.
- **Greg Sullivan's E-Pianos** (CP80, Pianet, Wurlitzer): the Discord GM bank labels them CC0, but the
  original repository says CC-BY 3.0 (attribution). Not bundled; the GM bank's per-instrument license
  lines are not reliable.
- **Salamander Grand Piano**: CC-BY. Not bundled.
- **Sonic Pi samples** (freesound uploads): the CC0 statement is the Sonic Pi project's, and the
  freesound pages could not be checked. Not bundled; the earlier experiment's drums were removed.
- Hydrogen drum kits (GPL), Drum Abuse packs and Retro Drums (no license): not bundled.
"""


def main() -> None:
    """Rebuild the samples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=None, help="where the sources are cloned")
    parser.add_argument(
        "--only",
        nargs="*",
        default=[
            "felt_piano",
            "strings",
            "ep",
            "pluck",
            "drums",
            "slit_drum",
            "woodblock",
            "grand_piano",
        ],
    )
    args = parser.parse_args()
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg is needed")
    cache = args.cache or Path(tempfile.mkdtemp(prefix="entropy-samples-"))
    cache.mkdir(parents=True, exist_ok=True)
    vsco = vcsl = piano = None
    if {"felt_piano", "strings"} & set(args.only):
        vsco = clone(
            VSCO_URL,
            cache / "vsco",
            "Keys/Upright Nr1",
            "Strings/Violin Section/susVib",
            "Strings/Cello Section/susvib",
            "LICENSE",
        )
    if {"ep", "pluck", "drums", "slit_drum", "woodblock"} & set(args.only):
        vcsl = clone(
            VCSL_URL,
            cache / "vcsl",
            "Electrophones/TX81Z/FM Piano",
            "Chordophones/Composite Chordophones/Strumstick/Finger",
            "Membranophones/Struck Membranophones/Bass Drum 2",
            "Membranophones/Struck Membranophones/Snare Drum, Modern 2",
            "Idiophones/Struck Idiophones/Claves",
            "Idiophones/Struck Idiophones/Slit Drum",
            "Idiophones/Struck Idiophones/Woodblock",
            "LICENSE",
        )
    if "grand_piano" in args.only:
        piano = clone(PIANO_URL, cache / "piano", "Samples")
    builders = {
        "felt_piano": lambda: felt_piano(vsco),  # type: ignore[arg-type]
        "strings": lambda: strings(vsco),  # type: ignore[arg-type]
        "ep": lambda: electric_piano(vcsl),  # type: ignore[arg-type]
        "pluck": lambda: pluck(vcsl),  # type: ignore[arg-type]
        "drums": lambda: drums(vcsl),  # type: ignore[arg-type]
        "slit_drum": lambda: slit_drum(vcsl),  # type: ignore[arg-type]
        "woodblock": lambda: woodblock(vcsl),  # type: ignore[arg-type]
        "grand_piano": lambda: grand_piano(piano),  # type: ignore[arg-type]
    }
    for name in args.only:
        folder = OUT / name
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        (folder / "instrument.json").write_text(json.dumps(builders[name](), indent=1))
        flush()
        print("wrote", name)
    (OUT / "LICENSES.md").write_text(
        LICENSE_TEXT.format(vsco=VSCO_URL, vcsl=VCSL_URL, piano=PIANO_URL)
    )
    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"{total / 1e6:.2f} MB in {OUT}")


if __name__ == "__main__":
    main()
