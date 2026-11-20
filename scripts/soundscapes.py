"""
Render the soundscapes of Entropy FM and write what is needed to judge them without listening.

    uv run python scripts/soundscapes.py [output folder] [--fast] [--only excerpts parts voices full notes condensed]

For each of the five styles, on the anomaly day of the demo house and on the kitchen of the real house:

- `entropy_fm_<style>_demo_excerpts.mp3` and `entropy_fm_<style>_kitchen_excerpts.mp3`: three minutes made of
  one minute from the night, one from the active daytime and one from the evening of the piece, with short cross
  fades, so that the arc of the day can be heard quickly;
- `entropy_fm_<style>_parts.mp3`: each layer of the piece on its own for a few seconds, then all together;
- `entropy_fm_ambient_demo_full.mp3` and `entropy_fm_berlin_school_demo_full.mp3`: the whole piece;
- `condensed/entropy_fm_<style>_demo_condensed.mp3` (only with `--only condensed`): the whole condensed piece of the demo day;
- `entropy_fm_voice_library.mp3`: every voice of the library alone for four seconds on a short phrase;
- `soundscapes.md`: for each piece its length, its arc, what the day chose, how the house shaped it, the voices.
"""

from __future__ import annotations

import sys
import time
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import demo_render

if TYPE_CHECKING:
    from entropy_fm.plan import Plan
    from entropy_fm.quality import Quality
    from entropy_fm.score import Score

RATE = 44100
EXCERPT_SECONDS = 62
CROSS_SECONDS = 2
PART_SECONDS = 8
TOGETHER_SECONDS = 16
STYLE_TITLES = {
    "ambient": "Ambient",
    "berlin_school": "Berlin School",
    "lofi": "Lo-fi",
    "ritual": "Ritual",
    "glitch": "Cinematic Glitch",
}


def pick_windows(plan: Plan, length_s: int = EXCERPT_SECONDS) -> list[tuple[str, int]]:
    """
    Return where the three excerpts start: the middle of the night, the most active stretch of the day, the evening.

    :param plan: The plan of the piece.
    :param length_s: The length of an excerpt.
    """
    arc = {name: (a, b) for name, a, b in plan.arc()}
    length = plan.length
    activity = plan.activity + 0.5 * plan.power
    per_second = activity[: length * 10].reshape(length, 10).mean(axis=1)
    smooth = np.convolve(per_second, np.ones(length_s) / length_s, mode="valid")

    def best(low: float, high: float) -> int:
        first = int(max(low, 0))
        last = int(min(high - length_s, length - length_s - 3))
        if last <= first:
            return max(min(first, length - length_s - 3), 0)
        return first + int(np.argmax(smooth[first : last + 1]))

    night_a, night_b = arc["night"]
    day_a = arc["morning"][0]
    evening_a, evening_b = arc["evening"]
    return [
        ("night", int(max((night_a + night_b) / 2 - length_s / 2, 2))),
        ("active daytime", best(day_a, evening_a)),
        ("evening", best(evening_a, evening_b + 20)),
    ]


def crossfade(parts: list[np.ndarray], seconds: float = CROSS_SECONDS) -> np.ndarray:
    """Join pieces of audio with equal-power cross fades."""
    n = int(seconds * RATE)
    fade = np.linspace(0.0, np.pi / 2.0, n, dtype=np.float32)
    out = parts[0]
    for part in parts[1:]:
        mixed = out[-n:] * np.cos(fade)[:, None] + part[:n] * np.sin(fade)[:, None]
        out = np.concatenate([out[:-n], mixed, part[n:]])
    return out


def excerpts(
    score: Score,
    plan: Plan,
    tier: Quality,
    stats: dict[str, float] | None = None,
    seed: str = "x",
) -> np.ndarray:
    """
    Render the three-minute excerpts of a piece: the night, the active daytime and the evening.

    :param score: The score.
    :param plan: Its plan.
    :param tier: The quality tier.
    :param stats: Filled with what the limiter did.
    :param seed: Seeds the textures.
    """
    from entropy_fm.render import render_all  # noqa: PLC0415

    pieces = []
    worst = 0.0
    total = 0.0
    for _, start in pick_windows(plan):
        done: dict[str, float] = {}
        pieces.append(
            render_all(score, tier, seed, done, start_s=start, end_s=start + EXCERPT_SECONDS)
        )
        worst = max(worst, done["max_reduction_db"])
        total += done["mean_reduction_db"]
    if stats is not None:
        stats["max_reduction_db"] = worst
        stats["mean_reduction_db"] = total / 3.0
        stats["clip_share"] = 0.0
    return crossfade(pieces)[: 180 * RATE]


def best_part_window(score: Score, length: int, seconds: int = PART_SECONDS) -> int:
    """Return the start of the stretch of the piece in which the most layers have notes."""
    layers = list(score.layers)
    starts = np.array([n.t for n in score.notes])
    names = np.array([n.layer for n in score.notes])
    best, best_start = -1.0, 0
    for start in range(10, max(length - TOGETHER_SECONDS - 10, 11), 4):
        inside = (starts >= start - 6) & (starts < start + seconds)
        found = sum(1 for layer in layers if int(np.sum(names[inside] == layer)) >= 2)
        score_here = found + 0.001 * float(inside.sum())
        if score_here > best:
            best, best_start = score_here, start
    return best_start


def parts(score: Score, plan: Plan, tier: Quality, seed: str = "x") -> tuple[np.ndarray, list[str]]:
    """
    Render the parts showcase of a piece: each layer alone for a few seconds, then everything together.

    :param score: The score.
    :param plan: Its plan.
    :param tier: The quality tier.
    :param seed: Seeds the textures.
    :return: The audio and the list of what plays when.
    """
    from entropy_fm.render import render_all  # noqa: PLC0415

    start = best_part_window(score, plan.length)
    layers = list(score.layers)
    names = np.array([n.layer for n in score.notes])
    times = np.array([n.t for n in score.notes])
    chunks: list[np.ndarray] = []
    index: list[str] = []
    clock = 0.0
    gap = np.zeros((int(0.6 * RATE), 2), dtype=np.float32)
    for layer in layers:
        inside = (times >= start - 6) & (times < start + PART_SECONDS) & (names == layer)
        if int(inside.sum()) < 2:
            continue
        audio = render_all(
            score,
            tier,
            seed,
            None,
            start_s=start,
            end_s=start + PART_SECONDS,
            solo=frozenset({layer}),
        )
        voices = sorted(
            {
                n.voice
                for n in score.notes
                if n.layer == layer and start - 6 <= n.t < start + PART_SECONDS
            }
        )
        chunks += [audio, gap]
        index.append(
            f"{clock:6.1f} s  {score.layers[layer].title or layer} alone ({', '.join(voices)})"
        )
        clock += (len(audio) + len(gap)) / RATE
    together = render_all(score, tier, seed, None, start_s=start, end_s=start + TOGETHER_SECONDS)
    index.append(f"{clock:6.1f} s  everything together")
    chunks.append(together)
    return np.concatenate(chunks), index


PHRASES = {
    "bass": ([38, 38, 45, 41, 38], 0.62),
    "pad": ([50, 57, 62, 65], 3.0),
    "lead": ([69, 72, 74, 72, 69], 0.7),
    "key": ([62, 65, 69, 72, 69, 65], 0.55),
    "pluck": ([62, 69, 65, 72, 69], 0.5),
}


def voice_library(tier: Quality) -> tuple[np.ndarray, list[str]]:
    """
    Render every voice of the library alone for four seconds on a short phrase.

    :param tier: The quality tier.
    :return: The audio and a list of what plays when.
    """
    from pedalboard import Pedalboard, Reverb  # type: ignore[attr-defined]  # noqa: PLC0415

    from entropy_fm.voices import VOICES, render_note  # noqa: PLC0415

    slot = 4 * RATE
    chunks: list[np.ndarray] = []
    index: list[str] = []
    room = Pedalboard([Reverb(room_size=0.5, damping=0.6, wet_level=0.18, dry_level=0.9)])
    for number, (key, voice) in enumerate(VOICES.items()):
        pitches, span = PHRASES[voice.family] if voice.family != "pad" else PHRASES["pad"]
        if key == "tape_choir":
            pitches = [57, 60, 64]
        buffer = np.zeros((slot + 6 * RATE, 2), dtype=np.float32)
        if voice.family == "pad":
            for k, pitch in enumerate(pitches):
                wave = render_note(key, pitch, 3.0 - 0.1 * k, 0.6, tier)
                at = int(0.25 * k * RATE)
                buffer[at : at + len(wave)] += wave[: len(buffer) - at]
        else:
            for k, pitch in enumerate(pitches):
                wave = render_note(key, float(pitch), span * 0.9, 0.7, tier)
                at = int(span * k * RATE)
                buffer[at : at + len(wave)] += wave[: len(buffer) - at]
        out = room(np.ascontiguousarray(buffer.T), RATE).T[:slot].copy()
        fade = int(0.3 * RATE)
        out[-fade:] *= np.linspace(1.0, 0.0, fade)[:, None]
        chunks.append(out)
        index.append(f"{number * 4:4d} s  {voice.title} ({key}, {voice.family})")
    audio = np.concatenate(chunks)
    audio = audio - audio.mean(axis=0, keepdims=True)
    peak = float(np.abs(audio).max())
    if peak > 0.89:
        audio = audio * (0.89 / peak)
    return audio, index


def summary(score: Score, plan: Plan) -> list[str]:
    """Describe a piece in words and numbers: its arc, its choices, how the house shaped it, its voices."""
    lines: list[str] = []
    length = plan.length
    lines.append(
        f"- Length {length // 60} min {length % 60} s ({length} s)."
        + (f" Tempo {score.tempo:.0f} BPM." if score.tempo else " No beat.")
    )
    lines.append("- Arc (the day spread over the piece, in order):")
    names = {key: layer.title or key for key, layer in score.layers.items()}
    for name, a, b in plan.arc():
        inside = [n for n in score.notes if a <= n.t < b]
        counts: dict[str, int] = {}
        for n in inside:
            counts[n.layer] = counts.get(n.layer, 0) + 1
        i, j = int(a * 10), max(int(b * 10), int(a * 10) + 1)
        clock = f"{int(a) // 60}:{int(a) % 60:02d} to {int(b) // 60}:{int(b) % 60:02d}"
        playing = ", ".join(
            f"{names[k]} ({v})" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])
        )
        lines.append(
            f"  - {name}, {clock}: activity {plan.activity[i:j].mean():.2f}, power {plan.power[i:j].mean():.2f}, "
            f"brightness {plan.bright[i:j].mean():.2f}, humidity {plan.humid[i:j].mean():.2f}, lights {plan.lights[i:j].mean():.2f}; "
            f"playing: {playing or 'nothing'}."
        )
    lines.append("- What the day chose (dealt from the library, never the same two days in a row):")
    lines.extend(f"  - {key}: {value}" for key, value in score.choices.items())
    lines.append("- The generative systems at work:")
    lines.extend(f"  - {text}" for text in score.notes_about)
    shaping = []
    for name, curve in plan.runs.items():
        on = np.flatnonzero(curve > 0.5)
        if len(on):
            shaping.append(
                f"the {name} appliance ran from {on[0] // 10 // 60}:{on[0] // 10 % 60:02d} to {on[-1] // 10 // 60}:{on[-1] // 10 % 60:02d} of the piece and held its layer in"
            )
    for second, strength in plan.events:
        shaping.append(
            f"an anomaly at {int(second) // 60}:{int(second) % 60:02d} (strength {strength:.1f}) made a swell, a filter dip and a harmonic shift"
        )
    lines.append(
        "- How the house shaped it: "
        + ("; ".join(shaping) if shaping else "no appliance run and no anomaly on this day")
        + "."
    )
    used = score.voices_used()
    lines.append(
        "- Voices used (notes): "
        + ", ".join(f"{k} {v}" for k, v in sorted(used.items(), key=lambda kv: -kv[1]))
        + "."
    )
    return lines


def write_notes(
    out: Path,
    entries: list[tuple[str, str, Score, Plan]],
    voice_index: list[str],
    parts_index: dict[str, list[str]],
    extra: list[str],
) -> None:
    """Write soundscapes.md."""
    import importlib  # noqa: PLC0415

    lines = [
        "# Soundscapes",
        "",
        "Generated by `scripts/soundscapes.py`. A soundscape is one long piece for one room on one day;",
        "the day (night, morning, daytime, evening, late night) is spread over it in order.",
        "",
    ]
    lines += extra
    lines += ["", "## The generative system of each style", ""]
    for style in STYLE_TITLES:  # noqa: PLC0206
        module = importlib.import_module(
            f"entropy_fm.scape_{style.replace('lofi', 'lofi')}"
            if style != "berlin_school"
            else "entropy_fm.scape_berlin"
        )
        doc = (module.__doc__ or "").strip().split("\n\n")
        lines += [f"### {STYLE_TITLES[style]}", "", "\n\n".join(doc[1:4]).replace("\n", " "), ""]
    lines += ["## The pieces", ""]
    for title, style, score, plan in entries:  # noqa: B007
        lines += [f"### {title}", "", *summary(score, plan), ""]
        if f"{title}" in parts_index:
            lines += ["Parts showcase:", ""] + [f"    {row}" for row in parts_index[title]] + [""]
    lines += (
        ["## The voice library showcase (`entropy_fm_voice_library.mp3`)", ""]
        + [f"    {row}" for row in voice_index]
        + [""]
    )
    (out / "soundscapes.md").write_text("\n".join(lines))


def condensed_pieces(demo: demo_render.Demo, tier: Quality, out: Path) -> None:
    """
    Render the whole condensed piece of the demo day (anomaly day, kitchen) in every style.

    :param demo: The invented fortnight.
    :param tier: The quality tier.
    :param out: The folder, which gets `entropy_fm_<style>_demo_condensed.mp3`.
    """
    from entropy_fm.render import render_all  # noqa: PLC0415

    for style in demo.styles.STYLES:
        began = time.time()
        score, _ = demo.score(demo_render.ANOMALY_DAY, ("kitchen",), style, tier, "condensed")
        audio = render_all(score, tier, "condensed")
        demo_render.render_to(out, f"entropy_fm_{style}_demo_condensed", audio)
        print(
            f"{STYLE_TITLES[style]} condensed: {len(audio) / RATE:.1f} s, {time.time() - began:.0f} s",
            flush=True,
        )


def main() -> None:
    """Render everything."""
    arguments = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = Path(arguments[0]) if arguments else Path("/home/claude/out/soundscapes")
    out.mkdir(parents=True, exist_ok=True)
    only = (
        set(sys.argv[sys.argv.index("--only") + 1 :])
        if "--only" in sys.argv
        else {"excerpts", "parts", "voices", "full", "notes"}
    )
    demo = demo_render.Demo()
    import real_house_demo  # noqa: PLC0415

    tier = demo.quality.STANDARD
    if "condensed" in only:
        condensed_pieces(demo, tier, out / "condensed")
        only -= {"condensed"}
        if not only:
            return
    entries: list[tuple[str, str, Score, Plan]] = []
    parts_index: dict[str, list[str]] = {}
    extra: list[str] = []
    for style in demo.styles.STYLES:
        for label, make in (
            ("demo", partial(demo.score, demo_render.ANOMALY_DAY, ("kitchen",), style, tier)),
            ("kitchen", partial(real_house_demo.real_score, demo, style, tier)),
        ):
            began = time.time()
            score, plan = make()
            title = f"{STYLE_TITLES[style]}, {'demo day (anomaly day, kitchen)' if label == 'demo' else 'real-house kitchen'}"
            entries.append((title, style, score, plan))
            if "excerpts" in only:
                audio = excerpts(score, plan, tier)
                demo_render.render_to(out, f"entropy_fm_{style}_{label}_excerpts", audio)
            if "parts" in only and label == "demo":
                audio, index = parts(score, plan, tier)
                demo_render.render_to(out, f"entropy_fm_{style}_parts", audio)
                parts_index[title] = index
            if "full" in only and label == "demo" and style in {"ambient", "berlin_school"}:
                from entropy_fm.render import render_all  # noqa: PLC0415

                audio = render_all(score, tier, "full")
                demo_render.render_to(out, f"entropy_fm_{style}_demo_full", audio)
            print(f"{title}: {time.time() - began:.0f} s", flush=True)
    voice_index: list[str] = []
    if "voices" in only:
        audio, voice_index = voice_library(tier)
        demo_render.render_to(out, "entropy_fm_voice_library", audio)
    if "notes" in only:
        write_notes(out, entries, voice_index, parts_index, extra)


if __name__ == "__main__":
    main()
