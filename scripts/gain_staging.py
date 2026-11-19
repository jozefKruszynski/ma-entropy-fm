"""
Measure the level of every layer of every style before the master, and print the trims that balance the mix.

    uv run python scripts/gain_staging.py [--write]

Each layer is rendered on its own (reverb and echo included, no master) in the busiest stretches of three days of
the demo house, in the Standard tier. The trim is the design level of `levels.DESIGN_DB` minus the level that was
measured with the trims at zero. With `--write` the table is written into `entropy_fm/levels.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1]))

import demo_render
from soundscapes import pick_windows

from entropy_fm import levels
from entropy_fm.render import Renderer

DAYS = (demo_render.ANOMALY_DAY, demo_render.ANOMALY_DAY - 3, demo_render.ANOMALY_DAY - 6)


def level_of(
    score: object, tier: object, start: int, layer: str, seconds: int = 24
) -> float | None:
    """Return the RMS in dB of a layer alone, before the master, over some seconds; None when it is silent."""
    renderer = Renderer(score, tier, "gain", frozenset({layer}))  # type: ignore[arg-type]
    blocks = [renderer.render_block(block) for block in range(start - 8, start + seconds)]
    audio = np.concatenate(blocks[8:])
    if float(np.abs(audio).max()) < 1e-5:
        return None
    # only the frames in which the layer plays count: a layer that rests half of the time is as loud as its notes
    frame = 4410
    frames = audio[: len(audio) // frame * frame].reshape(-1, frame, 2)
    power = (frames**2).mean(axis=(1, 2))
    live = power > power.max() * 0.01
    return float(10.0 * np.log10(power[live].mean() + 1e-12))


def voice_energies() -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    """Measure the energy of every voice (a + b times the held seconds) and of every drum hit, for the loudness estimate."""
    from entropy_fm.drums import HITS, render_hit  # noqa: PLC0415
    from entropy_fm.quality import STANDARD  # noqa: PLC0415
    from entropy_fm.voices import VOICES, render_note  # noqa: PLC0415

    voices: dict[str, tuple[float, float]] = {}
    for key, voice in VOICES.items():
        energies = []
        for held in (1.0, 4.0):
            wave = render_note(key, voice.ref_midi, held, 0.75, STANDARD)
            energies.append(float(np.sum(wave.astype(np.float64) ** 2)) / 2.0 / 44100.0)
        slope = max((energies[1] - energies[0]) / 3.0, 0.0)
        voices[key] = (round(max(energies[0] - slope, 0.0), 5), round(slope, 5))
    from entropy_fm import stretch  # noqa: PLC0415

    for key in stretch.WASHES:
        # a wash is as loud all along (fades aside), so it has no fixed part: its energy is per second of its own length
        wave = stretch.render_wash(key, 50.0, 24.0, 0.75)
        voices[key] = (
            0.0,
            round(float(np.sum(wave.astype(np.float64) ** 2)) / 2.0 / 44100.0 / 24.0, 5),
        )
    hits = {
        name: round(float(np.sum(render_hit(name, 0.8).astype(np.float64) ** 2)) / 2.0 / 44100.0, 6)
        for name in HITS
    }
    return voices, hits


def main() -> None:
    """Measure and print."""
    demo = demo_render.Demo()
    voices, hits = voice_energies()
    tier = demo.quality.STANDARD
    levels.TRIMS.clear()
    levels.VOICE_ENERGY.clear()
    levels.HIT_ENERGY.clear()
    result: dict[str, dict[str, float]] = {}
    for style, design in levels.DESIGN_DB.items():
        found: dict[str, list[float]] = {layer: [] for layer in design}
        for index in DAYS:
            score, plan = demo.score(index, ("kitchen",), style, tier)
            for _, start in pick_windows(plan)[1:]:
                for layer in design:
                    if layer in score.layers:
                        value = level_of(score, tier, start, layer)
                        if value is not None:
                            found[layer].append(value)
        result[style] = {
            layer: round(design[layer] - float(np.median(values)), 1)
            for layer, values in found.items()
            if values
        }
        print(
            style, {layer: f"{np.median(values):.1f}" for layer, values in found.items() if values}
        )
    print("TRIMS =", result)
    if "--write" in sys.argv:
        path = Path(levels.__file__)
        text = path.read_text()
        head = (
            text[: text.index("# the trims in dB")]
            + "# the trims in dB that scripts/gain_staging.py measured; zero where nothing was measured yet\n"
        )
        body = "TRIMS: dict[str, dict[str, float]] = {\n"
        for style, trims in result.items():
            body += (
                f'    "{style}": {{' + ", ".join(f'"{k}": {v}' for k, v in trims.items()) + "},\n"
            )
        body += "}\n"
        body += "# the energy of a note of each voice at velocity 0.75, in mean-square seconds, as a + b * the seconds it is held\n"
        body += "VOICE_ENERGY: dict[str, tuple[float, float]] = {\n"
        body += "".join(f'    "{k}": ({a}, {b}),\n' for k, (a, b) in voices.items())
        body += "}\n# the energy of a drum hit at velocity 0.8, in mean-square seconds\n"
        body += (
            "HIT_ENERGY: dict[str, float] = {"
            + ", ".join(f'"{k}": {v}' for k, v in hits.items())
            + "}\n"
        )
        body += "# a note keeps sounding this long after its key goes up, for the estimate of the loudness\nTAIL_SECONDS = 0.8\n"
        tail = text[text.index("\n\ndef trim_db") :]
        path.write_text(head + body + tail)
        print("written")


if __name__ == "__main__":
    main()
