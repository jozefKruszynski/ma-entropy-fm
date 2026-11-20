"""
Measure how fully the six styles play and how much the days differ, on the two demo houses.

    uv run python scripts/measure_drivers.py [--engine PATH] [--days 6] [--json out.json] [style ...]

For the motion-rich demo house (kitchen, living room) and the sparse house (no motion sensors: living room,
bedroom, office, bathroom, kitchen) it writes the soundscape score of the last days of the fortnight in each style, and
for each prints

- layers: how many layers play a note in a minute of the piece (mean, and the quietest minute),
- notes per minute of the piece, each counted as loud as it plays (its velocity times the presence of its layer):
  the mean, and the quietest and the busiest hour of the day,
- the day to day distance: how far consecutive days are from each other, as the mean absolute difference of the
  hourly profile (notes, layers, loudness of the notes) divided by its mean,
- the room to room distance: the same measure between the rooms of a house on the last day.

`--engine` points at another checkout of the repository (for instance a git worktree of an older commit) whose
`entropy_fm` package is measured instead of this one; the demo houses are always the ones of this folder.
"""

from __future__ import annotations

import json
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import demo_render

if "--engine" in sys.argv:
    # the demo houses come from here, the engine from the checkout that was asked for
    ENGINE = Path(sys.argv[sys.argv.index("--engine") + 1]).resolve()
    sys.path.insert(0, str(ENGINE))
LAST = demo_render.DAYS - 1
HOUSES = {
    "motion": ("kitchen", "living"),
    "sparse": ("living", "bedroom", "office", "kitchen", "bathroom"),
}
PRESENCE_FLOOR = 0.12


def layers_per_minute(score: object) -> np.ndarray:
    """Return, for each minute of the piece, how many layers have a note sounding in it while they are present."""
    length = int(score.length)  # type: ignore[attr-defined]
    minutes = max(length // 60, 1)
    found: dict[int, set[str]] = {m: set() for m in range(minutes)}
    layers = score.layers  # type: ignore[attr-defined]
    for note in score.notes:  # type: ignore[attr-defined]
        layer = layers.get(note.layer)
        if layer is None:
            continue
        first, last = int(note.t // 60), int(min(note.t + note.dur, length - 1) // 60)
        for minute in range(first, min(last, minutes - 1) + 1):
            if layer.presence is not None:
                index = int(np.clip((minute * 60 + 30) * 10, 0, len(layer.presence) - 1))
                if layer.presence[index] < PRESENCE_FLOOR:
                    continue
            found[minute].add(note.layer)
    return np.array([len(found[m]) for m in range(minutes)], dtype=float)


def hourly_profile(score: object, plan: object) -> np.ndarray:
    """Return 3 values for each of the 24 hours of the day: notes a minute, layers, mean velocity."""
    length = float(score.length)  # type: ignore[attr-defined]
    covered = float(plan.covered)  # type: ignore[attr-defined]
    per_minute = layers_per_minute(score)
    notes = np.zeros(24)
    velocity = np.zeros(24)
    layers = np.zeros(24)
    seconds = np.zeros(24)
    layers_of = score.layers  # type: ignore[attr-defined]
    for note in score.notes:  # type: ignore[attr-defined]
        hour = int(min(note.t / length * covered / 3600.0, 23.99))
        # a note counts as loud as it plays: its velocity and how present its layer is
        layer = layers_of.get(note.layer)
        here = 1.0
        if layer is not None and layer.presence is not None:
            here = float(layer.presence[int(np.clip(note.t * 10, 0, len(layer.presence) - 1))])
        notes[hour] += note.vel * here
        velocity[hour] += note.vel
    for minute, count in enumerate(per_minute):
        hour = int(min((minute * 60 + 30) / length * covered / 3600.0, 23.99))
        layers[hour] += count
        seconds[hour] += 1.0
    span = np.maximum(np.array([length / covered * 3600.0 / 60.0] * 24), 1e-9)
    # a piece of today may cover less than the whole day: hours it does not reach stay empty
    minutes = np.maximum(seconds, 1.0)
    return np.stack(
        [
            notes / np.maximum(span, 1e-9),
            layers / minutes,
            np.where(notes > 0, velocity / np.maximum(notes, 1.0), 0.0),
        ]
    )


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return the mean absolute difference of two profiles, relative to the mean of the two."""
    scale = (np.abs(a).mean(axis=1) + np.abs(b).mean(axis=1)) / 2.0
    scale = np.maximum(scale, 1e-9)
    return float(np.mean(np.abs(a - b).mean(axis=1) / scale))


def measure(house: str, style: str, room: str, days: int) -> dict[str, float]:
    """Measure one room of one house in one style over the last days."""
    demo = MEASURERS[house]
    profiles, layer_means, layer_floors, per_min = [], [], [], []
    for index in range(LAST - days + 1, LAST + 1):
        score, plan = demo.score(index, (room,), style)
        profile = hourly_profile(score, plan)
        profiles.append(profile)
        per = layers_per_minute(score)
        # the first and last minute are fades
        core = per[1:-1] if len(per) > 4 else per
        layer_means.append(float(core.mean()))
        layer_floors.append(float(np.percentile(core, 10)))
        per_min.append(float(profile[0].mean()))
    steps = [distance(a, b) for a, b in pairwise(profiles)]
    LAST_PROFILES[(house, style, room)] = profiles[-1]
    notes_by_hour = np.mean([p[0] for p in profiles], axis=0)
    return {
        "layers": float(np.mean(layer_means)),
        "layers_floor": float(np.mean(layer_floors)),
        "notes_min": float(np.mean(per_min)),
        "notes_hour_low": float(notes_by_hour.min()),
        "notes_hour_high": float(notes_by_hour.max()),
        "day_to_day": float(np.mean(steps)),
    }


MEASURERS: dict[str, demo_render.Demo] = {}
LAST_PROFILES: dict[tuple[str, str, str], np.ndarray] = {}


def main() -> None:
    """Print the table."""
    arguments = sys.argv[1:]
    days = int(arguments[arguments.index("--days") + 1]) if "--days" in arguments else 6
    out = arguments[arguments.index("--json") + 1] if "--json" in arguments else None
    skip = {"--engine", "--days", "--json"}
    values = {
        arguments[i + 1] for i, a in enumerate(arguments) if a in skip and i + 1 < len(arguments)
    }
    wanted = [a for a in arguments if a not in skip and a not in values and not a.startswith("--")]
    for house in HOUSES:
        MEASURERS[house] = demo_render.Demo(house)
    styles = wanted or list(MEASURERS["motion"].styles.STYLES)
    results: dict[str, dict[str, dict[str, float]]] = {}
    print(
        "| house | style | room | layers (mean / quiet 10%) | loud notes/min | by hour (low-high) | day to day |"
    )
    print("|---|---|---|---|---|---|---|")
    for house, rooms in HOUSES.items():
        for style in styles:
            for room in rooms:
                m = measure(house, style, room, days)
                results.setdefault(f"{house}/{style}", {})[room] = m
                print(
                    f"| {house} | {style} | {room} | {m['layers']:.1f} / {m['layers_floor']:.1f} | "
                    f"{m['notes_min']:.0f} | {m['notes_hour_low']:.0f}-{m['notes_hour_high']:.0f} | "
                    f"{m['day_to_day']:.2f} |",
                    flush=True,
                )
    print()
    print("| house | style | room to room |")
    print("|---|---|---|")
    for house, rooms in HOUSES.items():
        for style in styles:
            found = [LAST_PROFILES[(house, style, room)] for room in rooms]
            pairs = [distance(a, b) for i, a in enumerate(found) for b in found[i + 1 :]]
            spread = float(np.mean(pairs))
            results[f"{house}/{style}"]["_room_to_room"] = {"distance": spread}
            print(f"| {house} | {style} | {spread:.2f} |", flush=True)
    if out:
        Path(out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
