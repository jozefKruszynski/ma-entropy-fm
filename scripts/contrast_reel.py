"""
Render the contrast reel: six excerpts of three pairs of room and day, back to back, one reel per style.

    uv run python scripts/contrast_reel.py OUT_DIR [--style STYLE ...] [--source real|sparse] [--seconds 20]
        [--pairs core|extra] [--engine PATH]

Each pair is one room on a quiet and a busy day, two rooms on one day, or the whole house on a workday and a day off.
Every item is scored in the Standard tier, the busiest stretch of the piece is rendered, and the six excerpts are
joined with 1 s cross fades into `OUT_DIR/contrast_<style>.mp3` (a wav without ffmpeg), next to a table of what
differs per pair in `contrast_<style>.md`. The source is the real house of `pull_house_fixture.py` when its folder
has files, else the invented sparse house. `--engine` plays another checkout (a worktree of `main`). Real data stays in OUT_DIR, which should be outside the repository or in the gitignored `out/`.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY = Path(__file__).parents[1]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(REPOSITORY))

import analyze_audio  # noqa: E402
import demo_render  # noqa: E402

FIXTURE_DIR = REPOSITORY / ".data" / "fixtures" / "real_house"
REAL = "real"
SPARSE = "sparse"
CORE = "core"
EXTRA = "extra"
DEFAULT_SECONDS = 20
CROSS_SECONDS = 1.0
SAMPLE_RATE = 44100
# two days sound alike when the tempo is closer than this, the key and mode are the same and no voice differs
SAMEY_TEMPO_BPM = 4.0
HOME_ROOT = 2
HOME_MODE = "dorian"
ANOMALY_SCORE = 0.5
# the plan's curves have this many values per second
CURVE_RATE = 10
MP3_SUFFIX = ".mp3"
WAV_SUFFIX = ".wav"
# the keys of `plan.choices` that name a voice or a patch; the others are patterns and progressions
VOICE_KEYS = frozenset(
    {
        "loop voice A",
        "loop voice B",
        "piano",
        "drone voice",
        "sequence voice",
        "second voice",
        "lead voice",
        "strings",
        "growl bass",
        "growl character",
        "keys",
        "bass voice",
        "pluck",
        "pluck voice",
        "figure voice",
        "pad",
        "kick",
        "washes",
        "patch",
    }
)
# choices that the table shows elsewhere: the tempo, and the root, which is in the key
NOT_A_PATTERN = frozenset({"tempo", "root"})
SPARSE_WEEKDAY = 3
SPARSE_DAY_OFF = 6
WHOLE_HOUSE = "whole_house"
# a pair is two (room, day) items; a room of None is the whole house
Entry = tuple[str | None, date]
Pair = tuple[Entry, Entry]
# the bedroom's busiest day is Monday 16, not Tuesday 17 (the CO2 peak of Tuesday is the night before it)
REAL_CORE: tuple[Pair, ...] = (
    (("living_room", date(2026, 11, 14)), ("living_room", date(2026, 11, 16))),
    (("bedroom", date(2026, 11, 15)), ("bedroom", date(2026, 11, 16))),
    ((None, date(2026, 11, 19)), (None, date(2026, 11, 15))),
)
REAL_EXTRA: tuple[Pair, ...] = (
    (("bathroom", date(2026, 11, 15)), ("bathroom", date(2026, 11, 19))),
    (("office", date(2026, 11, 15)), ("office", date(2026, 11, 18))),
)


@dataclass
class Item:
    """One room on one day: what is rendered and measured."""

    room: str
    day: date
    character: Any
    plan: Any
    score: Any
    start: int
    audio: np.ndarray

    @property
    def tempo(self) -> float:
        """Return the tempo of the piece in bpm, 0 for a style without one."""
        return float(self.score.tempo)

    @property
    def key(self) -> str:
        """Return the key and mode of the piece."""
        from entropy_fm.synth import NOTE_NAMES  # noqa: PLC0415

        return f"{NOTE_NAMES[self.plan.scale.root % len(NOTE_NAMES)]} {self.plan.scale.mode}"

    @property
    def voices(self) -> dict[str, str]:
        """Return the choices of the plan that name a voice or patch."""
        return {k: v for k, v in self.plan.choices.items() if k in VOICE_KEYS}

    @property
    def patterns(self) -> dict[str, str]:
        """Return the choices of the plan that are patterns, cells, progressions or the like."""
        return {
            k: v
            for k, v in self.plan.choices.items()
            if k not in VOICE_KEYS and k not in NOT_A_PATTERN
        }

    @property
    def notes_per_minute(self) -> float:
        """Return how many notes the score plays per minute inside the excerpt."""
        seconds = len(self.audio) / SAMPLE_RATE
        inside = sum(1 for note in self.score.notes if self.start <= note.t < self.start + seconds)
        return inside * 60.0 / seconds

    @property
    def centroid(self) -> float:
        """Return the spectral centroid of the excerpt in Hz."""
        return float(analyze_audio.measure(self.audio.astype(np.float64))["centroid_hz"])

    @property
    def rms_db(self) -> float:
        """Return the RMS loudness of the excerpt in dBFS."""
        return float(analyze_audio.measure(self.audio.astype(np.float64))["rms_db"])

    def label(self) -> str:
        """Return the one-line log label of the item."""
        choices = ", ".join(f"{k}={v}" for k, v in self.plan.choices.items())
        tempo = f"{self.tempo:.0f} bpm" if self.tempo else "no tempo"
        return (
            f"{self.room} {self.day:%a %d %b}: {self.character.label()} | {self.key}, {tempo}"
            f" | {choices}"
        )


class Source:
    """Where the series of a room and day come from: the real fixture or the sparse house."""

    def __init__(self, kind: str) -> None:
        """
        Load the source.

        :param kind: `real` or `sparse`.
        """
        self.kind = kind
        self.demo: Any = demo_render.Demo("sparse") if kind == SPARSE else None
        if kind == REAL:
            import house_fixture  # noqa: PLC0415

            self.fixture: Any = house_fixture

    def inputs(
        self, room: str | None, day: date
    ) -> tuple[list[Any], dict[str, Any], float, str | None, bool | None, list[float]]:
        """
        Return the series, baselines, covered seconds, weather, workday sensor and week of a room (None: the house) on a day.

        The week is how busy the room's earlier days were, which the energy of the day is ranked against.

        :param room: The room, or None for the whole house.
        :param day: The day.
        """
        if self.kind == REAL:
            series, baselines, covered = self.fixture.load_day(room, day)
            return (
                series,
                baselines,
                covered,
                self.fixture.weather_of(day),
                self.fixture.workday_of(day),
                self.fixture.load_week(room, day, baselines),
            )
        from entropy_fm.references import rescale  # noqa: PLC0415

        index = (day - demo_render.FIRST_DAY).days
        rooms = (room,) if room else tuple(k for k in self.demo.days[index] if k != "sun")
        series = [s for area in rooms for s in self.demo.days[index][area]] + self.demo.days[index][
            "sun"
        ]
        baselines = self.demo.baselines(index, rooms)
        series, _ = rescale(series, baselines)
        return (
            series,
            baselines,
            demo_render.DAY,
            None,
            None,
            self.demo.week(index, rooms, baselines),
        )


@dataclass
class Prepared:
    """A room and day as the provider reads it, before the score."""

    series: list[Any]
    baselines: dict[str, Any]
    covered: float
    traits: Any
    analysis: Any
    character: Any


def prepare(source: Source, room: str | None, day: date) -> Prepared:
    """
    Measure a room (None: the whole house) on a day: its traits, analysis and character.

    :param source: The series of the house.
    :param room: The room, or None for the whole house.
    :param day: The day.
    """
    from entropy_fm.palette import measure_traits  # noqa: PLC0415
    from entropy_fm.scape import analysis_of  # noqa: PLC0415
    from entropy_fm.surprise import Surprise  # noqa: PLC0415

    series, baselines, covered, weather, workday, week = source.inputs(room, day)
    traits = measure_traits(series, baselines, covered, weather)
    featured = Surprise(series, baselines).featured(0.0, 1.0, covered)
    anomalies = [(t, score, entity) for t, score, entity in featured if score >= ANOMALY_SCORE]
    analysis = analysis_of(series, covered, anomalies)
    try:
        from entropy_fm.character import measure_character  # noqa: PLC0415
    except ImportError:  # an older engine has no character of the day
        character = None
    else:
        character = measure_character(analysis, traits, day, series, workday=workday, week=week)
    return Prepared(series, baselines, covered, traits, analysis, character)


def build_item(source: Source, room: str | None, day: date, style: str, seconds: int) -> Item:
    """
    Score a room and day, find the busiest stretch, and render it.

    :param source: The series of the house.
    :param room: The room, or None for the whole house.
    :param day: The day.
    :param style: The style.
    :param seconds: The length of the excerpt.
    """
    from entropy_fm.palette import derive_palette  # noqa: PLC0415
    from entropy_fm.quality import STANDARD  # noqa: PLC0415
    from entropy_fm.render import render_all  # noqa: PLC0415
    from entropy_fm.scape import scape_score  # noqa: PLC0415
    from entropy_fm.styles import get_style  # noqa: PLC0415
    from entropy_fm.synth import Scale  # noqa: PLC0415

    found = prepare(source, room, day)
    name = room or WHOLE_HOUSE
    home = Scale(HOME_ROOT, HOME_MODE)
    extra: dict[str, Any] = {"character": found.character} if found.character else {}
    score, plan = scape_score(
        found.series,
        found.analysis,
        style,
        f"{name}|{day.isoformat()}|_scape|{style}",
        home,
        day=day,
        palette=derive_palette(get_style(style), day, home, found.traits),
        quality=STANDARD,
        area_key=name,
        **extra,
    )
    start = busiest_start(plan, seconds)
    audio = render_all(score, STANDARD, "contrast", None, start_s=start, end_s=start + seconds)
    return Item(name, day, found.character or plan.character, plan, score, start, audio)


def busiest_start(plan: Any, seconds: int) -> int:
    """
    Return the second of the piece where the stretch with the highest mean activity starts.

    :param plan: The plan of the piece.
    :param seconds: The length of the stretch.
    """
    length = int(plan.length)
    per_second = plan.activity[: length * CURVE_RATE].reshape(length, CURVE_RATE).mean(axis=1)
    if length <= seconds:
        return 0
    window = np.convolve(per_second, np.ones(seconds) / seconds, mode="valid")
    return int(np.argmax(window))


def real_pairs(*, extra: bool) -> list[Pair]:
    """
    Return the pairs of the handover on the real house.

    :param extra: Whether the bathroom and the office are added.
    """
    return [*REAL_CORE, *(REAL_EXTRA if extra else ())]


def sparse_pairs(source: Source, *, extra: bool) -> list[Pair]:
    """
    Return the pairs of the sparse house: a room's quietest and busiest day, two rooms, the house weekday and weekend.

    :param source: The sparse house.
    :param extra: Whether the bathroom and the office are added, each on its quietest and busiest day.
    """
    days = [demo_render.FIRST_DAY + timedelta(days=i) for i in range(demo_render.DAYS)]

    def span(room: str) -> Pair:
        energy = {d: prepare(source, room, d).character.energy for d in days}
        return (room, min(energy, key=lambda d: energy[d])), (
            room,
            max(energy, key=lambda d: energy[d]),
        )

    living = span("living")
    busy = living[1][1]
    weekday = next(d for d in days if d.weekday() == SPARSE_WEEKDAY)
    off = next(d for d in days if d.weekday() == SPARSE_DAY_OFF)
    pairs: list[Pair] = [
        living,
        (("living", busy), ("bedroom", busy)),
        ((None, weekday), (None, off)),
    ]
    if extra:
        pairs += [span("bathroom"), span("office")]
    return pairs


def difference(first: Item, second: Item) -> dict[str, Any]:
    """
    Return what differs between the two items of a pair.

    :param first: The first item.
    :param second: The second item.
    """
    first_voices, second_voices = first.voices, second.voices
    changed = _changed(first_voices, second_voices)
    tempo_gap = abs(first.tempo - second.tempo)
    samey = tempo_gap < SAMEY_TEMPO_BPM and first.key == second.key and not changed
    return {
        "tempo_gap": tempo_gap,
        "changed": changed,
        "patterns": _changed(first.patterns, second.patterns),
        "samey": samey,
    }


def _changed(first: dict[str, str], second: dict[str, str]) -> list[str]:
    """Return the keys whose values differ between two sets of choices."""
    return sorted(k for k in first.keys() | second.keys() if first.get(k) != second.get(k))


def table_of(style: str, pairs: list[tuple[Item, Item]]) -> str:
    """
    Return the markdown table of the measured differences of a reel.

    :param style: The style.
    :param pairs: The items of each pair.
    """
    lines = [
        f"# Contrast reel: {style}",
        "",
        "| pair | tempo | key and mode | voices that differ | patterns and progressions that differ | notes/min | centroid Hz | RMS dBFS | verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for first, second in pairs:
        diff = difference(first, second)
        lines.append(
            f"| {first.room} {first.day:%a %d} vs {second.room} {second.day:%a %d} "
            f"| {first.tempo:.0f} vs {second.tempo:.0f} "
            f"| {first.key} vs {second.key} "
            f"| {', '.join(diff['changed']) or 'none'} "
            f"| {', '.join(diff['patterns']) or 'none'} "
            f"| {first.notes_per_minute:.0f} vs {second.notes_per_minute:.0f} "
            f"| {first.centroid:.0f} vs {second.centroid:.0f} "
            f"| {first.rms_db:.1f} vs {second.rms_db:.1f} "
            f"| {'SAMEY' if diff['samey'] else 'apart'} |"
        )
    return "\n".join(lines) + "\n"


def write_audio(out: Path, name: str, audio: np.ndarray) -> Path:
    """
    Write the reel as mp3, or as wav when there is no ffmpeg.

    :param out: The folder.
    :param name: The file name without suffix.
    :param audio: Float stereo samples.
    """
    out.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg"):
        return demo_render.render_to(out, name, audio)
    path = out / f"{name}{WAV_SUFFIX}"
    demo_render.write_wav(path, audio)
    return path


def main() -> None:
    """Render the reels."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument(
        "--style", action="append", help="a style key; repeat for more (default all)"
    )
    parser.add_argument("--source", choices=(REAL, SPARSE))
    parser.add_argument("--seconds", type=int, default=DEFAULT_SECONDS)
    parser.add_argument("--pairs", choices=(CORE, EXTRA), default=CORE)
    parser.add_argument("--engine", type=Path, help="another checkout of the repository to play")
    args = parser.parse_args()
    if args.engine:
        sys.path.insert(0, str(args.engine.resolve()))
    has_real = FIXTURE_DIR.is_dir() and any(FIXTURE_DIR.glob("*.json"))
    kind = args.source or (REAL if has_real else SPARSE)
    if kind == REAL and not has_real:
        sys.exit(
            f"No real house in {FIXTURE_DIR}: run scripts/pull_house_fixture.py first, or use --source sparse."
        )
    if kind == SPARSE and not args.source:
        print(f"No real house in {FIXTURE_DIR}, using the sparse house.")
    from entropy_fm.styles import STYLES  # noqa: PLC0415

    source = Source(kind)
    for style in args.style or list(STYLES):
        pairs = (
            real_pairs(extra=args.pairs == EXTRA)
            if kind == REAL
            else sparse_pairs(source, extra=args.pairs == EXTRA)
        )
        built: list[tuple[Item, Item]] = []
        for entries in pairs:
            both = []
            for room, day in entries:
                item = build_item(source, room, day, style, args.seconds)
                print(item.label(), flush=True)
                both.append(item)
            built.append((both[0], both[1]))
        audio = join([item.audio for pair in built for item in pair])
        path = write_audio(args.out_dir, f"contrast_{style}", audio)
        table = table_of(style, built)
        (args.out_dir / f"contrast_{style}.md").write_text(table)
        print(table)
        print(path)


def join(parts: list[np.ndarray]) -> np.ndarray:
    """
    Join excerpts with equal-power cross fades.

    :param parts: Float stereo excerpts.
    """
    count = int(CROSS_SECONDS * SAMPLE_RATE)
    fade = np.linspace(0.0, np.pi / 2.0, count, dtype=np.float32)
    out = parts[0]
    for part in parts[1:]:
        mixed = out[-count:] * np.cos(fade)[:, None] + part[:count] * np.sin(fade)[:, None]
        out = np.concatenate([out[:-count], mixed, part[count:]])
    return out


if __name__ == "__main__":
    main()
