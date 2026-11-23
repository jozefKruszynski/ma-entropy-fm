"""Tests that the data picks the identity of Lo-fi and Cinematic Glitch: a quiet and a busy day of one room differ."""

# ruff: noqa: SLF001

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from functools import cache
from unittest.mock import patch

import pytest

from entropy_fm import glitch_kits, harmony, scape_glitch, scape_lofi
from entropy_fm.character import ALL_DAY, FLAT, HIGH, LOW, NEUTRAL, scale_of_day
from entropy_fm.plan import CLIMAX, FORM_BARS, MIN_FORM_CELLS, SHORT_FORM_BARS, Plan
from entropy_fm.score import Score
from entropy_fm.styles import GLITCH, LOFI, STYLES
from entropy_fm.synth import SCALES, Scale

from .scape_tools import demo

QUIET = replace(NEUTRAL, energy=0.08, level=LOW, shape=FLAT, warmth=-0.8, stretches=1)
BUSY = replace(NEUTRAL, energy=0.95, level=HIGH, shape=ALL_DAY, warmth=0.8, stretches=4)
HOME = Scale(2, "dorian")
KIT_LAYERS = {"kick", "snare", "hats"}
BASS_CEILING = 48
MOTIF_LOWEST = 60
# a day this much busier plays at least this many times the notes of the quiet one
LOFI_NOTES_RATIO = 1.5
GLITCH_KIT_RATIO = 2.0
GLITCH_FAMILIES = 6
# the most a voice moves between two chords, and the share of the chords that hold a common tone where there is one
MOST_VOICE_MOVE = 7
HELD_SHARE = 0.6


@cache
def _made(style: str, *, quiet: bool) -> tuple[Score, Plan]:
    """Return the score and the plan of the same room and day, made on a clearly quiet or a clearly busy character."""
    _, base = demo(style)
    assert base.day is not None
    character = QUIET if quiet else BUSY
    scale = scale_of_day(STYLES[style], HOME, base.day, character)
    plan = replace(base, character=character, scale=scale, choices={}, notes=[])
    module = scape_lofi if style == "lofi" else scape_glitch
    return module.compose(plan), plan


def _per_minute(score: Score, layers: set[str] | None = None) -> float:
    notes = [n for n in score.notes if layers is None or n.layer in layers]
    return len(notes) / score.length * 60.0


def _pitches(scale: Scale) -> set[int]:
    return {(scale.root + step) % 12 for step in scale.intervals}


# ---- Lo-fi


def test_lofi_quiet_and_busy_days_differ_in_the_expected_direction() -> None:
    """A busy day is faster, changes chord sooner, plays a busier kit and far more notes than a quiet one."""
    quiet, quiet_plan = _made("lofi", quiet=True)
    busy, busy_plan = _made("lofi", quiet=False)
    assert quiet.tempo < busy.tempo
    assert int(quiet_plan.choices["chord bars"]) > int(busy_plan.choices["chord bars"])
    energy = {p.name: p.energy for p in scape_lofi.DRUMS}
    assert energy[quiet_plan.choices["drum pattern"]] < energy[busy_plan.choices["drum pattern"]]
    comp = {p.name: p.energy for p in scape_lofi.COMPS}
    assert comp[quiet_plan.choices["comping"]] < comp[busy_plan.choices["comping"]]
    assert _per_minute(busy) > LOFI_NOTES_RATIO * _per_minute(quiet)
    assert _per_minute(busy, {"drums"}) > 2.0 * _per_minute(quiet, {"drums"})


@pytest.mark.parametrize("mode", LOFI.modes)
def test_lofi_progression_agrees_with_the_mode(mode: str) -> None:
    """Whatever the mode of the day, the progression that is picked is one that belongs to it."""
    _, base = demo("lofi")
    for quiet in (True, False):
        plan = replace(
            base,
            character=QUIET if quiet else BUSY,
            scale=Scale(base.scale.root, mode),
            choices={},
            notes=[],
        )
        scape_lofi.compose(plan)
        chosen = next(p for p in scape_lofi.PROGRESSIONS if p.name == plan.choices["progression"])
        assert mode in chosen.modes


def test_lofi_progressions_stay_inside_their_modes() -> None:
    """The chord tones are notes of the mode, but for the dominant and the borrowed backdoor chord."""
    for prog in scape_lofi.PROGRESSIONS:
        for mode in prog.modes:
            allowed = _pitches(Scale(0, mode))
            for offset, quality in prog.chords:
                if "backdoor" in prog.name:
                    continue
                assert offset % 12 in allowed, (prog.name, mode, offset)
                for name in {quality, scape_lofi.EXTENDED[quality]}:
                    tones = {(offset + i) % 12 for i in scape_lofi.QUALITIES[name]}
                    if name.startswith("dom"):
                        continue
                    assert tones <= allowed, (prog.name, mode, offset, name)
    for mode in LOFI.modes:
        assert sum(mode in p.modes for p in scape_lofi.PROGRESSIONS) >= 3


def test_lofi_chords_are_led_by_step_and_hold_common_tones() -> None:
    """Between two chords of a progression the voices move a few semitones, and mostly a shared pitch stays put."""
    held, shared_count = 0, 0
    for prog in scape_lofi.PROGRESSIONS:
        for extended in (False, True):
            previous = None
            for chord in (*prog.chords, prog.chords[0]):
                notes = scape_lofi._voicing(0, chord, previous, extended=extended)
                assert min(notes) >= scape_lofi.KEYS_LOW
                assert max(notes) <= scape_lofi.KEYS_HIGH
                if previous:
                    moves = [abs(a - b) for a, b in zip(sorted(previous), notes, strict=False)]
                    assert max(moves) <= MOST_VOICE_MOVE, (prog.name, moves)
                    if {n % 12 for n in previous} & {n % 12 for n in notes}:
                        shared_count += 1
                        held += bool(set(previous) & set(notes))
                previous = notes
    assert held >= HELD_SHARE * shared_count


def test_lofi_states_the_motif_in_the_key_of_the_day_and_keeps_its_registers() -> None:
    """The motif of the room is played on the plucked line, from C4 up; the bass stays under C3."""
    for quiet in (True, False):
        score, plan = _made("lofi", quiet=quiet)
        expected = harmony.motif_notes(plan, plan.motif())
        assert min(midi for midi, _ in expected) >= MOTIF_LOWEST
        assert {midi % 12 for midi, _ in expected} <= _pitches(plan.scale)
        played = [n.midi for n in score.notes if n.layer == "pluck"]
        assert {float(midi) for midi, _ in expected} <= set(played)
        assert max(n.midi for n in score.notes if n.layer == "bass") < BASS_CEILING


def test_lofi_sections_sit_on_phrase_lines_and_the_layers_enter_there() -> None:
    """Sections start on multiples of 8 bars (4 in a short piece); the bass and the kit enter on those lines."""
    for quiet in (True, False):
        score, plan = _made("lofi", quiet=quiet)
        bar = 4.0 * 60.0 / score.tempo
        phrase = (
            FORM_BARS if plan.length / (FORM_BARS * bar) >= MIN_FORM_CELLS else SHORT_FORM_BARS
        ) * bar
        sections = plan.sections(bar)
        assert all(abs(a / phrase - round(a / phrase)) < 1e-6 for a, *_ in sections)
        for layer in ("bass", "drums") if not quiet else ("bass",):
            first = min(
                n.t for n in score.notes if n.layer == layer and n.voice != "drum:snare_lofi"
            )
            assert abs(first / phrase - round(first / phrase)) * phrase < 0.25, (
                layer,
                first,
                phrase,
            )


def test_lofi_tempo_is_one_tempo_inside_the_range() -> None:
    """The tempo is one number inside the style's range."""
    for quiet in (True, False):
        score, _ = _made("lofi", quiet=quiet)
        assert LOFI.tempo[0] <= score.tempo <= LOFI.tempo[1]


# ---- Cinematic Glitch


def test_glitch_quiet_and_busy_days_differ_in_the_expected_direction() -> None:
    """A busy day is a little faster, changes chord sooner, plays a busier pattern and a far fuller kit."""
    quiet, quiet_plan = _made("glitch", quiet=True)
    busy, busy_plan = _made("glitch", quiet=False)
    assert quiet.tempo < busy.tempo
    assert int(quiet_plan.choices["chord bars"]) > int(busy_plan.choices["chord bars"])
    energy = glitch_kits.DRUM_ENERGY
    assert energy[quiet_plan.choices["drum patterns"]] < energy[busy_plan.choices["drum patterns"]]
    assert quiet_plan.choices["growl character"] != busy_plan.choices["growl character"]
    assert quiet_plan.choices["washes"] != busy_plan.choices["washes"]
    assert _per_minute(busy, KIT_LAYERS) > GLITCH_KIT_RATIO * _per_minute(quiet, KIT_LAYERS)
    assert _per_minute(busy, {"bass"}) > _per_minute(quiet, {"bass"})
    assert _per_minute(busy) > _per_minute(quiet)
    assert len(busy.glitch.edits) > len(quiet.glitch.edits)  # type: ignore[union-attr]


@pytest.mark.parametrize("mode", GLITCH.modes)
def test_glitch_chord_loops_make_sense_in_the_mode(mode: str) -> None:
    """The triads of every loop and the drop chord are made of notes of the mode, and the loop picked belongs to it."""
    allowed = _pitches(Scale(0, mode))
    chords = [chord for loop in scape_glitch.LOOPS[mode] for chord in loop.chords]
    chords.append(scape_glitch.DROP_CHORDS[mode])
    for offset, quality in chords:
        triad = {offset % 12, (offset + (3 if quality == 0 else 4)) % 12, (offset + 7) % 12}
        assert triad <= allowed, (mode, offset, quality)
    _, base = demo("glitch")
    plan = replace(base, scale=Scale(base.scale.root, mode), character=BUSY, choices={}, notes=[])
    score = scape_glitch.compose(plan)
    assert plan.choices["chord loop"] in {loop.name for loop in scape_glitch.LOOPS[mode]}
    assert GLITCH.tempo[0] <= score.tempo <= GLITCH.tempo[1]
    assert mode in SCALES


def test_glitch_states_the_motif_after_the_drop_in_the_key_of_the_day() -> None:
    """The strings play the motif of the room from C4 up, in the notes of the day's scale."""
    for quiet in (True, False):
        score, plan = _made("glitch", quiet=quiet)
        expected = harmony.motif_notes(plan, plan.motif())
        assert min(midi for midi, _ in expected) >= MOTIF_LOWEST
        assert {midi % 12 for midi, _ in expected} <= _pitches(plan.scale)
        played = {n.midi for n in score.notes if n.layer == "str"}
        assert {float(midi) for midi, _ in expected} <= played


def test_glitch_sections_sit_on_phrase_lines_and_every_peak_opens_with_a_drop() -> None:
    """Sections start on multiples of 8 bars (4 in a short piece) and a peak begins with the big stab of a drop (or the distant boom of a night one)."""
    for quiet in (True, False):
        score, plan = _made("glitch", quiet=quiet)
        bar = 4.0 * 60.0 / score.tempo
        phrase = (
            FORM_BARS if plan.length / (FORM_BARS * bar) >= MIN_FORM_CELLS else SHORT_FORM_BARS
        ) * bar
        sections = plan.sections(bar)
        assert all(abs(a / phrase - round(a / phrase)) < 1e-6 for a, *_ in sections)
        bars = int((plan.length - 5.0) / bar)
        for start, _, role, _ in sections:
            d = round(start / bar)
            if role == CLIMAX and d >= 4 and d + 4 < bars:
                # a short piece can put a peak in the night, whose drop is the distant boom
                assert any(
                    n.voice in {"glitch:stab_big", "glitch:boomfar"} and abs(n.t - start) < 0.05
                    for n in score.notes
                ), (quiet, start)


def test_glitch_strings_are_led_from_chord_to_chord() -> None:
    """The upper strings move less than a placement that ignores the chord before."""
    chords = [scape_glitch._Chord(0, 2, 36, 48, 0), scape_glitch._Chord(2, 2, 41, 53, 0)]
    first = scape_glitch._lead_strings(chords[0], [], 3)
    led = scape_glitch._lead_strings(chords[1], first, 3)
    plain = scape_glitch._lead_strings(chords[1], [], 3)
    assert sum(abs(a - b) for a, b in zip(sorted(led), first, strict=False)) <= sum(
        abs(a - b) for a, b in zip(sorted(plain), first, strict=False)
    )


# ---- Cinematic Glitch: modes and drum families

WEEK = (0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95)
FORTNIGHT = WEEK + tuple(reversed(WEEK))
HEAVY_TEMPO = (54.0, 60.0)
# a fill is at least this many hits in the last half bar before a new section
FILL_HITS = 6
# the kit of a quiet 5 minute piece plays in only about four bars, of which at least this many differ
BAR_SIGNATURES = 3


def _day_of(
    index: int, energy: float | None, mode: scape_glitch.Mode | None = None
) -> tuple[Score, Plan]:
    """Return the score and the plan of the nth day of a room; its character has this energy, or is the demo's own."""
    _, base = demo("glitch")
    assert base.day is not None
    character = base.character
    if energy is not None:
        level = LOW if energy < 0.34 else (HIGH if energy > 0.67 else NEUTRAL.level)
        character = replace(NEUTRAL, energy=energy, level=level, shape=ALL_DAY, stretches=3)
    day = base.day + timedelta(days=index)
    scale = scale_of_day(GLITCH, HOME, day, character)
    plan = replace(base, character=character, scale=scale, day=day, choices={}, notes=[])
    with patch.object(scape_glitch, "MODES", (mode,) if mode else scape_glitch.MODES):
        return scape_glitch.compose(plan), plan


def _kit_hits(score: Score, a: float, b: float) -> int:
    kit = KIT_LAYERS | {"fx"}
    return len(
        [n for n in score.notes if n.layer in kit and a <= n.t < b and n.midi in {0.0, 36.0}]
    )


def test_glitch_drum_families_are_drawn_on_the_grid_with_variants() -> None:
    """There are at least six families of three or four variants, each lane as long as its pattern, every mode has some."""
    families = glitch_kits.FAMILIES
    assert len(families) >= GLITCH_FAMILIES
    for family in families:
        assert set(family.patterns) == set(glitch_kits.KINDS)
        for variants in family.patterns.values():
            assert 3 <= len(variants) <= 4
            for pattern in variants:
                steps = pattern["span"] * glitch_kits.STEPS
                assert len(pattern["hats"]) == steps
                lanes = ("kick", "snare", "ghost", "rim", "ticks", "open", "toms")
                assert all(0 <= s < steps for lane in lanes for s, _ in pattern[lane])
        assert family.fills
    for mode in scape_glitch.MODES:
        assert sum(mode.name in family.modes for family in families) >= 2


def test_glitch_a_fortnight_of_characters_rotates_modes_and_drum_families() -> None:
    """From quiet to busy days the mode and the family vary, and at least four families are heard."""
    made = [_day_of(i, e) for i, e in enumerate(FORTNIGHT)]
    modes = {plan.choices["mode"] for _, plan in made}
    families = {plan.choices["drum patterns"] for _, plan in made}
    assert len(modes) >= 2
    assert len(families) >= 4
    for score, plan in made:
        mode = next(m for m in scape_glitch.MODES if m.name == plan.choices["mode"])
        assert mode.tempo[0] <= score.tempo <= mode.tempo[1]
        assert plan.choices["drum patterns"] in {
            f.name for f in glitch_kits.FAMILIES if mode.name in f.modes
        }


def test_glitch_the_heavy_mode_is_slow_and_sparser_than_the_driving_one() -> None:
    """Heavy plays between 54 and 60 beats per minute, with far fewer hits of the kit per minute than driving."""
    heavy_mode, _, driving_mode = scape_glitch.MODES
    assert heavy_mode.name == "heavy"
    assert driving_mode.name == "driving"
    # driving leaves the kit to the heat, so on a day of 0.2 energy it plays less than the floor of heavy
    for energy in (0.5, 0.8):
        heavy, _ = _day_of(3, energy, heavy_mode)
        driving, _ = _day_of(3, energy, driving_mode)
        assert HEAVY_TEMPO[0] <= heavy.tempo <= HEAVY_TEMPO[1]
        assert _per_minute(heavy, KIT_LAYERS) < _per_minute(driving, KIT_LAYERS)


def test_glitch_fills_land_on_the_phrase_end_before_a_new_section() -> None:
    """The last half bar before a section that the kit leads into is a roll of at least six hits."""
    found = 0
    for index, energy in enumerate((0.5, 0.8, 0.95, 0.95)):
        score, plan = _day_of(index, energy)
        bar, beat = 4.0 * 60.0 / score.tempo, 60.0 / score.tempo
        starts = [first for first, *_ in plan.sections(bar) if first > 0]
        for start in starts:
            before = _kit_hits(score, start - bar, start - 2.0 * beat)
            drop = any(
                n.voice == "glitch:stab_big" and abs(n.t - start) < 0.05 for n in score.notes
            )
            if before and not drop:
                assert _kit_hits(score, start - 2.0 * beat, start) >= FILL_HITS, (index, start)
                found += 1
    assert found


def test_glitch_a_piece_does_not_loop_one_bar() -> None:
    """The kit plays at least three different bars over a piece, whatever the family."""
    for index, energy in enumerate((0.15, 0.5, 0.9)):
        score, _ = _day_of(index, energy)
        bar, step = 4.0 * 60.0 / score.tempo, 60.0 / score.tempo / 8.0
        signatures: set[tuple[int, ...]] = set()
        for b in range(int(score.length / bar)):
            hits = tuple(
                round((n.t - b * bar) / step)
                for n in score.notes
                if n.layer in {"kick", "snare"} and b * bar <= n.t < (b + 1) * bar
            )
            if hits:
                signatures.add(hits)
        assert len(signatures) >= BAR_SIGNATURES, (index, energy)


@pytest.mark.parametrize("mode", scape_glitch.MODES, ids=lambda m: m.name)
def test_glitch_every_mode_keeps_the_night_empty_and_the_bass_before_the_drums(
    mode: scape_glitch.Mode,
) -> None:
    """In every mode the night has no kit, growl or stabs, and the first growl note comes before the first kick."""
    score, plan = _day_of(2, None, mode)
    night_a, night_b = next((a, b) for n, a, b in plan.arc() if n == "night")
    assert not [
        n
        for n in score.notes
        if night_a <= n.t < night_b and n.layer in KIT_LAYERS | {"bass", "stab"}
    ]
    assert min(n.t for n in score.notes if n.layer == "bass") < min(
        n.t for n in score.notes if n.layer == "kick"
    )
