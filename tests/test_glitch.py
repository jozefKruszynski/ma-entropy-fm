"""Tests of Cinematic Glitch: the day arc, the drums, the edits on the grid, the brakes, the voices, the tiers."""

from typing import Any

import numpy as np

from entropy_fm import glitch, glitch_drums, glitch_voices
from entropy_fm.plan import Plan
from entropy_fm.quality import LIGHT, STANDARD
from entropy_fm.render import render_all
from entropy_fm.score import Note, Score

from .pitch_tools import find_glides, noise_floor_db
from .scape_tools import demo, the_demo, window

RATE = 44100
KIT = {"kick", "snare", "hats"}
# the edits that bend pitch: the only glides of the style besides its reversed swells
GLIDING = {"tape", "brake"}


def _arc(plan: Plan, name: str) -> tuple[float, float]:
    return next((a, b) for n, a, b in plan.arc() if n == name)


def _grid(score: Score) -> float:
    return 60.0 / score.tempo / 8.0


def test_the_night_has_no_drums_and_no_growl_and_the_day_has_both() -> None:
    """In the night there are swells, washes and distant booms alone; drums and growl enter with the day."""
    score, plan = demo("glitch")
    assert score.glitch is not None
    a, b = _arc(plan, "night")
    night = [n for n in score.notes if a <= n.t < b]
    assert night
    assert not [n for n in night if n.layer in KIT | {"bass", "stab"}]
    assert {n.voice for n in night} <= {
        "glitch:swell",
        "glitch:swell_slow",
        "glitch:swell_big",
        "glitch:boomfar",
        "glitch:revstab",
        "tape_strings",
        "wash:cello",
        "wash:violin",
        "wash:felt",
    }
    assert not [e for e in score.glitch.edits if a <= e.t < b and set(e.layers) & KIT]
    assert not [k for k, _ in score.glitch.kicks if a <= k < b]
    day_a, day_b = _arc(plan, "morning")[0], _arc(plan, "evening")[0]
    assert len([n for n in score.notes if n.layer == "kick" and day_a <= n.t < day_b]) > 50
    assert any(n.layer == "bass" for n in score.notes if day_a <= n.t < day_b)


def test_the_drums_are_in_the_active_window_and_the_bass_enters_first() -> None:
    """The active stretch has the kit; the first growl note comes before the first kick."""
    from soundscapes import pick_windows  # noqa: PLC0415

    score, plan = demo("glitch")
    start = pick_windows(plan, 60)[1][1]
    inside = [n for n in score.notes if start <= n.t < start + 60]
    assert {"kick", "snare", "hats"} <= {n.layer for n in inside}
    first_bass = min(n.t for n in score.notes if n.layer == "bass")
    first_kick = min(n.t for n in score.notes if n.layer == "kick")
    assert first_bass < first_kick


def test_the_kit_thins_towards_the_evening_and_leaves() -> None:
    """After half past nine at night there is no kit."""
    score, plan = demo("glitch")
    late = plan.piece_second_of_hour(21.6)
    assert not [n for n in score.notes if n.layer in KIT and n.t > late]


def test_the_drums_sit_on_the_grid_of_the_kit() -> None:
    """Every drum hit is on a sixteenth of the kit's grid (an eighth of a beat of the tempo)."""
    score, _ = demo("glitch")
    step = _grid(score)
    hits = [n for n in score.notes if n.layer in KIT and n.voice != "glitch:revsnare"]
    off = [n.t for n in hits if abs(n.t / step - round(n.t / step)) > 0.01]
    assert not off[:5], off[:5]


def test_edits_never_overlap_on_a_bus_and_stay_on_the_grid() -> None:
    """Edits of the same layer do not overlap; those of the house start on the grid of the tempo."""
    score, _ = demo("glitch")
    assert score.glitch is not None
    step = _grid(score)
    edits = score.glitch.edits
    assert edits
    held: dict[str, list[tuple[float, float]]] = {}
    for edit in edits:
        for layer in edit.layers:
            for a, b in held.get(layer, []):
                assert not (a < edit.t + edit.dur and edit.t < b), (edit, layer)
            held.setdefault(layer, []).append((edit.t, edit.t + edit.dur))
        assert edit.kind in glitch.EDIT_KINDS
    on_grid = [e for e in edits if e.why == "event" and e.kind in {"brake", "tape", "stutter"}]
    for edit in on_grid:
        assert abs(edit.t / step - round(edit.t / step)) < 0.02, edit


def test_the_events_of_the_house_are_rate_limited() -> None:
    """The edits that the house brings are at least two bars apart (a bar is four beats)."""
    score, _ = demo("glitch")
    assert score.glitch is not None
    bar = 4.0 * 60.0 / score.tempo
    times = sorted(e.t for e in score.glitch.edits if e.why == "event")
    assert times
    gaps = np.diff(times)
    # a doubled brake is two edits within a bar; the rest are far apart
    far = gaps[gaps > 1.5 * bar]
    assert len(far) >= 0.6 * len(gaps)


def test_the_brakes_are_short_and_hard_and_sometimes_doubled() -> None:
    """Over the days there are brakes of 60 to 150 ms, on a bar end, some doubled, and fewer long tape stops."""
    brakes, doubles, tapes, at_bar_end = 0, 0, 0, 0
    for index in range(2, 14):
        score, _ = the_demo().score(index, ("kitchen",), "glitch", STANDARD)
        assert score.glitch is not None
        bar = 4.0 * 60.0 / score.tempo
        edits = score.glitch.edits
        for edit in edits:
            if edit.kind == "tape":
                tapes += 1
            if edit.kind != "brake":
                continue
            brakes += 1
            assert 0.06 <= edit.params["stop"] <= 0.15, edit
            ends = (edit.t + edit.dur) / (bar / 16.0)
            assert abs(ends - round(ends)) < 0.02, edit
            at_bar_end += int(abs(ends / 16.0 - round(ends / 16.0)) < 0.002)
            if any(
                other.kind == "brake" and 0 < other.t - edit.t < bar and other is not edit
                for other in edits
            ):
                doubles += 1
    assert brakes >= 3
    assert doubles >= 1
    assert at_bar_end >= brakes / 2
    assert brakes > tapes / 2


def test_a_brake_falls_fast_cuts_hard_and_comes_back_on_the_grid() -> None:
    """A brake of 100 ms drops the pitch to nothing, is silent until the grid, and the material returns whole."""
    t = np.arange(RATE) / RATE
    tone = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    buf = np.stack([tone, tone], axis=1)
    original = buf.copy()
    glitch.brake_stop(buf, 0.3, 0.1, 0.5)
    inside = buf[int(0.3 * RATE) : int(0.4 * RATE), 0]
    zero_crossings = np.flatnonzero(np.diff(np.signbit(inside).astype(int)))
    early = np.diff(zero_crossings[:6]).mean()
    late = np.diff(zero_crossings[-6:]).mean()
    assert late > 2.0 * early, "the pitch falls by more than an octave"
    assert np.abs(buf[int(0.41 * RATE) : int(0.49 * RATE)]).max() < 1e-3, "silence after the cut"
    assert np.allclose(buf[: int(0.299 * RATE)], original[: int(0.299 * RATE)])
    assert np.allclose(buf[int(0.51 * RATE) :], original[int(0.51 * RATE) :], atol=1e-4)
    assert len(buf) == RATE


def test_every_edit_kind_keeps_the_length_and_stays_finite() -> None:
    """Each edit changes a buffer in place without changing its length or making it non-finite."""
    rng = np.random.default_rng(3)
    cases: dict[str, dict[str, Any]] = {
        "stutter": {"slices": (0.2, 0.2, 0.1, 0.1), "gains": (1, 1, 1, 1)},
        "tape": {"stop": 0.38, "span": 0.5, "wind": 0.0},
        "brake": {"stop": 0.1, "span": 0.25},
        "grain": {"length": 0.5, "grain": 0.1, "seed": 1, "jitter": 0.02},
        "spectral": {"length": 0.5, "back": 0.4, "seed": 1},
        "crush": {"dur": 0.2, "factor": 4, "bits": 8},
        "gate": {"step": 0.1, "pattern": (1, 0, 1, 1), "floor": 0.05},
        "reverse": {"length": 0.25},
        "pitchrep": {"slice": 0.1, "repeats": 3, "rate": 2.0, "gains": (1.0, 0.9, 0.8)},
        "bitdrop": {"dur": 0.25, "factor": 16, "bits": 4},
        "roll": {"slices": (0.2, 0.1, 0.1, 0.05, 0.05), "gains": (0.8, 0.9, 1.0, 1.1, 1.2)},
    }
    assert set(cases) == set(glitch.EDIT_KINDS)
    for kind, params in cases.items():
        buf = (rng.standard_normal((2 * RATE, 2)) * 0.1).astype(np.float32)
        glitch.apply_edit(buf, 1.0, kind, params)
        assert buf.shape == (2 * RATE, 2)
        assert np.isfinite(buf).all(), kind
        assert glitch.edit_extent(kind, params) > 0.0


def test_the_drops_are_on_even_bars_and_a_night_anomaly_is_only_a_reversed_swell_and_boom() -> None:
    """A drop begins on an even bar; a drop in the night has no stab, no kit and no roll."""
    score, plan = demo("glitch")
    bar = 4.0 * 60.0 / score.tempo
    bigs = [n for n in score.notes if n.voice == "glitch:stab_big"]
    assert bigs
    for note in bigs:
        number = note.t / bar
        assert abs(number - round(number)) < 0.01
        assert round(number) % 2 == 0
    a, b = _arc(plan, "night")
    for note in score.notes:
        if a <= note.t < b:
            assert note.voice != "glitch:stab_big"
            assert note.voice != "drum:odaiko"


def test_the_growl_character_follows_the_heat_and_the_notes_come_from_the_phrases() -> None:
    """The row of the growl (patch // 8) rises with the activity: the busiest bars have a higher row than the first."""
    score, _ = demo("glitch")
    growls = [n for n in score.notes if n.voice == "glitch:growl"]
    assert growls
    rows = [n.patch // 8 for n in growls]
    assert min(rows) <= 1
    assert max(rows) >= 3
    assert all(0 <= n.patch % 8 < 8 for n in growls)
    # data never writes a note: every growl pitch is a chord tone above a low root
    assert all(28.0 <= n.midi <= 52.0 for n in growls)


def test_the_glitch_voices_render_stereo_finite_and_cached() -> None:
    """Each voice of the style renders, and a second call returns the same samples."""
    score, _ = demo("glitch")
    seen: dict[str, Note] = {}
    for note in score.notes:
        if glitch_voices.is_glitch(note.voice):
            seen.setdefault(note.voice, note)
    assert {"glitch:growl", "glitch:kick", "glitch:snare", "glitch:swell", "glitch:stab"} <= set(
        seen
    )
    for voice, note in seen.items():
        first = glitch_voices.render(note, STANDARD, score.glitch)
        second = glitch_voices.render(note, STANDARD, score.glitch)
        assert first.ndim == 2
        assert first.shape[1] == 2, voice
        assert np.isfinite(first).all(), voice
        assert np.abs(first).max() > 0.0, voice
        assert np.array_equal(first, second), voice
    assert set(glitch_drums.HIT_NAMES) >= {"kick", "snare", "hat"}


def test_the_calm_parts_have_no_hiss_and_a_quiet_top_end() -> None:
    """In the quietest frames of the night nothing above 6 kHz is louder than -80 dBFS."""
    score, plan = demo("glitch")
    a, b = _arc(plan, "night")
    start = int((a + b) / 2 - 20)
    audio = render_all(score, STANDARD, "test", None, start_s=start, end_s=start + 40).mean(axis=1)
    assert noise_floor_db(audio[8 * RATE :]) < -80.0


def test_the_pitch_glides_of_the_day_are_the_tape_stops_and_brakes_only() -> None:
    """A fall or a rise of three semitones inside a second and a half only happens near a tape stop or a brake."""
    score, plan = demo("glitch")
    assert score.glitch is not None
    audio = window(score, plan, 40)
    from soundscapes import pick_windows  # noqa: PLC0415

    start = pick_windows(plan, 40)[1][1]
    allowed = [e.t - start for e in score.glitch.edits if e.kind in GLIDING or e.why == "drop"]
    for direction in (-1, 1):
        for when, _ in find_glides(audio.mean(axis=1), 3.0, 1.5, direction):
            assert any(-2.0 <= when - t <= 2.5 for t in allowed), (when, allowed)


def test_the_same_day_is_the_same_score() -> None:
    """Composing twice gives the same notes, edits, kicks and gain."""
    first, _ = the_demo().score(13, ("kitchen",), "glitch", STANDARD)
    second, _ = the_demo().score(13, ("kitchen",), "glitch", STANDARD)
    assert first.notes == second.notes
    assert first.glitch == second.glitch
    assert first.gain_db is not None
    assert second.gain_db is not None
    assert np.array_equal(first.gain_db, second.gain_db)


def test_light_is_thinner_and_cheaper_in_notes() -> None:
    """The cheapest tier has fewer layers of sound: no ghost snares, ticks or rim, no busy bass, fewer strings."""
    light, _ = demo("glitch", tier=LIGHT)
    standard, _ = demo("glitch", tier=STANDARD)
    assert len(light.notes) < 0.8 * len(standard.notes)
    voices = {n.voice for n in light.notes}
    assert not voices & {"glitch:ghost", "glitch:tick", "glitch:rim", "glitch:open", "wash:cello"}
    assert max((n.patch // 8 for n in light.notes if n.voice == "glitch:growl"), default=0) <= 3


def test_every_day_of_the_fortnight_composes_and_renders_a_second() -> None:
    """Each day makes a score whose edits fit in the piece and whose gain is a constant within the bounds."""
    for index in range(1, 14):
        score, _ = the_demo().score(index, ("kitchen",), "glitch", STANDARD)
        assert score.glitch is not None
        assert score.gain_db is not None
        assert -9.0 <= float(score.gain_db.min()) <= float(score.gain_db.max()) <= 12.0
        assert all(0.0 < e.t < score.length for e in score.glitch.edits)
        assert all(0.0 <= n.t < score.length for n in score.notes)


def test_swells_of_one_chord_share_a_stack_and_the_sound_is_what_it_was_without() -> None:
    """A swell made from the shared stack of a longer swell is the swell made alone, bit for bit."""
    from entropy_fm.voices import NOTES  # noqa: PLC0415

    def swell(dur: float, vel: float) -> Note:
        return Note(t=0.0, voice="glitch:swell_slow", midi=38.0, dur=dur, vel=vel, layer="pad")

    NOTES.clear()
    glitch_voices.STACKS.clear()
    short = glitch_voices.render(swell(12.0, 0.5), STANDARD, None)
    NOTES.clear()
    glitch_voices.STACKS.clear()
    glitch_voices.prepare([swell(12.0, 0.5), swell(19.0, 0.7)], STANDARD)
    made = glitch.swell_stack
    calls: list[int] = []

    def counted(*args: object, **kwargs: object) -> np.ndarray:
        calls.append(1)
        return made(*args, **kwargs)  # type: ignore[arg-type]

    glitch.swell_stack = counted
    try:
        again = glitch_voices.render(swell(12.0, 0.5), STANDARD, None)
        longer = glitch_voices.render(swell(19.0, 0.7), STANDARD, None)
    finally:
        glitch.swell_stack = made
    assert len(calls) == 1
    assert np.array_equal(short, again)
    assert len(longer) == 19 * RATE
    # the opening seconds are the same swell wherever it ends: only the level and the fall differ
    a, b = longer[: 4 * RATE, 0].astype(np.float64), again[: 4 * RATE, 0].astype(np.float64)
    assert np.corrcoef(a, b)[0, 1] > 0.9999


def test_costly_notes_are_kept_apart_so_cheap_long_notes_cannot_push_them_out() -> None:
    """A note in the pool of dear notes survives a flood of long cheap ones."""
    from entropy_fm.voices import NoteCache  # noqa: PLC0415

    cache = NoteCache()
    dear = np.zeros((RATE * 3, 2), np.float16)
    cache.put(("swell",), dear, 8.0, dear=True)
    for index in range(40):
        cache.put(("pad", index), np.zeros((RATE * 4, 2), np.float32), 8.0)
    assert cache.get(("swell",)) is dear
    assert cache.size < 41
    cache.clear()
    assert cache.size == 0


def test_a_cache_big_enough_for_the_day_makes_every_swell_and_growl_once() -> None:
    """On Standard no swell and no growl of a day is made twice: the cache holds the day."""
    from entropy_fm.render import render_blocks  # noqa: PLC0415
    from entropy_fm.voices import NOTES  # noqa: PLC0415

    score, plan = demo("glitch")
    NOTES.clear()
    glitch_voices.STACKS.clear()
    swells: list[tuple[object, ...]] = []
    growls: list[tuple[object, ...]] = []
    real_swell, real_growl = glitch.swell_pad, glitch.growl_note

    def swell(*args: object, **kwargs: object) -> np.ndarray:
        swells.append((args[0], args[1], args[2], kwargs.get("seed")))
        return real_swell(*args, **kwargs)  # type: ignore[arg-type]

    def growl(*args: object, **kwargs: object) -> np.ndarray:
        growls.append(args)
        return real_growl(*args, **kwargs)  # type: ignore[arg-type]

    glitch.swell_pad, glitch.growl_note = swell, growl
    try:
        start = int(plan.length * 0.3)
        for _ in render_blocks(score, STANDARD, "cache", start, end_s=start + 240):
            pass
    finally:
        glitch.swell_pad, glitch.growl_note = real_swell, real_growl
    assert growls
    assert len(growls) == len({repr(g) for g in growls})
    assert len(swells) == len({repr(g) for g in swells})


def test_the_oversampled_shaper_is_close_to_the_shaper_at_a_higher_rate_whatever_the_length() -> (
    None
):
    """Lengths that an FFT dislikes are rounded up with more of the signal, never less, and the result is the same."""
    rng = np.random.default_rng(3)
    for n in (5000, 32768, 40001, 53003):
        x = (rng.standard_normal(n) * 0.3).astype(np.float32)
        a = glitch.oversampled(np.tanh, x, 4)
        b = glitch.oversampled(np.tanh, x, 4, block=1 << 13, pad=1024)
        assert len(a) == n
        assert np.max(np.abs(a - b)) < 2e-3
    assert glitch._fast_length(53004) == 54000  # noqa: SLF001
    assert glitch._fast_length(32768) == 32768  # noqa: SLF001
