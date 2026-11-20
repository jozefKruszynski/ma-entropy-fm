"""Tests of the busier Cinematic Glitch: edits every bar or two, the new kinds, and edits of the instruments."""

from itertools import pairwise

import numpy as np

from entropy_fm import glitch
from entropy_fm.score import Edit, Score

from .scape_tools import demo

RATE = 44100
HOUSE = ("kitchen", "living")
INSTRUMENTS = {"str", "bass", "stab"}
# the days of the fortnight that the tests look at
DAYS = range(2, 14)
# on a busy day (driving mode) an edit comes at least this often in the stretches where the drums play: one per 2.25 bars
BUSY_EDITS_PER_BAR = 1.0 / 2.25
# at the least, on the quietest day with a kit (heavy mode), one per 3.5 bars; at the most, on any day, one per bar
QUIET_EDITS_PER_BAR = 1.0 / 3.5
MOST_EDITS_PER_BAR = 1.0
# the share of the edits that sit on an instrument bus alone, on a busy day
INSTRUMENT_SHARE = 0.15
KINDS_IN_A_BUSY_PIECE = 5
MOST_IN_A_ROW = 2
# the longest gap between two kicks that is still the same stretch of drums
STRETCH_GAP_BARS = 2
# a bin within this many Hz of an overtone is the tone, and the bit drop must have this times the grit of the crush
GRIT_BAND_HZ = 25.0
GRIT_FACTOR = 5.0


def _drum_stretches(score: Score) -> list[tuple[float, float]]:
    """Return the stretches (start, end) where the kit plays, whose kicks are less than two bars apart."""
    bar = 4.0 * 60.0 / score.tempo
    kicks = sorted(n.t for n in score.notes if n.layer == "kick")
    found: list[list[float]] = []
    for t in kicks:
        if found and t - found[-1][1] <= STRETCH_GAP_BARS * bar:
            found[-1][1] = t
        else:
            found.append([t, t])
    return [(a, b + bar) for a, b in found]


def _edits_per_bar(score: Score) -> float:
    """Return the edits (drops left out) per bar inside the stretches of drums."""
    assert score.glitch is not None
    bar = 4.0 * 60.0 / score.tempo
    stretches = _drum_stretches(score)
    bars = sum(b - a for a, b in stretches) / bar
    edits = [
        e for e in score.glitch.edits if e.why != "drop" and any(a <= e.t < b for a, b in stretches)
    ]
    return len(edits) / bars


def _busy_day() -> tuple[Score, int]:
    """Return the score of the day of the fortnight with the most kicks, whole house, and its index."""
    best = max(
        DAYS, key=lambda i: sum(n.layer == "kick" for n in demo("glitch", i, HOUSE)[0].notes)
    )
    return demo("glitch", best, HOUSE)[0], best


def _quiet_day() -> Score:
    """Return the score of the day with the fewest kicks that still has a kit."""
    kits = {i: sum(n.layer == "kick" for n in demo("glitch", i, HOUSE)[0].notes) for i in DAYS}
    return demo("glitch", min((i for i in kits if kits[i] > 200), key=lambda i: kits[i]), HOUSE)[0]


def test_a_busy_day_is_cut_every_bar_or_two_where_the_drums_play() -> None:
    """The stretches of drums of the busiest day have an edit about every one to two bars, and never more than one a bar."""
    score, _ = _busy_day()
    assert BUSY_EDITS_PER_BAR <= _edits_per_bar(score) <= MOST_EDITS_PER_BAR


def test_every_day_with_a_kit_is_cut_at_least_every_few_bars() -> None:
    """Even the quietest day with a kit has an edit about every two to four bars, and never more than one a bar."""
    assert QUIET_EDITS_PER_BAR <= _edits_per_bar(_quiet_day()) <= MOST_EDITS_PER_BAR


def test_a_busy_piece_mixes_at_least_five_kinds_and_none_runs_on() -> None:
    """Five different kinds or more, and the same kind never plays more than twice in a row."""
    for index in DAYS:
        score, _ = demo("glitch", index, HOUSE)
        assert score.glitch is not None
        ordered = sorted(score.glitch.edits, key=lambda e: e.t)
        run = 1
        for before, after in pairwise(ordered):
            run = run + 1 if before.kind == after.kind else 1
            assert run <= MOST_IN_A_ROW, (index, after.t, after.kind)
    busy, _ = _busy_day()
    assert busy.glitch is not None
    assert len({e.kind for e in busy.glitch.edits}) >= KINDS_IN_A_BUSY_PIECE


def test_the_new_kinds_are_dealt() -> None:
    """A reversed chunk, a pitch repeat, a bit drop and a roll all play on a busy day."""
    score, _ = _busy_day()
    assert score.glitch is not None
    kinds = {e.kind for e in score.glitch.edits}
    assert {"reverse", "pitchrep", "bitdrop", "roll"} <= kinds


def test_a_meaningful_share_of_the_edits_is_on_the_instruments() -> None:
    """The strings, the growl and the stabs get edits of their own, apart from those of the drums."""
    score, _ = _busy_day()
    assert score.glitch is not None
    edits = score.glitch.edits
    own = [e for e in edits if set(e.layers) <= INSTRUMENTS]
    assert len(own) / len(edits) >= INSTRUMENT_SHARE
    assert {layer for e in own for layer in e.layers} >= {"str", "bass"}


def test_no_two_edits_overlap_on_one_bus_on_any_day() -> None:
    """The edits of a bus follow one another, whatever the day brings."""
    for index in DAYS:
        score, _ = demo("glitch", index, HOUSE)
        assert score.glitch is not None
        held: dict[str, list[tuple[float, float]]] = {}
        for edit in sorted(score.glitch.edits, key=lambda e: e.t):
            for layer in edit.layers:
                for a, b in held.get(layer, []):
                    assert not (a < edit.t + edit.dur and edit.t < b), (index, edit, layer)
                held.setdefault(layer, []).append((edit.t, edit.t + edit.dur))


def test_the_edits_of_the_groove_are_on_the_grid_and_the_night_has_none() -> None:
    """The groove starts on the grid of the tempo and only where the kit plays."""
    score, _ = _busy_day()
    assert score.glitch is not None
    step = 60.0 / score.tempo / 8.0
    stretches = _drum_stretches(score)
    groove = [e for e in score.glitch.edits if e.why in {"groove", "house"}]
    assert groove
    for edit in groove:
        assert abs(edit.t / step - round(edit.t / step)) < 0.02, edit
        assert any(a <= edit.t < b for a, b in stretches), edit


def _tone(hz: float, seconds: float = 2.0) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return np.repeat((0.4 * np.sin(2.0 * np.pi * hz * t))[:, None], 2, axis=1).astype(np.float32)


def _peak_hz(chunk: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(chunk[:, 0] * np.hanning(len(chunk))))
    return float(np.argmax(spectrum) * RATE / len(chunk))


def test_a_reversed_chunk_is_the_chunk_played_backwards() -> None:
    """The middle of a reversed beat is the original read from the other end."""
    rng = np.random.default_rng(1)
    buf = (rng.standard_normal((2 * RATE, 2)) * 0.1).astype(np.float32)
    original = buf.copy()
    glitch.apply_edit(buf, 1.0, "reverse", {"length": 0.5})
    a, b = int(1.0 * RATE), int(1.5 * RATE)
    middle = slice(a + 400, b - 400)
    assert np.allclose(buf[middle], original[a:b][::-1][400 : b - a - 400], atol=1e-6)
    assert np.allclose(buf[b + 300 :], original[b + 300 :])


def test_a_pitch_repeat_jumps_an_octave_and_holds_it() -> None:
    """A tone repeated at twice the speed sounds an octave up, at half the speed an octave down."""
    for rate, hz in ((2.0, 880.0), (0.5, 220.0)):
        buf = _tone(440.0)
        params = {"slice": 0.2, "repeats": 4, "rate": rate, "gains": (1.0, 1.0, 1.0, 1.0)}
        glitch.apply_edit(buf, 0.5, "pitchrep", params)
        start = int(0.55 * RATE)
        assert abs(_peak_hz(buf[start : start + int(0.7 * RATE)]) - hz) < 6.0, rate


def _grit(x: np.ndarray, base: float) -> float:
    """Return the share of the energy of a stretch that lies away from the overtones of a tone."""
    spectrum = np.abs(np.fft.rfft(x[:, 0] * np.hanning(len(x)))) ** 2
    hz = np.arange(len(spectrum)) * RATE / len(x)
    near = np.abs(hz / base - np.round(hz / base)) * base < GRIT_BAND_HZ
    return float(spectrum[~near].sum() / spectrum.sum())


def test_a_bit_drop_is_harsher_than_a_crush() -> None:
    """On a tone with overtones a bit drop puts far more of its energy between the overtones than the crush does."""
    source = np.zeros((2 * RATE, 2), np.float32)
    for k in range(1, 9):
        source += _tone(440.0 * k, 2.0) * np.float32(0.5 / k)
    crushed, dropped = source.copy(), source.copy()
    glitch.apply_edit(crushed, 1.0, "crush", {"dur": 0.3, "factor": 4, "bits": 8})
    glitch.apply_edit(dropped, 1.0, "bitdrop", {"dur": 0.3, "factor": 16, "bits": 4})
    a, b = int(1.05 * RATE), int(1.25 * RATE)
    assert _grit(dropped[a:b], 440.0) > GRIT_FACTOR * _grit(crushed[a:b], 440.0)
    rms = float(np.sqrt(np.mean(source[a:b] ** 2)))
    assert 0.6 * rms < float(np.sqrt(np.mean(dropped[a:b] ** 2))) < 1.6 * rms


def test_a_roll_repeats_shorter_and_louder_slices() -> None:
    """The slices of every roll of the groove get shorter and their gains rise."""
    score, _ = _busy_day()
    assert score.glitch is not None
    rolls = [e for e in score.glitch.edits if e.kind == "roll" and e.why in {"groove", "house"}]
    assert rolls
    for edit in rolls:
        assert list(edit.params["slices"]) == sorted(edit.params["slices"], reverse=True)
        assert list(edit.params["gains"]) == sorted(edit.params["gains"])
        assert isinstance(edit, Edit)


def test_the_hottest_peaks_go_all_the_way_for_at_most_a_few_bars() -> None:
    """The extreme sections are the hottest peaks or plateaus, capped in length, one or two by the energy, never where it is calm."""
    from entropy_fm.plan import BUILD, CLIMAX, PLATEAU  # noqa: PLC0415
    from entropy_fm.scape_glitch import EXTREME_MAX_BARS, _extreme_bars  # noqa: PLC0415

    bar = 2.0
    sections = [
        (0.0, 40.0, BUILD, 0.5),
        (40.0, 120.0, CLIMAX, 1.0),
        (120.0, 160.0, PLATEAU, 0.8),
        (160.0, 200.0, CLIMAX, 1.0),
    ]
    per_bar = [0.2] * 20 + [0.9] * 40 + [0.5] * 20 + [0.7] * 21
    quiet = _extreme_bars(sections, per_bar, bar, 0.0)
    busy = _extreme_bars(sections, per_bar, bar, 1.0)
    assert quiet == set(range(20, 20 + EXTREME_MAX_BARS))
    assert busy == quiet | set(range(80, 80 + EXTREME_MAX_BARS))
    assert not busy & set(range(20))
    calm = [0.3] * len(per_bar)
    assert _extreme_bars(sections, calm, bar, 1.0) == set()
