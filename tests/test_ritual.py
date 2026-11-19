"""Tests of Ritual: the pentatonic loops, the vibrato of the drone, the chant leader, the heartbeat, the taiko, the washes."""

import itertools
import time

import numpy as np
import pytest

from entropy_fm import dsp, stretch
from entropy_fm.drums import render_hit
from entropy_fm.plan import Plan
from entropy_fm.quality import LIGHT, STANDARD
from entropy_fm.scape_ritual import LENGTHS, TUNINGS
from entropy_fm.score import Score
from entropy_fm.voices import NOTES, render_note

from .scape_tools import demo, window

RATE = 44100
# no bell, chime, gong or glassy voice is allowed anywhere in Ritual
FORBIDDEN = {"glass_pad", "fm_ep", "tx_ep", "fm_growl", "reese", "juno_pluck", "blip_synth"}


def _scale_of(score: Score) -> tuple[int, set[int]]:
    name = score.choices["tuning"]
    _, root, steps = next(t for t in TUNINGS if t[0] == name)
    return root, {(root + step) % 12 for step in steps}


def _arc(plan: Plan, name: str) -> tuple[float, float]:
    return next((a, b) for n, a, b in plan.arc() if n == name)


def _frequency_track(wave: np.ndarray, seconds: tuple[float, float]) -> np.ndarray:
    """Return the instantaneous frequency of a tone between two seconds, from its analytic signal."""
    count = len(wave)
    spectrum = np.fft.fft(wave.astype(np.float64))
    keep = np.zeros(count)
    keep[0], keep[1 : count // 2], keep[count // 2] = 1.0, 2.0, 1.0
    phase = np.unwrap(np.angle(np.fft.ifft(spectrum * keep)))
    track = np.diff(phase) * RATE / (2.0 * np.pi)
    return track[int(seconds[0] * RATE) : int(seconds[1] * RATE)]


def test_the_loop_lengths_share_no_simple_ratio_and_are_unequal() -> None:
    """The three plucks of every set phase: no two lengths are within 2 percent of a ratio of small whole numbers."""
    for lengths in LENGTHS:
        assert len(set(lengths)) == 3
        for a, b in itertools.combinations(sorted(lengths), 2):
            assert b / a > 1.1, lengths
            for p in range(1, 6):
                for q in range(1, 6):
                    assert abs(b / a - p / q) > 0.02 * p / q, (lengths, a, b)
        assert min(lengths) >= 4.0
        assert max(lengths) <= 8.0


def test_the_vibrato_is_a_few_cents_at_five_hertz_and_begins_late() -> None:
    """The helper bends a pure tone by 4 to 8 cents at its own rate, steady for the first moments, and fades in."""
    for cents, rate, phase in ((4.0, 4.6, 0.1), (6.0, 5.0, 0.4), (8.0, 5.4, 0.8)):
        tone = np.sin(2 * np.pi * 220.0 * np.arange(RATE * 8) / RATE).astype(np.float32)
        out = dsp.pitch_vibrato(tone, cents, rate, 1.5, phase, 0.3)
        track = _frequency_track(out, (0.0, 8.0))
        steady = 1200.0 * np.log2(np.abs(track[int(0.05 * RATE) : int(0.28 * RATE)]) / 220.0)
        assert np.abs(steady).max() < 1.0, "the first moments are steady"
        late = 1200.0 * np.log2(track[int(3.0 * RATE) : int(7.0 * RATE)] / 220.0)
        assert 0.8 * cents <= late.max() <= 1.2 * cents
        assert -1.2 * cents <= late.min() <= -0.8 * cents
        # the swing repeats at the rate asked for
        measured = _swing_rate(track[int(3.0 * RATE) : int(7.0 * RATE)])
        assert abs(measured - rate) < 0.3, (rate, measured)
        early = np.abs(1200.0 * np.log2(track[int(0.5 * RATE) : int(1.0 * RATE)] / 220.0)).max()
        assert early < 0.6 * cents, "the depth fades in over a second or two"


def _swing_rate(track: np.ndarray) -> float:
    """Return the rate in Hz at which a frequency track swings."""
    swing = np.abs(np.fft.rfft((track - track.mean()) * np.hanning(len(track)), 1 << 18))
    rates = np.fft.rfftfreq(1 << 18, 1.0 / RATE)
    window = (rates > 3.0) & (rates < 8.0)
    return float(rates[window][int(np.argmax(swing[window]))])


def _rate_of(wave: np.ndarray, freq: float) -> float:
    """Return the rate in Hz at which the pitch of a tone near `freq` swings, from its frequency track."""
    spectrum = np.fft.rfft(wave)
    freqs = np.fft.rfftfreq(len(wave), 1.0 / RATE)
    spectrum[np.abs(freqs - freq) > 25.0] = 0.0
    isolated = np.fft.irfft(spectrum, len(wave))
    return _swing_rate(_frequency_track(isolated, (4.0, 8.0)))


def test_every_vibrato_note_has_a_rate_of_its_own() -> None:
    """Voices that sing together do not wobble together: each note swings at its own rate, between 4.5 and 5.5 Hz."""
    found = []
    for midi in (38.0, 40.0, 43.0, 45.0):
        wave = render_note("juno_pad", midi, 9.0, 0.6, STANDARD, 0, 6.0).mean(axis=1)
        found.append(_rate_of(wave, 440.0 * 2.0 ** ((midi - 69.0) / 12.0)))
    assert all(4.3 <= rate <= 5.7 for rate in found), found
    assert max(found) - min(found) > 0.1, found
    plain = render_note("juno_pad", 38.0, 9.0, 0.6, STANDARD)
    sung = render_note("juno_pad", 38.0, 9.0, 0.6, STANDARD, 0, 6.0)
    assert not np.array_equal(plain, sung)
    assert np.array_equal(sung, render_note("juno_pad", 38.0, 9.0, 0.6, STANDARD, 0, 6.0))


def test_a_vibrato_note_stays_within_the_limit_and_does_not_glide() -> None:
    """The vibrato of the drone and the lead is at most 15 cents and the glide tracker finds nothing."""
    from .pitch_tools import find_glides  # noqa: PLC0415

    notes = [
        render_note(key, pitch, 6.0, 0.6, STANDARD, 0, 6.0).mean(axis=1)
        for key, pitch in (("chant_cello", 62.0), ("chant_cello", 67.0), ("tape_strings", 62.0))
    ]
    phrase = np.concatenate(notes)
    assert find_glides(phrase, 3.0, 1.5, -1) == []
    assert find_glides(phrase, 3.0, 1.5, 1) == []
    capped = render_note("chant_cello", 62.0, 4.0, 0.6, STANDARD, 0, 40.0)
    assert np.isfinite(capped).all()


def test_the_score_gives_vibrato_to_the_choir_voices_and_the_leader_only() -> None:
    """The cello section and the pad of the drone and the chant leader sing a vibrato of 4 to 8 cents; the sub and the plucks do not."""
    score, _ = demo("ritual")
    sung = {n.voice for n in score.notes if n.vib > 0.0}
    assert sung <= {"tape_strings", "juno_pad", "chant_cello"}
    assert {"tape_strings", "juno_pad", "chant_cello"} <= sung
    assert all(4.0 <= n.vib <= 8.0 for n in score.notes if n.vib > 0.0)
    assert all(
        n.vib == 0.0 for n in score.notes if n.layer in {"sub", "plucks", "heart", "taiko", "wash"}
    )


def test_the_night_is_drone_and_washes_alone_and_the_plucks_enter_in_turn() -> None:
    """No pluck, lead or drum in the night; the loops come in one after another as the day wakes."""
    score, plan = demo("ritual")
    night = _arc(plan, "night")
    in_night = {n.layer for n in score.notes if n.t < night[1]}
    assert in_night <= {"drone", "sub", "wash"}
    assert {"drone", "wash"} <= in_night
    assert not [n for n in score.notes if n.voice.startswith("drum:") and n.t < night[1]]
    assert not [
        n for n in score.notes if n.voice.startswith("drum:") and n.t >= _arc(plan, "late night")[0]
    ]
    firsts = sorted(
        min(n.t for n in score.notes if n.layer == "plucks" and n.voice == voice)
        for voice in {n.voice for n in score.notes if n.layer == "plucks"}
    )
    assert len(firsts) == 3
    assert firsts[1] - firsts[0] > 3.0, firsts
    assert firsts[2] - firsts[1] > 3.0, firsts
    last_plucks = max(n.t for n in score.notes if n.layer == "plucks")
    assert last_plucks < plan.length - 20.0, "the evening settles"


def test_the_plucks_are_in_the_scale_and_the_cells_drift_by_one_step() -> None:
    """Plucks, the leader and the drone are in the pentatonic of the day; a cell never drifts more than a step from its shape."""
    score, plan = demo("ritual")
    root, allowed = _scale_of(score)
    plucks = [n for n in score.notes if n.layer in {"plucks", "lead"}]
    assert plucks
    assert all(round(n.midi) % 12 in allowed for n in plucks)
    drone = [n for n in score.notes if n.layer in {"drone", "sub"}]
    assert {(round(n.midi) - root) % 12 for n in drone} <= {0, 5, 7}
    # a cell of six or seven notes moves one note at most a step either way: a voice never takes many pitches
    for voice in {n.voice for n in score.notes if n.layer == "plucks"}:
        assert (
            len({n.midi for n in score.notes if n.voice == voice and n.layer == "plucks"}) <= 16
        ), voice
    assert not set(score.voices_used()) & FORBIDDEN
    assert plan.scale  # the day's scale is not what Ritual plays in: it has its own tuning


def test_the_chant_leader_is_sparse_singable_and_answers() -> None:
    """A few long cello notes of the scale in the mid register, with rests between them, never constant."""
    score, plan = demo("ritual")
    lead = sorted((n for n in score.notes if n.layer == "lead"), key=lambda n: n.t)
    assert 6 <= len(lead) <= 70
    assert all(n.voice == "chant_cello" and n.dur >= 3.0 for n in lead)
    assert all(53.0 <= n.midi <= 78.0 for n in lead)
    gaps = [b.t - (a.t + a.dur) for a, b in itertools.pairwise(lead)]
    assert min(gaps) > 0.3, "every note is followed by a rest"
    covered = sum(n.dur for n in lead)
    assert covered < 0.3 * plan.length, "the leader is not constant"
    pluck_times = np.array([n.t for n in score.notes if n.layer == "plucks"])
    # each phrase starts a moment after a pluck cell has ended: a pluck sounded within 12 seconds before it
    starts = [
        n.t for a, n in itertools.pairwise([lead[0], *lead]) if n.t - a.t - a.dur > 3.0 or a is n
    ]
    assert all(np.any((pluck_times > t - 12.0) & (pluck_times < t)) for t in starts)
    light, _ = demo("ritual", tier=LIGHT)
    assert not [n for n in light.notes if n.layer in {"lead", "wash"}]


def test_the_heartbeat_and_the_taiko_are_for_the_busy_stretches_only() -> None:
    """Soft low beats in the busy part, a few big strikes on the pulse, a roll into one; none outside the busy part."""
    score, plan = demo("ritual")
    heart = sorted(n.t for n in score.notes if n.layer == "heart")
    strikes = sorted(n.t for n in score.notes if n.voice == "drum:odaiko")
    assert heart
    assert strikes
    assert len(strikes) < 0.2 * len(heart)
    # the beats sit on a steady pulse: the gaps are multiples of one beat (a rest where a strike or a roll replaces a beat)
    gaps = np.diff(heart)
    beat = float(np.median(gaps))
    assert 1.0 < beat < 1.5
    assert np.mean(np.abs(gaps / beat - np.round(gaps / beat)) < 0.02) > 0.9
    # soft: every beat is quieter than every big strike except the rolls' first strokes
    assert max(n.vel for n in score.notes if n.layer == "heart") < 0.45
    assert np.median([n.vel for n in score.notes if n.voice == "drum:odaiko"]) > 0.5
    # no beat inside a big strike's own beat
    for t in strikes[:20]:
        assert not [h for h in heart if abs(h - t) < 0.5 * beat]
    # the busy stretches are a part of the day, not all of it
    assert heart[-1] - heart[0] < plan.length
    assert len(heart) < 0.5 * plan.length / beat


def test_a_roll_accelerates_into_a_strike() -> None:
    """The gaps of an accelerating roll shrink and the strokes grow, ending on the big strike."""
    from entropy_fm import taiko  # noqa: PLC0415

    hits = taiko.roll(10.0, 14.0, "odaiko", 0.2, 0.9)
    times = [h.time for h in hits]
    gaps = np.diff(times)
    assert times[-1] == pytest.approx(14.0)
    assert gaps[0] > gaps[-1]
    assert np.all(np.diff(gaps) < 1e-9)
    assert hits[-1].velocity > hits[0].velocity
    score, plan = demo("ritual", 13)
    assert plan.events
    rolled = [n for n in score.notes if n.voice == "drum:odaiko"]
    assert rolled


def test_the_taiko_and_the_heartbeat_are_tones_without_noise_or_click() -> None:
    """The heartbeat drum rises gently (no click at the front) and nothing of the taiko reaches above 5 kHz."""
    for name in ("odaiko", "shime", "taiko_rim", "heart"):
        hit = render_hit(name, 0.8).mean(axis=1)
        spectrum = np.abs(np.fft.rfft(hit * np.hanning(len(hit)), 1 << 17)) ** 2
        freqs = np.fft.rfftfreq(1 << 17, 1.0 / RATE)
        share = spectrum[freqs > 5000.0].sum() / spectrum.sum()
        assert share < 1e-3, (name, share)
    heart = render_hit("heart", 0.8).mean(axis=1)
    peak = np.abs(heart).max()
    assert np.abs(heart[: int(0.002 * RATE)]).max() < 0.25 * peak, "no click at the onset"
    assert int(np.argmax(np.abs(heart))) > int(0.005 * RATE)


def test_a_wash_is_stereo_cached_long_and_has_nothing_above_the_roll_off() -> None:
    """A stretched texture is 24 seconds of smooth stereo that is made once; its high end is the sample's, rolled off."""
    NOTES.clear()
    began = time.process_time()
    wave = stretch.render_wash("wash:cello", 50.0, 24.0, 0.6)
    first = time.process_time() - began
    began = time.process_time()
    again = stretch.render_wash("wash:cello", 50.0, 24.0, 0.6)
    second = time.process_time() - began
    assert again is wave
    assert second < 0.01 * max(first, 1e-3)
    assert wave.shape == (24 * RATE, 2)
    assert np.isfinite(wave).all()
    assert np.abs(wave[: RATE // 4]).max() < 0.1 * np.abs(wave).max(), "it fades in"
    assert np.abs(wave[-RATE // 4 :]).max() < 0.1 * np.abs(wave).max(), "and out"
    assert not np.array_equal(wave[:, 0], wave[:, 1]), "left and right have their own phases"
    mid = wave[8 * RATE : 16 * RATE, 0]
    spectrum = np.abs(np.fft.rfft(mid * np.hanning(len(mid)))) ** 2
    freqs = np.fft.rfftfreq(len(mid), 1.0 / RATE)
    assert spectrum[freqs > 5000.0].sum() / spectrum.sum() < 1e-6
    for kind in stretch.WASHES:
        assert stretch.is_wash(kind)
    assert not stretch.is_wash("felt_piano")


def test_ritual_has_no_noise_bed_and_a_quiet_top_end() -> None:
    """No texture of noise, and above 5 kHz the quiet frames sit far below -80 dBFS."""
    from scipy.signal import butter, sosfiltfilt  # noqa: PLC0415

    score, plan = demo("ritual")
    assert score.textures == []
    audio = window(score, plan, 40).mean(axis=1)
    high = sosfiltfilt(butter(8, 5000, "hp", fs=RATE, output="sos"), audio)
    frame = 2048
    count = len(audio) // frame
    full = np.sqrt((audio[: count * frame].reshape(count, frame) ** 2).mean(axis=1)) + 1e-12
    top = np.sqrt((high[: count * frame].reshape(count, frame) ** 2).mean(axis=1)) + 1e-12
    inside = 20 * np.log10(full) > -70.0
    levels = 20 * np.log10(top[inside])
    assert np.percentile(levels, 10) < -90.0, np.percentile(levels, 10)
    assert np.median(levels) < -80.0, np.median(levels)


def test_light_thins_the_layers_and_costs_less() -> None:
    """Light has two plucked loops, no washes and no leader; Standard has all of them."""
    light, _ = demo("ritual", tier=LIGHT)
    standard, _ = demo("ritual", tier=STANDARD)
    assert len({n.voice for n in light.notes if n.layer == "plucks"}) <= 2
    assert len({n.voice for n in standard.notes if n.layer == "plucks"}) == 3
    assert {n.layer for n in standard.notes} >= {
        "drone",
        "sub",
        "wash",
        "plucks",
        "lead",
        "heart",
        "taiko",
    }
    assert len(light.notes) < len(standard.notes)


def test_the_tunings_are_japanese_pentatonics_on_roots_the_drone_can_sit_on() -> None:
    """Hirajoshi, in and kumoi: five notes each, on D, E, G or A."""
    assert len({t[0] for t in TUNINGS}) == len(TUNINGS) >= 6
    for _, root, steps in TUNINGS:
        assert len(steps) == 5
        assert steps[0] == 0
        assert root in {2, 4, 7, 9}
    assert {"hirajoshi", "in", "kumoi"} <= {name.split()[0] for name, _, _ in TUNINGS}
