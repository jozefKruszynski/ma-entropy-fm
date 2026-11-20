"""Tests of Ritual: the pentatonic loops, the vibrato of the drone, the chant leader, the heartbeat, the taiko, the washes."""

import itertools
import time

import numpy as np
import pytest

from entropy_fm import dsp, stretch
from entropy_fm.drums import render_hit
from entropy_fm.plan import Plan
from entropy_fm.quality import LIGHT, STANDARD
from entropy_fm.scape_ritual import (
    BLOCK_PATTERNS,
    CELLS,
    CLACK_DELAY,
    LENGTHS,
    MIN_GAP,
    PLUCKS,
    SLIT_PATTERNS,
    TUNINGS,
)
from entropy_fm.score import Score
from entropy_fm.voices import NOTES, render_note

from .scape_tools import demo, window

RATE = 44100
# no bell, chime, gong or glassy voice is allowed anywhere in Ritual
FORBIDDEN = {"glass_pad", "fm_ep", "tx_ep", "fm_growl", "reese", "juno_pluck", "blip_synth"}
# every voice that is plucked or struck on a string or a key: there is one of them at a time
PLUCKED = {"strum_pluck", "nylon_ks", "felt_piano", "grand_piano", "juno_pluck", "moog_seq"}
WOOD = {"slit", "block", "clack"}


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


def test_the_wood_loops_are_of_other_lengths_than_the_pluck_and_phase_against_it() -> None:
    """The cell is eight pulses; the slit drum and the woodblock loop in 9 to 19 pulses, none a multiple or a simple fraction of it."""
    assert len(LENGTHS) >= 4
    assert all(6.0 <= seconds <= 7.2 for _, seconds in LENGTHS)
    lengths = [pulses for _, pulses, _ in (*SLIT_PATTERNS, *BLOCK_PATTERNS)]
    assert len(set(lengths)) >= 5
    for pulses in lengths:
        assert pulses % 2 == 1, pulses
    for _, pulses, hits in SLIT_PATTERNS:
        assert 9 <= pulses <= 13
        assert 3 <= len(hits) <= 6
    for _, pulses, hits in BLOCK_PATTERNS:
        assert 15 <= pulses <= 19
        assert len(hits) <= 3
    # the slit drum and the woodblock do not share a length
    assert not {p for _, p, _ in SLIT_PATTERNS} & {p for _, p, _ in BLOCK_PATTERNS}


def test_every_hit_of_the_wood_is_on_an_off_beat_of_the_pluck_pulse() -> None:
    """A pattern has a core that always plays and extras that activity calls in, and every hit is off the beat (a half or three quarters)."""
    for _, pulses, hits in (*SLIT_PATTERNS, *BLOCK_PATTERNS):
        assert any(need == 0.0 for *_, need in hits), "a core"
        assert len({pulse for pulse, *_ in hits}) == len(hits)
        for pulse, frac, _tone, need in hits:
            assert 0 <= pulse < pulses
            assert frac in {0.5, 0.75}
            assert 0.0 <= need < 1.0
    # a pulse is at least 0.75 s, so even a late off-beat is 0.19 s from the next plucked note
    assert min(seconds for _, seconds in LENGTHS) / 8.0 * 0.25 > 1.5 * MIN_GAP


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
    """The cello section and the pad of the drone and the chant leader sing a vibrato of 4 to 8 cents; the sub, the pluck and the wood do not."""
    score, _ = demo("ritual")
    sung = {n.voice for n in score.notes if n.vib > 0.0}
    assert sung <= {"tape_strings", "juno_pad", "chant_cello"}
    assert {"tape_strings", "juno_pad", "chant_cello"} <= sung
    assert all(4.0 <= n.vib <= 8.0 for n in score.notes if n.vib > 0.0)
    assert all(
        n.vib == 0.0
        for n in score.notes
        if n.layer in {"sub", "plucks", "heart", "taiko", "wash", *WOOD}
    )


def test_the_night_is_drone_and_washes_alone_and_the_pluck_comes_before_the_wood() -> None:
    """No pluck, wood, lead or drum in the night; the pluck wakes first, then the slit drum, then the woodblock."""
    score, plan = demo("ritual")
    night = _arc(plan, "night")
    in_night = {n.layer for n in score.notes if n.t < night[1]}
    assert in_night <= {"drone", "sub", "wash"}
    assert {"drone", "wash"} <= in_night
    assert not [n for n in score.notes if n.voice.startswith("drum:") and n.t < night[1]]
    assert not [
        n for n in score.notes if n.voice.startswith("drum:") and n.t >= _arc(plan, "late night")[0]
    ]
    first = {
        layer: min(n.t for n in score.notes if n.layer == layer)
        for layer in ("plucks", "slit", "block")
    }
    assert first["plucks"] < first["slit"] < first["block"], first
    assert first["slit"] - first["plucks"] > 3.0, first
    last_plucks = max(n.t for n in score.notes if n.layer == "plucks")
    assert last_plucks < plan.length - 20.0, "the evening settles"
    # the wood leaves with the evening too
    assert max(n.t for n in score.notes if n.layer in {"slit", "block"}) < plan.length - 10.0


def test_there_is_one_plucked_voice_at_a_time_and_never_two_plucks_together() -> None:
    """The day deals one of strumstick, nylon and felt piano; no other voice is plucked and the notes of the line never overlap."""
    seen = set()
    for index in (13, 10, 7, 4, 1):
        score, _ = demo("ritual", index)
        voices = {n.voice for n in score.notes if n.voice in PLUCKED}
        assert len(voices) == 1, (index, voices)
        assert voices <= {key for _, key, _ in PLUCKS}
        assert {n.voice for n in score.notes if n.layer == "plucks"} == voices
        seen |= voices
        line = sorted(n.t for n in score.notes if n.layer == "plucks")
        # a cell is one line: its onsets sit on the grid of eight and are at least a fraction of a slot apart
        assert min(b - a for a, b in itertools.pairwise(line)) > 0.2, index
        # nothing plucked sounds on any other layer
        assert not [n for n in score.notes if n.voice in PLUCKED and n.layer != "plucks"]
    assert len(seen) >= 2, "the day changes the voice"
    assert {key for _, key, _ in PLUCKS} == {"strum_pluck", "nylon_ks", "felt_piano"}
    assert len(CELLS) >= 6


def _onsets(score: Score, layers: set[str]) -> list[float]:
    return sorted(n.t for n in score.notes if n.layer in layers)


def test_the_wood_keeps_a_gap_to_the_pluck_and_the_hits_are_syncopated() -> None:
    """No hit of the slit drum or the woodblock is within 120 ms of a plucked note; they fall between the plucks, on off-beats."""
    for index in (13, 10, 7):
        score, _ = demo("ritual", index)
        plucks = np.array(_onsets(score, {"plucks"}))
        assert len(plucks) > 20
        for layer in ("slit", "block"):
            hits = _onsets(score, {layer})
            assert len(hits) > 5, (index, layer)
            nearest = [float(np.abs(plucks - h).min()) for h in hits]
            assert min(nearest) >= MIN_GAP, (index, layer, min(nearest))
            # off the beat, not just clear of it: the typical hit is far from the nearest pluck
            assert float(np.median(nearest)) > 0.25, (index, layer, np.median(nearest))
        both = np.array(_onsets(score, {"slit", "block"}))
        assert min(np.diff(both)) > 0.2, "the slit drum and the woodblock never sound together"


def test_the_slit_drum_is_sparse_in_the_scale_and_the_busy_part_calls_in_more_of_it() -> None:
    """Slit drum hits are low notes of the day's pentatonic, a few seconds apart; an active stretch plays more of the pattern than a calm one."""
    score, plan = demo("ritual")
    _, allowed = _scale_of(score)
    slit = [n for n in score.notes if n.layer == "slit"]
    assert slit
    assert all(round(n.midi) % 12 in allowed for n in slit)
    assert all(42.0 <= n.midi <= 55.0 and n.voice == "slit_drum" for n in slit)
    assert len(slit) < 0.4 * plan.length, "sparse: well under one hit in two seconds on average"
    block = [n for n in score.notes if n.layer == "block"]
    assert block
    assert all(round(n.midi) % 12 in allowed and n.voice == "woodblock" for n in block)
    assert len(block) < 0.35 * len(slit), "very few accents"
    assert np.mean([n.vel for n in block]) < np.mean([n.vel for n in slit]), "lower in the mix"
    # the activity steers the density: in the busiest quarter of the awake day more of the pattern plays than in the calmest
    awake = np.where(plan.daylight[: plan.count] > 0.5)[0] / 10.0
    assert len(awake) > 100
    per_second = np.array([plan.activity[int(t * 10)] for t in awake])
    busy_t = set(np.round(awake[per_second >= np.quantile(per_second, 0.75)]).astype(int))
    calm_t = set(np.round(awake[per_second <= np.quantile(per_second, 0.25)]).astype(int))
    busy_hits = sum(round(n.t) in busy_t for n in slit) / max(len(busy_t), 1)
    calm_hits = sum(round(n.t) in calm_t for n in slit) / max(len(calm_t), 1)
    assert busy_hits >= calm_hits, (busy_hits, calm_hits)


def test_a_few_shishi_odoshi_clacks_fall_in_a_piece_a_tick_then_a_deep_knock() -> None:
    """A soft woodblock tick, then a lowered slit drum knock 0.14 s later; a few in a piece, apart, on a scale tone, none in the night."""
    for index in (13, 10, 7):
        score, plan = demo("ritual", index)
        clack = sorted((n for n in score.notes if n.layer == "clack"), key=lambda n: n.t)
        ticks = [n for n in clack if n.voice == "woodblock"]
        knocks = [n for n in clack if n.voice == "slit_drum"]
        assert 1 <= len(knocks) <= 5, (index, len(knocks))
        assert len(ticks) == len(knocks)
        _, allowed = _scale_of(score)
        for tick, knock in zip(ticks, knocks, strict=True):
            assert knock.t - tick.t == pytest.approx(CLACK_DELAY)
            assert tick.vel < 0.5 < knock.vel, "a soft tick, then the clack"
            assert knock.midi <= 50.0
            assert round(knock.midi) % 12 in allowed
            assert 80.0 <= tick.midi < 92.0
            assert round(tick.midi) % 12 in allowed
        if len(knocks) > 1:
            assert min(b.t - a.t for a, b in itertools.pairwise(knocks)) >= 30.0
        assert min(n.t for n in clack) > _arc(plan, "night")[1]
        # a clack is clear of the plucked notes and of the other wood
        plucks = np.array(_onsets(score, {"plucks"}))
        assert min(float(np.abs(plucks - n.t).min()) for n in clack) >= 0.3
        other = np.array(_onsets(score, {"slit", "block"}))
        assert min(float(np.abs(other - n.t).min()) for n in clack) >= 0.3
    light, _ = demo("ritual", tier=LIGHT)
    assert not [n for n in light.notes if n.layer in {"clack", "block"}]
    assert [n for n in light.notes if n.layer == "slit"]


def test_the_wood_voices_are_the_vcsl_samples_and_make_no_noise_or_ring() -> None:
    """The slit drum and the woodblock are rendered from the bundled VCSL samples; neither rings for long nor has a bright edge."""
    for key, midi, ring in (("slit_drum", 47.0, 1.0), ("woodblock", 85.0, 0.55)):
        wave = render_note(key, midi, 0.5, 0.7, STANDARD).mean(axis=1)
        assert np.isfinite(wave).all()
        assert np.abs(wave).max() > 0.01
        assert len(wave) / RATE < ring + 0.3
        spectrum = np.abs(np.fft.rfft(wave * np.hanning(len(wave)), 1 << 16)) ** 2
        freqs = np.fft.rfftfreq(1 << 16, 1.0 / RATE)
        assert spectrum[freqs > 6000.0].sum() / spectrum.sum() < 2e-3, key
    slit = render_note("slit_drum", 47.0, 0.5, 0.7, STANDARD).mean(axis=1)
    spectrum = np.abs(np.fft.rfft(slit[: RATE // 2] * np.hanning(RATE // 2), 1 << 16))
    freqs = np.fft.rfftfreq(1 << 16, 1.0 / RATE)
    band = (freqs > 60.0) & (freqs < 400.0)
    peak = float(freqs[band][np.argmax(spectrum[band])])
    assert abs(12.0 * np.log2(peak / 440.0) + 69.0 - 47.0) < 1.0, "the knock is read onto the note"


def test_the_plucks_are_in_the_scale_and_the_cells_drift_by_one_step() -> None:
    """The pluck, the wood, the leader and the drone are in the pentatonic of the day; a cell never drifts more than a step from its shape."""
    score, plan = demo("ritual")
    root, allowed = _scale_of(score)
    plucks = [n for n in score.notes if n.layer in {"plucks", "lead", *WOOD}]
    assert plucks
    assert all(round(n.midi) % 12 in allowed for n in plucks)
    drone = [n for n in score.notes if n.layer in {"drone", "sub"}]
    assert {(round(n.midi) - root) % 12 for n in drone} <= {0, 5, 7}
    # a cell of five to seven notes moves one note at most a step either way: the line never takes many pitches
    assert len({n.midi for n in score.notes if n.layer == "plucks"}) <= 14
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
    """Light has the pluck and the slit drum, no washes, leader, woodblock or clack; Standard has all of them."""
    light, _ = demo("ritual", tier=LIGHT)
    standard, _ = demo("ritual", tier=STANDARD)
    assert len({n.voice for n in light.notes if n.layer == "plucks"}) == 1
    assert len({n.voice for n in standard.notes if n.layer == "plucks"}) == 1
    assert {n.layer for n in standard.notes} >= {
        "drone",
        "sub",
        "wash",
        "plucks",
        "slit",
        "block",
        "clack",
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
