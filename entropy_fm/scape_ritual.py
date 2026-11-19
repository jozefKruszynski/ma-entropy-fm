"""
Ritual: three plucked loops of unequal length that phase over a deep choir drone, stretched string washes, a slow chant leader and taiko.

The generative system is a slow ceremony in a Japanese pentatonic (hirajoshi, in or kumoi, on a root the day deals).
Three short call-and-response cells of different lengths (4 to 8 seconds, no two in a simple ratio) are plucked on a
strumstick, a nylon string and a felt piano, round and low-passed, and drift against each other as they repeat; about
one repeat in three moves a single note by one step of the scale and never further from where the cell began, so it
stays a chant and never becomes noodling. Below them a drone of a cello section and a pad, and the pad an octave up (with a hint of
vibrato that begins after a second or two and is different for every voice) moves root, fifth, root, fourth over
minutes, its low-pass opening and closing like breath over a dozen seconds and its level swelling over eleven.
Stretched washes of string sections and piano (`stretch`) overlap under everything. Now and then a chant leader, a
cello section playing two or three long notes of the scale with rests, answers the end of a pluck cell. In the
busy part of the day a soft heartbeat drum (a low tom, no click) keeps a slow pulse and a few big taiko strikes
fall on that pulse, with a short accelerating roll into a strike at an unusual moment. There are no bells, no
chimes, no gongs and no noise: the only things that are not tones are the skins of the drums.

The day is spread over the piece as in the other styles. The night is drone and washes alone; the plucks come in one
loop after another as the day wakes and leave one after another as the evening settles; the lead and the drums
are for the awake and busy stretches.

How the house steers it: activity (with the daylight) brings the loops in and out and sets their loudness, the
heartbeat and how often a taiko is struck; the brightness opens the filters of the plucks and the drone; the power of
the appliances thickens the drone and hardens the strikes; humidity makes the room bigger; the lights and the evening
make the chant leader more likely; an anomaly is a roll into one big strike, a swell, a dip of the filters and a
move of the drone to a fourth or a fifth. The house never writes a note: the notes come from the cells, the path of
the drone, the pulse and the phrases of the leader, and the day deals which of each.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import taiko
from .gen import dips, swells, weighted
from .score import CR, Layer, Note, Score
from .synth import smoothstep

if TYPE_CHECKING:
    from .plan import Plan

HIRAJOSHI = (0, 2, 3, 7, 8)
IN_SEN = (0, 1, 5, 7, 8)
KUMOI = (0, 2, 3, 7, 9)
# the tunings: a name, the pitch class of the root and the five notes above it; each root is one that the drone can sit on
TUNINGS: tuple[tuple[str, int, tuple[int, ...]], ...] = (
    ("hirajoshi on D", 2, HIRAJOSHI),
    ("kumoi on A", 9, KUMOI),
    ("in on E", 4, IN_SEN),
    ("hirajoshi on G", 7, HIRAJOSHI),
    ("kumoi on D", 2, KUMOI),
    ("in on A", 9, IN_SEN),
    ("hirajoshi on E", 4, HIRAJOSHI),
    ("kumoi on G", 7, KUMOI),
)
# loop lengths in seconds of the mid, the high and the low pluck: none is in a simple ratio to another, so they keep phasing
LENGTHS: tuple[tuple[float, float, float], ...] = (
    (4.6, 6.5, 7.3),
    (4.3, 6.8, 7.7),
    (4.2, 6.6, 7.5),
    (4.8, 6.8, 7.6),
)
# which voice plays which pluck: the mid, the high and the low
VOICE_TRIOS: tuple[tuple[str, tuple[str, str, str]], ...] = (
    ("strumstick, nylon, felt piano", ("strum_pluck", "nylon_ks", "felt_piano")),
    ("nylon, strumstick, soft grand", ("nylon_ks", "strum_pluck", "grand_piano")),
    ("strumstick, felt piano, nylon", ("strum_pluck", "felt_piano", "nylon_ks")),
    ("felt piano, nylon, strumstick", ("felt_piano", "nylon_ks", "strum_pluck")),
)
# the cells: (a slot on the grid of eight of the loop, a step of the scale from the root; 0 is the root at about D4).
# Three notes, a rest, and the answer: every cell calls and responds.
Cell = tuple[tuple[int, int], ...]
MID_CELLS: tuple[tuple[str, Cell], ...] = (
    ("call and fall", ((0, 3), (1, 5), (2, 7), (4, 6), (5, 5), (6, 3))),
    ("upper turn", ((0, 5), (1, 7), (2, 5), (4, 3), (5, 5), (6, 3))),
    ("rising answer", ((0, 3), (2, 5), (3, 6), (5, 5), (6, 3), (7, 2))),
    ("descent", ((0, 7), (1, 6), (2, 5), (4, 3), (5, 5))),
)
HIGH_CELLS: tuple[tuple[str, Cell], ...] = (
    ("rocking", ((0, 5), (1, 6), (2, 5), (4, 4), (5, 3), (6, 4), (7, 3))),
    ("leap", ((0, 6), (1, 5), (3, 7), (4, 6), (6, 5))),
    ("sigh", ((0, 5), (2, 6), (3, 5), (5, 3), (6, 4))),
    ("bright call", ((0, 7), (1, 5), (2, 6), (4, 5), (5, 4), (6, 3))),
)
LOW_CELLS: tuple[tuple[str, Cell], ...] = (
    ("root and fifth", ((0, 0), (1, 3), (2, 0), (4, 2), (5, 1), (6, 0))),
    ("wide", ((0, 0), (2, 2), (3, 0), (4, 3), (6, 1))),
    ("falling", ((0, 3), (1, 2), (2, 0), (4, 1), (6, 0))),
    ("deep step", ((0, 0), (2, 3), (4, 2), (5, 0), (6, -2))),
)
# the path of the drone: the interval above the root of each stretch (seven is the fifth, five the fourth), repeated
DRONE_PATHS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("settled", (0, 7, 0, 5)),
    ("rising", (0, 5, 7, 0)),
    ("rocking", (0, 7, 5, 7)),
    ("deep", (0, 0, 7, 5)),
)
# the washes: an instrument and the step (in semitones) above the drone root one octave up; the last one pulls
WASH_RECIPES: tuple[tuple[str, int], ...] = (
    ("wash:cello", 0),
    ("wash:cello", 7),
    ("wash:felt", -4),
    ("wash:violin", 12),
    ("wash:cello", 6),
)
WASH_SETS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("low cellos", (0, 1, 4, 2)),
    ("violin veil", (0, 3, 1, 2)),
    ("pulled", (0, 4, 3, 1)),
)
# how often the chant leader answers: the range of seconds between two phrases
LEAD_PRESENCE: tuple[tuple[str, tuple[float, float]], ...] = (
    ("rare", (95.0, 150.0)),
    ("sparse", (55.0, 90.0)),
    ("answering", (32.0, 55.0)),
    ("attentive", (22.0, 38.0)),
)
HEARTBEATS: tuple[tuple[str, int], ...] = (("single beat", 0), ("lub-dub", 1))
WASH_SECONDS = 24.0
DRONE_STEP = 18.0
DRONE_OVERLAP = 4.5


def _lfo(count: int, period: float, phase: float) -> np.ndarray:
    """Return a slow 0 to 1 swing with `period` seconds, a value every tenth of a second."""
    t = (np.arange(count) + 0.5) / CR
    return np.asarray(0.5 - 0.5 * np.cos(2.0 * np.pi * (t / period + phase)))


def compose(plan: Plan) -> Score:  # noqa: C901, PLR0912, PLR0915
    """
    Make the score of a Ritual soundscape.

    :param plan: The curves and the deals of the day.
    """
    quality = plan.quality
    length = plan.length
    rng = plan.rng("ritual")
    light = quality.max_layers <= 4

    tuning_name, root_pc, steps = plan.deal("tuning", TUNINGS)
    lengths = plan.deal("loop lengths", LENGTHS)
    trio_name, trio = plan.deal("pluck voices", VOICE_TRIOS)
    mid_name, mid_cell = plan.deal("mid cell", MID_CELLS)
    high_name, high_cell = plan.deal("high cell", HIGH_CELLS)
    low_name, low_cell = plan.deal("low cell", LOW_CELLS)
    path_name, path = plan.deal("drone path", DRONE_PATHS)
    wash_name, wash_order = plan.deal("wash set", WASH_SETS)
    lead_name, lead_gap = plan.deal("chant leader", LEAD_PRESENCE)
    heart_name, heart_style = plan.deal("heartbeat", HEARTBEATS)
    patch = int(plan.rng("patch").integers(0, 2))

    def ladder(index: int, base: int) -> float:
        """Return the pitch of a step of the scale: five notes an octave, `base` is the root."""
        octave, degree = divmod(index, 5)
        return float(base + 12 * octave + steps[degree])

    pluck_base = 55 + (root_pc - 55) % 12
    lead_base = 53 + (root_pc - 53) % 12
    drone_root = 38 + (root_pc - 38) % 12

    # -- the curves of the day
    count = plan.count
    activity = plan.activity
    # how awake the room is: the daylight, lifted by activity; a little of the evening stays while the house is lively
    awake = np.clip(
        plan.daylight * (0.4 + 0.6 * activity) + 0.3 * plan.evening * activity, 0.0, 1.0
    )
    busy = plan.gate(awake, 0.34, 0.6)
    swell = swells(plan)
    breath = _lfo(count, float(rng.uniform(11.5, 13.5)), float(rng.uniform()))
    surge = 0.55 + 0.45 * _lfo(count, float(rng.uniform(10.0, 12.0)), float(rng.uniform()))

    notes: list[Note] = []

    # -- the drone: a cello section and a Juno pad with a vibrato of their own, a quiet pad an octave up, and a round sub; the root moves along the path, and at an anomaly it goes to the fifth or the fourth
    segments: list[tuple[float, float, int]] = []
    clock, place = 0.0, 0
    while clock < length:
        span = float(rng.uniform(55.0, 105.0))
        end = min(clock + span, float(length))
        if length - end < 35.0:
            end = float(length)
        segments.append((clock, end, path[place % len(path)] if clock > 0.0 else 0))
        clock, place = end, place + 1
    for second, _ in plan.events:
        for k, (start, end, _interval) in enumerate(segments):
            if start <= second < end and end - second > 12.0:
                far = int(weighted(rng, (7, 5), (0.6, 0.4)))
                segments[k] = (start, second, segments[k][2])
                segments.insert(k + 1, (second, min(second + 45.0, end), far))
                if second + 45.0 < end:
                    segments.insert(k + 2, (second + 45.0, end, segments[k][2]))
                break
    index = 0
    for start, end, interval in segments:
        second = start
        while second < end - 3.0:
            dur = float(min(DRONE_STEP + DRONE_OVERLAP, end + DRONE_OVERLAP - second))
            pitch = float(drone_root + interval)
            vel = 0.5 + 0.12 * plan.at(plan.power, second)
            notes.append(Note(second, dur, pitch, vel, "tape_strings", "drone", index % 2, 6.0))
            notes.append(
                Note(second, dur, pitch, vel - 0.05, "juno_pad", "drone", (index + 1) % 2, 5.0)
            )
            if not light:
                notes.append(Note(second, dur, pitch + 12.0, 0.3, "juno_pad", "drone", index % 2))
            sub = pitch - 12.0 if pitch - 12.0 >= 26.0 else pitch
            notes.append(Note(second, dur, sub, 0.6, "sub_round", "sub", 1, 0.0))
            second += DRONE_STEP
            index += 1

    # -- the washes: stretched sections that overlap and replace each other, drawn from a few recipes so that each is made once
    if not light:
        recipes = [WASH_RECIPES[i] for i in wash_order]
        second, turn = 0.0, int(rng.integers(0, len(recipes)))
        while second < length:
            voice, step = recipes[turn % len(recipes)]
            pitch = float(drone_root + 12 + step)
            notes.append(Note(second, WASH_SECONDS, pitch, 0.6, voice, "wash", 0, 0.0))
            second += float(rng.uniform(15.0, 24.0))
            turn += 1

    # -- the plucks: three loops that enter one after another as the day wakes and leave as the evening settles
    loops = 2 if light else 3
    if plan.analysis.busyness < 0.25 and not light:
        loops = 3 if plan.rng("loops").random() < 0.6 else 2
    slots = (
        ("mid", mid_cell, lengths[0], trio[0], 0.62, 0.26),
        ("high", high_cell, lengths[1], trio[1], 0.55, 0.42),
        ("low", low_cell, lengths[2], trio[2], 0.5, 0.10),
    )
    entering = sorted(slots, key=lambda s: s[5])[:loops]
    last_cells: list[
        tuple[float, float, float]
    ] = []  # (end of the response, the pitch it ended on, length of the loop)
    for name, cell, loop_len, voice, loud, threshold in entering:
        loop_rng = plan.rng("loop", name)
        base_index = [step for _, step in cell]
        current = list(base_index)
        second = float(loop_rng.uniform(0.0, loop_len))
        slot = loop_len / 8.0
        while second < length - 2.0:
            presence = float(np.clip((plan.at(awake, second) - threshold) / 0.26, 0.0, 1.0))
            presence = float(smoothstep(np.array(presence)))
            if presence > 0.04 and loop_rng.random() < presence**0.5:
                if loop_rng.random() < 0.33:
                    pick = int(loop_rng.integers(0, len(current)))
                    current[pick] = int(
                        np.clip(
                            current[pick] + int(loop_rng.choice((-1, 1))),
                            base_index[pick] - 1,
                            base_index[pick] + 1,
                        )
                    )
                pitch = pluck_base
                for number, ((where, _), step) in enumerate(zip(cell, current, strict=True)):
                    when = second + where * slot + float(loop_rng.uniform(-0.008, 0.008))
                    accent = 1.12 if number == 0 else 1.0
                    vel = (
                        loud
                        * accent
                        * (0.78 + 0.3 * presence)
                        * (0.92 + 0.12 * plan.at(plan.bright, when))
                        * float(loop_rng.uniform(0.92, 1.06))
                    )
                    pitch = ladder(step, pluck_base)
                    notes.append(
                        Note(
                            when, 0.7, pitch, float(np.clip(vel, 0.1, 1.0)), voice, "plucks", patch
                        )
                    )
                last_cells.append((second + (cell[-1][0] + 1.0) * slot, pitch, loop_len))
            second += loop_len

    # -- the chant leader: a cello section answers the end of a pluck cell with two or three long notes and rests
    lead_notes = 0
    if not light and last_cells:
        lead_rng = plan.rng("lead")
        last_phrase = -1000.0
        previous = 1
        for end, pitch, _ in sorted(last_cells):
            if end - last_phrase < lead_gap[0] or end > length - 14.0:
                continue
            here = plan.at(awake, end)
            wanted = 0.45 + 0.55 * min(
                1.0, here + 0.5 * plan.at(plan.lights, end) + 0.5 * plan.at(plan.evening, end)
            )
            wait = lead_gap[0] + (lead_gap[1] - lead_gap[0]) * float(lead_rng.random())
            if here < 0.25 or end - last_phrase < wait or lead_rng.random() > wanted:
                continue
            # the pitch of the answer: the step of the lead's scale nearest to where the pluck ended, then a stepwise walk home
            reach = [
                (abs(ladder(i, lead_base) - (pitch - 12.0 * round((pitch - lead_base) / 12.0))), i)
                for i in range(-2, 7)
            ]
            current_step = min(reach)[1] if lead_rng.random() < 0.7 else previous
            count_in_phrase = int(weighted(lead_rng, (2, 3), (0.45, 0.55)))
            clock = end + 0.5 + float(lead_rng.uniform(0.0, 0.9))
            for number in range(count_in_phrase):
                final = number == count_in_phrase - 1
                if final:
                    current_step = int(weighted(lead_rng, (0, 3, 5), (0.5, 0.25, 0.25)))
                elif number > 0:
                    current_step = int(
                        np.clip(current_step + int(lead_rng.choice((-2, -1, -1, 1))), -2, 6)
                    )
                dur = float(lead_rng.uniform(5.0, 8.0) if final else lead_rng.uniform(3.2, 5.0))
                vel = float(lead_rng.uniform(0.46, 0.58)) * (
                    0.85 + 0.15 * plan.at(plan.bright, clock)
                )
                notes.append(
                    Note(
                        clock,
                        dur,
                        ladder(current_step, lead_base),
                        vel,
                        "chant_cello",
                        "lead",
                        patch,
                        6.0,
                    )
                )
                lead_notes += 1
                previous = current_step
                clock += dur + float(lead_rng.uniform(0.7, 1.9))
            last_phrase = end

    # -- the pulse and the taiko, in the busy stretches only
    bpm = float(rng.uniform(44.0, 54.0))
    beat = 60.0 / bpm
    taiko_hits: list[tuple[float, str, float]] = []
    big: list[float] = []
    quiet = 0.0
    drum_rng = plan.rng("taiko")
    strikes = 0
    rolls = 0
    grid = float(rng.uniform(0.0, beat))

    def on_grid(second: float) -> float:
        return grid + round((second - grid) / beat) * beat

    # an anomaly that is not in the dead of the night is a roll into one big strike
    event_starts: list[float] = []
    for second, _ in plan.events:
        if plan.at(plan.night, second) < 0.6:
            end = on_grid(second)
            span = float(drum_rng.uniform(3.0, 4.2))
            taiko_hits.extend(
                (hit.time, "odaiko", hit.velocity)
                for hit in taiko.roll(end - span, end, "odaiko", 0.18, 0.92)
            )
            event_starts.append(end - span)
            big.append(end)
            rolls += 1
    # chance strikes on the beat, rarer when the house is calm, never close to another one
    others: list[float] = [*big]
    beat_time = grid
    while beat_time < length - 4.0:
        gate = plan.at(busy, beat_time)
        near = any(abs(beat_time - other) < 5.0 for other in others)
        inside_roll = any(
            start - 0.5 <= beat_time <= end + beat
            for start, end in zip(event_starts, big, strict=True)
        )
        if gate > 0.55 and not near and not inside_roll:
            chance = (0.05 + 0.1 * plan.at(activity, beat_time)) * gate
            if drum_rng.random() < chance:
                vel = (
                    0.52
                    + 0.26 * plan.at(activity, beat_time)
                    + 0.14 * plan.at(plan.power, beat_time)
                )
                taiko_hits.append((beat_time, "odaiko", float(vel)))
                others.append(beat_time)
                strikes += 1
                extra = drum_rng.random()
                if extra < 0.18:
                    taiko_hits.append((beat_time + beat * 1.5, "taiko_rim", 0.5))
                elif extra < 0.32:
                    for k, v in enumerate((0.45, 0.35, 0.55)):
                        taiko_hits.append((beat_time + beat * (1.0 + 0.5 * k), "shime", v))
        beat_time += beat
    # a rare unprompted roll at the height of a busy stretch, not more often than every minute and a half
    roll_clock = 0.0
    while roll_clock < length - 12.0:
        roll_clock += float(drum_rng.uniform(30.0, 60.0))
        if (
            roll_clock < length - 12.0
            and plan.at(busy, roll_clock) > 0.7
            and drum_rng.random() < 0.55
            and not any(abs(roll_clock - other) < 20.0 for other in others)
        ):
            end = on_grid(roll_clock)
            taiko_hits.extend(
                (hit.time, "odaiko", hit.velocity)
                for hit in taiko.roll(end - 3.4, end, "odaiko", 0.18, 0.85)
            )
            event_starts.append(end - 3.4)
            big.append(end)
            others.append(end)
            rolls += 1
            roll_clock += 90.0
    # one last soft strike as the busy stretch ends
    awake_beats = [t for t in np.arange(grid, length - 4.0, beat) if plan.at(busy, float(t)) > 0.55]
    if taiko_hits and awake_beats:
        closing = float(awake_beats[-1]) + 4.0 * beat
        if closing < length - 6.0 and not any(abs(closing - other) < 4.0 for other in others):
            taiko_hits.append((closing, "odaiko", 0.5))
            others.append(closing)
            quiet = closing
    for when, hit, vel in taiko_hits:
        notes.append(
            Note(
                when,
                0.2,
                36.0,
                float(np.clip(vel, 0.1, 1.0)),
                f"drum:{hit}",
                "taiko",
                int(drum_rng.integers(0, 3)),
            )
        )
    # the heartbeat fills the beats between the big strikes
    heart_rng = plan.rng("heart")
    beat_time = grid
    beats = 0
    while beat_time < length - 3.0:
        gate = plan.at(busy, beat_time)
        taken = any(
            abs(beat_time - hit_time) < 0.9 * beat
            for hit_time, kind, _ in taiko_hits
            if kind == "odaiko"
        )
        rolling = any(
            start - 0.2 <= beat_time <= end + 0.6 * beat
            for start, end in zip(event_starts, big, strict=True)
        )
        if gate > 0.3 and not taken and not rolling:
            vel = (0.22 + 0.12 * gate + 0.04 * plan.at(plan.power, beat_time)) * float(
                heart_rng.uniform(0.92, 1.06)
            )
            variant = int(heart_rng.integers(0, 3))
            notes.append(Note(beat_time, 0.2, 36.0, vel, "drum:heart", "heart", variant))
            if heart_style == 1:
                notes.append(
                    Note(
                        beat_time + 0.27 * beat,
                        0.2,
                        36.0,
                        vel * 0.6,
                        "drum:heart",
                        "heart",
                        (variant + 1) % 3,
                    )
                )
            beats += 1
        beat_time += beat

    notes = [n for n in notes if n.t < length]

    # -- the layers
    bright = plan.bright
    low_cut, high_cut = 520.0, 2100.0
    drone_cut = low_cut * (high_cut / low_cut) ** breath * (0.85 + 0.3 * bright) * dips(plan, 0.85)
    pluck_cut = 2500.0 * (0.75 + 0.45 * bright) * dips(plan, 0.6)
    drone_presence = np.clip(surge * (0.9 + 0.12 * plan.power) * (1.0 + 0.25 * swell), 0.0, 1.25)
    wash_presence = np.clip(
        (0.45 + 0.45 * plan.night + 0.2 * swell) * (1.0 - 0.25 * busy), 0.0, 1.0
    )
    layers = {
        "drone": Layer(
            "drone", 1.0, drone_presence, drone_cut, 0.0, reverb=0.4, title="choir drone, breathing"
        ),
        "sub": Layer("sub", 0.9, drone_presence, None, 0.0, reverb=0.1, title="round sub"),
        "wash": Layer("wash", 1.0, wash_presence, None, 0.0, reverb=0.8, title="stretched washes"),
        "plucks": Layer("plucks", 1.0, None, pluck_cut, 0.0, reverb=0.62, title="plucked loops"),
        "lead": Layer("lead", 1.0, None, None, 0.0, reverb=0.6, title="chant leader"),
        "taiko": Layer("taiko", 1.0, None, None, 0.0, reverb=0.5, title="taiko"),
        "heart": Layer("heart", 1.0, None, None, 0.0, reverb=0.25, title="heartbeat"),
    }
    space = np.clip(0.5 + 0.25 * plan.humid + 0.2 * swell + 0.1 * plan.night, 0.0, 0.95)
    score = Score(
        length,
        notes,
        layers,
        [],
        space,
        events=list(plan.events),
        compress_db=-15.0,
        level_db=-18.8,
    )
    plan.choices["patch"] = str(patch)
    score.choices = dict(plan.choices)
    used = score.voices_used()
    score.notes_about = [
        (
            f"{tuning_name}; plucks of {lengths[0]:.1f}, {lengths[1]:.1f} and {lengths[2]:.1f} seconds ({loops} entering by how awake the room is) on {trio_name}, "
            f"cells '{mid_name}', '{high_name}' and '{low_name}', about one repeat in three moving one note by a step."
        ),
        (
            f"Drone on a cello section and a Juno pad (and a quiet one an octave up) with a vibrato of 4 to 6 cents that begins after a second or two (rate and phase of each note its own), path '{path_name}' "
            f"({len(segments)} stretches), washes '{wash_name}' ({used.get('wash:cello', 0) + used.get('wash:violin', 0) + used.get('wash:felt', 0)} placed, {WASH_SECONDS:.0f} seconds each)."
        ),
        (
            f"Chant leader '{lead_name}' ({lead_notes} long notes of a cello section in the scale, with rests); heartbeat '{heart_name}' at {bpm:.0f} per minute "
            f"({beats} beats in the busy stretches), {strikes} taiko strikes on the pulse and {rolls} accelerating rolls."
        ),
    ]
    if quiet:
        score.notes_about.append(
            f"A last soft strike at {int(quiet) // 60}:{int(quiet) % 60:02d} as the busy stretch ends."
        )
    return score
