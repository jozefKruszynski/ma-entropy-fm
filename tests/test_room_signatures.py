"""Tests that every room has a lasting signature sound per style, and that vetoed voices are never played."""

from __future__ import annotations

from functools import lru_cache
from itertools import pairwise

import pytest

from entropy_fm.day_analysis import analyze_day
from entropy_fm.plan import PieceOptions, Plan
from entropy_fm.quality import STANDARD
from entropy_fm.scape import scape_score
from entropy_fm.score import Score
from entropy_fm.signature import POOLS, SigVoice
from entropy_fm.styles import STYLES
from entropy_fm.synth import Scale

from .scape_tools import the_demo

ROOMS = ("kitchen", "bedroom", "office", "bathroom", "living_room", "_house", "hall", "garage")
DAYS = (7, 9, 11, 13)
HOME = Scale(2, "dorian")
# the name under which each signature layer notes its voice in `plan.choices`, and the layers its notes play on
CHOICES = {
    ("ambient", "piano"): ("piano", ("piano",)),
    ("ambient", "drone"): ("drone voice", ("drone",)),
    ("berlin_school", "lead"): ("lead voice", ("lead",)),
    ("berlin_school", "seqA"): ("sequence voice", ("seqA",)),
    ("lofi", "keys"): ("keys", ("keys", "chop")),
    ("lofi", "pluck"): ("pluck", ("pluck",)),
    ("lofi", "bass"): ("bass voice", ("bass",)),
    ("ritual", "plucks"): ("pluck voice", ("plucks",)),
    ("ritual", "lead"): ("leader voice", ("lead",)),
    ("ritual", "drone"): ("drone voice", ("drone",)),
    ("glitch", "str"): ("string voice", ("str",)),
    ("glitch", "bass"): ("bass voice", ("bass",)),
}
# the notes of a drone that the signature plays: the cello section of the ritual drone stays
SIGNATURE_NOTES_ONLY = {("ritual", "drone")}

# voices the owner vetoed: no composer may choose them, in any layer of any style
VETOED = frozenset({"flute_lead", "fm_ep", "glass_pad", "nylon_ks"})
# the owner kept the glass pad as the main pad of Ambient
VETO_EXCEPTIONS = {"ambient": frozenset({"glass_pad"})}


def vetoed(style: str) -> frozenset[str]:
    """Return the voices a style may not play."""
    return VETOED - VETO_EXCEPTIONS.get(style, frozenset())


@lru_cache(maxsize=256)
def _made(style: str, room: str, index: int) -> tuple[Score, Plan]:
    """Return the score and the plan of a day of the demo in a style, for a room key."""
    house = the_demo()
    series = house.days[index]["kitchen"] + house.days[index]["sun"]
    return scape_score(
        series,
        analyze_day(series),
        style,
        f"{room}|{house.day_of(index).isoformat()}|_scape|{style}",
        HOME,
        quality=STANDARD,
        options=PieceOptions(day=house.day_of(index), area_key=room),
    )


def _signature_notes(score: Score, style: str, layer: str) -> set[tuple[str, int]]:
    """Return the (voice, patch) of the notes of a signature layer."""
    keep = {sig.voice for sig in POOLS[style][layer]}
    return {
        (n.voice, n.patch)
        for n in score.notes
        if n.layer in CHOICES[style, layer][1]
        and (n.voice in keep or (style, layer) not in SIGNATURE_NOTES_ONLY)
    }


def test_every_signature_layer_has_a_pool_and_a_choice() -> None:
    """The table of this test covers every pool of the module, and no pool offers a vetoed voice."""
    assert {(style, layer) for style, layers in POOLS.items() for layer in layers} == set(CHOICES)
    assert set(POOLS) == set(STYLES)
    for layers in POOLS.values():
        for pool in layers.values():
            assert len(pool) >= 2
            assert not {sig.voice for sig in pool} & VETOED


@pytest.mark.parametrize(("style", "layer"), sorted(CHOICES))
def test_a_room_keeps_its_signature_on_every_day(style: str, layer: str) -> None:
    """The same room plays the same lead and bass or drone on different days, and only voices of its pool."""
    name, _ = CHOICES[style, layer]
    pool = {sig.label: sig for sig in POOLS[style][layer]}
    for room in ROOMS[:3]:
        chosen = {_made(style, room, index)[1].choices[name] for index in DAYS}
        assert len(chosen) == 1, (style, layer, room)
        assert chosen <= set(pool)
        sig: SigVoice = pool[next(iter(chosen))]
        for index in DAYS:
            played = _signature_notes(_made(style, room, index)[0], style, layer)
            assert played
            assert {voice for voice, _ in played} == {sig.voice}, (style, layer, room, index)
            if sig.patch is not None:
                assert {patch for _, patch in played} == {sig.patch}


@pytest.mark.parametrize(("style", "layer"), sorted(CHOICES))
def test_rooms_spread_over_the_pool(style: str, layer: str) -> None:
    """Across a handful of rooms, every signature layer uses more than one member of its pool."""
    name, _ = CHOICES[style, layer]
    chosen = {_made(style, room, DAYS[0])[1].choices[name] for room in ROOMS}
    assert len(chosen) > 1, (style, layer)


@pytest.mark.parametrize("style", sorted(POOLS))
def test_vetoed_voices_are_never_played(style: str) -> None:
    """No note of any style plays a vetoed voice, for several rooms and days."""
    for room in ROOMS:
        for index in DAYS[:2]:
            assert not set(_made(style, room, index)[0].voices_used()) & vetoed(style), (
                style,
                room,
                index,
            )


@pytest.mark.parametrize("style", sorted(POOLS))
def test_the_other_choices_still_vary_from_day_to_day(style: str) -> None:
    """Of one room, something that is not a signature changes from a day to the next."""
    signature_names = {name for (s, _), (name, _) in CHOICES.items() if s == style}
    plans = [_made(style, "kitchen", index)[1] for index in range(7, 14)]
    others = [{k: v for k, v in p.choices.items() if k not in signature_names} for p in plans]
    assert any(a != b for a, b in pairwise(others)), style
