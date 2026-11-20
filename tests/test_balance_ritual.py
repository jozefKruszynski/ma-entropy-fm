"""Test of the balance of Ritual: the plucked line leads, the woods answer it and the drone sits under it."""

from dataclasses import replace

import numpy as np

from entropy_fm.quality import STANDARD
from entropy_fm.render import render_all

from .scape_tools import demo

WINDOW_SECONDS = 40
# the slit drum and the woodblock answer the pluck from under it, by at least and at most this many dB
WOODS_MIN_UNDER_DB = 3.0
SLIT_MAX_UNDER_DB = 8.0
BLOCK_MAX_UNDER_DB = 12.0
# the pluck is at least level with the drone, give or take this much
DRONE_MAX_OVER_DB = 1.0
SILENCE_DB = -90.0


def _solo_db(layer: str) -> float:
    from soundscapes import pick_windows  # noqa: PLC0415

    shared, plan = demo("ritual")
    # the master dynamics off, on a copy of the shared score: a layer alone through the compressor reads louder than it is
    score = replace(shared, compress_db=None, gain_db=None)
    start = pick_windows(plan, WINDOW_SECONDS)[1][1]
    audio = render_all(
        score,
        STANDARD,
        "test",
        None,
        start_s=start,
        end_s=start + WINDOW_SECONDS,
        solo=frozenset({layer}),
    )
    return float(20.0 * np.log10(np.sqrt(np.mean(audio**2)) + 1e-9))


def test_the_pluck_leads_and_the_woods_answer_it_from_under() -> None:
    """In the active daytime the slit drum and the woodblock are heard against the pluck, and the drone does not bury it."""
    pluck, slit, block, drone = (_solo_db(layer) for layer in ("plucks", "slit", "block", "drone"))
    assert min(pluck, slit, block) > SILENCE_DB, (pluck, slit, block)
    assert WOODS_MIN_UNDER_DB <= pluck - slit <= SLIT_MAX_UNDER_DB, (pluck, slit)
    assert WOODS_MIN_UNDER_DB <= pluck - block <= BLOCK_MAX_UNDER_DB, (pluck, block)
    assert drone - pluck <= DRONE_MAX_OVER_DB, (pluck, drone)
