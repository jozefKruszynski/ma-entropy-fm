"""Test of the balance of Ritual: the plucked line leads, the woods answer it and the drone sits under it."""

import numpy as np

from entropy_fm.quality import STANDARD
from entropy_fm.render import render_all

from .scape_tools import demo

WINDOW_SECONDS = 40
# the slit drum and the woodblock answer the pluck from under it, by at most this many dB; the woodblock is a few
# hits in the window and the master compressor holds them, so it cannot be raised much further
SLIT_MAX_UNDER_DB = 8.0
BLOCK_MAX_UNDER_DB = 14.0
# the pluck is at least level with the drone, give or take this much
DRONE_MAX_OVER_DB = 1.0
SILENCE_DB = -90.0


def _solo_db(layer: str) -> float:
    from soundscapes import pick_windows  # noqa: PLC0415

    score, plan = demo("ritual")
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
    assert pluck - slit <= SLIT_MAX_UNDER_DB, (pluck, slit)
    assert pluck - block <= BLOCK_MAX_UNDER_DB, (pluck, block)
    assert drone - pluck <= DRONE_MAX_OVER_DB, (pluck, drone)
