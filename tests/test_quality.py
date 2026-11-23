"""Tests of the quality tiers."""

from entropy_fm.quality import (
    LIGHT,
    ORDER,
    REVERB_FREEVERB,
    REVERB_HALL,
    REVERB_ROOM,
    RICH,
    STANDARD,
    choose_tier,
    get_quality,
)

HEAVY_COST_MS = 900.0
LIGHT_COST_MS = 100.0


def test_tiers_run_from_the_cheapest_reverb_to_the_richest() -> None:
    """Each tier has a reverb of its own, from the Freeverb to the hall."""
    assert [tier.reverb for tier in ORDER] == [REVERB_FREEVERB, REVERB_ROOM, REVERB_HALL]


def test_choose_tier_takes_the_richest_that_fits_the_budget() -> None:
    """Auto picks the richest tier under the budget, and Light when none fits."""
    costs = {LIGHT.key: LIGHT_COST_MS, STANDARD.key: LIGHT_COST_MS, RICH.key: HEAVY_COST_MS}
    assert choose_tier(costs) is STANDARD
    assert choose_tier({LIGHT.key: HEAVY_COST_MS}) is LIGHT
    assert get_quality("nonsense") is STANDARD
