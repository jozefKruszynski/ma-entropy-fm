"""
The render quality tiers of Entropy FM: what the engine may spend on a second of audio.

Many people run Music Assistant on very small machines, so the engine has three tiers with hard
caps on notes, voices and effects. The setting is Light, Standard, Rich or Auto, and Auto picks the
richest tier that a short benchmark at start-up says would use less than a share of one core.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

from dataclasses import dataclass

# Auto picks the richest tier that would use under this share of one core in real time
AUTO_BUDGET = 0.30


@dataclass(frozen=True)
class Quality:
    """What one tier of the engine spends."""

    key: str
    title: str
    # the most layers (loops, sequences, pads) that play at the same time
    max_layers: int
    # the saws of an expensive voice (the supersaw has 7 at most, the reese 3), as a share of full
    unison: float
    # whether the growling bass plays all of its parts (the FM and the second saw)
    rich_bass: bool
    # reverb: freeverb (one cheap Freeverb), room (a convolution with a 2.5 second room) or hall (5 seconds)
    reverb: str
    # whether the pads are chorused, whether the granular shimmer plays, whether tape wobble plays
    chorus: bool
    granular: bool
    wobble: bool
    # how many samples pass between two updates of a filter that the house moves
    control_chunk: int
    # the memory the notes that were made may use, in megabytes
    cache_mb: int
    # the look-ahead limiter works out its gain every this many samples (a divisor of 44100)
    limiter_stride: int = 20


LIGHT = Quality(
    key="light",
    title="Light",
    max_layers=4,
    unison=0.3,
    rich_bass=False,
    reverb="freeverb",
    chorus=False,
    granular=False,
    wobble=False,
    control_chunk=2048,
    limiter_stride=63,
    cache_mb=64,
)
STANDARD = Quality(
    key="standard",
    title="Standard",
    max_layers=6,
    unison=0.7,
    rich_bass=True,
    reverb="room",
    chorus=True,
    granular=False,
    wobble=True,
    control_chunk=1024,
    cache_mb=128,
)
RICH = Quality(
    key="rich",
    title="Rich",
    max_layers=9,
    unison=1.0,
    rich_bass=True,
    reverb="hall",
    chorus=True,
    granular=True,
    wobble=True,
    control_chunk=512,
    cache_mb=256,
)
QUALITIES: dict[str, Quality] = {tier.key: tier for tier in (LIGHT, STANDARD, RICH)}
# the order from the cheapest to the richest, which Auto walks
ORDER = (LIGHT, STANDARD, RICH)
AUTO = "auto"
DEFAULT_QUALITY = AUTO


def get_quality(key: str | None) -> Quality:
    """
    Return a tier by key; anything else (including Auto, which needs a benchmark) is Standard.

    :param key: light, standard or rich.
    """
    return QUALITIES.get(key or "", STANDARD)


def choose_tier(cost_ms: dict[str, float], budget: float = AUTO_BUDGET) -> Quality:
    """
    Pick the richest tier whose measured cost fits the budget; Light when none does.

    :param cost_ms: The CPU time in milliseconds that one second of audio took, per tier key.
    :param budget: The share of one core that may be used in real time.
    """
    limit = budget * 1000.0
    chosen = LIGHT
    for tier in ORDER:
        cost = cost_ms.get(tier.key)
        if cost is None or cost > limit:
            break
        chosen = tier
    return chosen
