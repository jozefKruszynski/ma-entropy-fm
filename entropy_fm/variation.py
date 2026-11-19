"""
Variation from day to day: which member of a family of material a day plays.

A style has families of composed material (sequence cells, progressions, voices, patterns). Which member plays on a
day is decided from the style, the room and the date alone, so it is the same every time that day is played, and
it is chosen so that a member never plays on two days in a row and comes back as late as the size of the family
allows: the days are dealt out in rounds, and every round is a shuffle of the whole family.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from .timeline import stable_hash


def _round_order(style: str, round_index: int, count: int) -> list[int]:
    """Return the order in which the members play in one round: a shuffle of the whole family."""
    rng = np.random.default_rng([stable_hash(style, "round") % 2**32, round_index])
    return [int(index) for index in rng.permutation(count)]


def variant_of(style: str, day: date, count: int) -> int:
    """
    Return which member of a family plays on a day.

    Every round of `count` days plays each member once, in an order of its own, and the first member of a round is
    never the last one of the round before.

    :param style: What the family belongs to: the style, the room and the part, as text.
    :param day: The day.
    :param count: How many members the family has.
    """
    if count <= 1:
        return 0
    if count == 2:
        return (day.toordinal() + stable_hash(style)) % 2
    round_index, place = divmod(day.toordinal(), count)
    order = _round_order(style, round_index, count)
    if round_index > 0 and order[0] == _round_order(style, round_index - 1, count)[-1]:
        order[0], order[1] = order[1], order[0]
    return order[place]
