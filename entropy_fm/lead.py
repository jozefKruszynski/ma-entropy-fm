"""
Who leads a phrase: the rule that every style shares, so that it reads the same everywhere.

Each style has a few layers that can carry a melody (its candidates). The room picks its own by a stable hash, the
same every day. In the daytime build, peak and plateau phrases the leading layer plays a real line and comes up a few
decibels while the other candidates step back and play sparser. A happening of the house hands the next phrase to
another candidate by what kind of sound it makes: a door or motion to the most plucked one, an appliance or the hob to
the rhythmic one, a light, a cover or the media to the sustained one. Handovers are spaced (at most one in three
phrases), never in a night phrase, and the lead then returns to the room's own.

The composers write the lines; this module plans who leads which phrase and shapes the levels.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from .plan import (
    BUILD,
    CLIMAX,
    PLATEAU,
    TRIGGER_APPLIANCE,
    TRIGGER_COVER,
    TRIGGER_DOOR,
    TRIGGER_HOB,
    TRIGGER_LIGHT,
    TRIGGER_MEDIA,
    TRIGGER_MOTION,
)
from .score import CURVE_RATE
from .signature import gain_of_db
from .timeline import stable_hash

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .plan import Plan
    from .score import Layer, Note

LEAD_CHOICE = "lead"
LEAD_HANDOVERS_CHOICE = "lead handovers"
# the others keep this share of their notes in a phrase that another layer leads
LEAD_BACK_KEEP = 0.7
# the kinds of sound a candidate can have, which is what a happening of the house asks for
PERCUSSIVE = "percussive"
RHYTHMIC = "rhythmic"
SUSTAINED = "sustained"
HANDOVER_TRAIT = {
    TRIGGER_DOOR: PERCUSSIVE,
    TRIGGER_MOTION: PERCUSSIVE,
    TRIGGER_APPLIANCE: RHYTHMIC,
    TRIGGER_HOB: RHYTHMIC,
    TRIGGER_LIGHT: SUSTAINED,
    TRIGGER_COVER: SUSTAINED,
    TRIGGER_MEDIA: SUSTAINED,
}
# the lead is taken in the build, the peak and the plateau, by day (the night curve is under this)
LEAD_ROLES = frozenset({BUILD, CLIMAX, PLATEAU})
LEAD_DAYTIME = 0.3
# the leading layer comes up this many dB and the others step back as many, with a short fade at the phrase edges
LEAD_STEP_DB = 3.0
LEAD_FADE_SECONDS = 1.0
# one handover in this many phrases
HANDOVER_GAP = 3


def default_lead(plan: Plan, candidates: Sequence[str]) -> str:
    """
    Return the lead of the room, which is the same on every day, and note it in the choices of the plan.

    :param plan: The plan of the piece.
    :param candidates: The layers that can lead in the style.
    """
    lead = candidates[stable_hash("lead", plan.area_key, plan.style) % len(candidates)]
    plan.choices[LEAD_CHOICE] = lead
    return lead


def daytime_phrases(
    plan: Plan, sections: Sequence[tuple[float, float, str, float]], phrase: float
) -> list[bool]:
    """
    Return for each phrase whether it can have a lead: a build, peak or plateau by day.

    :param plan: The plan of the piece.
    :param sections: The form of the piece (`Plan.sections`).
    :param phrase: The length of a phrase in seconds.
    """
    starts = [section[0] for section in sections]
    found = []
    for index in range(math.ceil(plan.length / phrase)):
        start = index * phrase
        role = sections[max(bisect_right(starts, start) - 1, 0)][2]
        found.append(role in LEAD_ROLES and plan.at(plan.night, start) < LEAD_DAYTIME)
    return found


@dataclass(frozen=True)
class Leads:
    """Who leads each phrase of a piece (None where nobody does) and the phrases that a happening handed over."""

    phrases: list[str | None]
    phrase: float
    handed: list[int]

    def at(self, second: float) -> str | None:
        """Return who leads at a second."""
        return self.phrases[min(int(second / self.phrase), len(self.phrases) - 1)]


def phrase_leads(
    plan: Plan,
    phrase: float,
    default: str,
    allowed: Sequence[bool],
    by_trait: dict[str, str],
) -> Leads:
    """
    Return who leads each phrase and which phrases a happening handed over.

    :param plan: The plan, whose triggers hand a phrase to a layer.
    :param phrase: The length of a phrase in seconds.
    :param default: The lead of the room.
    :param allowed: For each phrase whether it can have a lead; a handover only goes to such a phrase.
    :param by_trait: The candidate of each kind of sound (`PERCUSSIVE`, `RHYTHMIC`, `SUSTAINED`).
    """
    phrases: list[str | None] = [default if ok else None for ok in allowed]
    handed: list[int] = []
    for second, _strength, kind in sorted(plan.triggers):
        trait = HANDOVER_TRAIT.get(kind)
        layer = by_trait.get(trait) if trait else None
        index = int(second / phrase) + 1
        if (
            layer is None
            or index >= len(phrases)
            or phrases[index] in {None, layer}
            or (handed and index - handed[-1] < HANDOVER_GAP)
        ):
            continue
        phrases[index] = layer
        handed.append(index)
    return Leads(phrases, phrase, handed)


def lead_curves(plan: Plan, leads: Leads, candidates: Sequence[str]) -> dict[str, np.ndarray]:
    """Return for each candidate a curve that is 1 where it leads, -1 where another one does and 0 elsewhere, faded."""
    curves = {layer: np.zeros(plan.count) for layer in candidates}
    for index, lead in enumerate(leads.phrases):
        if lead is None:
            continue
        span = slice(
            int(index * leads.phrase * CURVE_RATE), int((index + 1) * leads.phrase * CURVE_RATE)
        )
        for layer in candidates:
            curves[layer][span] = 1.0 if layer == lead else -1.0
    width = max(int(LEAD_FADE_SECONDS * CURVE_RATE), 1)
    kernel = np.ones(width) / width
    return {
        layer: np.convolve(np.pad(curve, width, mode="edge"), kernel, mode="same")[width:-width]
        for layer, curve in curves.items()
    }


def apply_curves(
    layers: dict[str, Layer],
    curves: dict[str, np.ndarray],
    step_db: float,
    trim_db: dict[str, float] | None = None,
) -> None:
    """
    Bring the leading layer up and the others down by `step_db` through their gain and presence.

    :param layers: The layers of the score, changed in place.
    :param curves: The lead curves (`lead_curves`).
    :param step_db: How far the lead comes up and the others step back.
    :param trim_db: Per layer, what to add to its level while it leads, so that every lead sits as far over the rest.
    """
    trims = trim_db or {}
    lifted = gain_of_db(step_db)
    for key, curve in curves.items():
        layers[key].gain *= lifted
        presence = layers[key].presence
        layers[key].presence = (1.0 if presence is None else presence) * gain_of_db(
            step_db * (curve - 1.0) + trims.get(key, 0.0) * np.clip(curve, 0.0, 1.0)
        )


def back_off(
    notes: list[Note],
    leads: Leads,
    layers: frozenset[str],
    keep: float,
    rng: np.random.Generator,
) -> list[Note]:
    """
    Return the notes with the share `1 - keep` of those of the other candidates left out in a phrase that has a lead.

    :param notes: All notes of the piece.
    :param leads: Who leads each phrase.
    :param layers: The layers that play sparser when another one leads.
    :param keep: The share of their notes that stays.
    :param rng: The generator that picks which notes stay.
    """
    return [
        note
        for note in notes
        if note.layer not in layers or leads.at(note.t) in {None, note.layer} or rng.random() < keep
    ]


class LedPiece(Protocol):
    """A piece of a style with leads: the notes it has made, how long it plays and who leads each phrase."""

    notes: list[Note]
    length: int
    leads: Leads
    plan: Plan


def back_off_followers(piece: LedPiece, layers: frozenset[str], label: str) -> None:
    """
    Drop the notes of a piece past its end and thin out the layers that do not lead.

    :param piece: The piece, whose notes are replaced.
    :param layers: The layers that play sparser when another one leads.
    :param label: The name of the random stream that picks which notes stay.
    """
    piece.notes = back_off(
        [n for n in piece.notes if n.t < piece.length],
        piece.leads,
        layers,
        LEAD_BACK_KEEP,
        piece.plan.rng(label),
    )


def summary(default: str, leads: Leads) -> str:
    """Return the line of the notes about a piece that says who led and when it changed hands."""
    when = ", ".join(f"{index * leads.phrase:.0f} s" for index in leads.handed) or "none"
    return f"Lead: the {default} by default, handed over {len(leads.handed)} times ({when})."
