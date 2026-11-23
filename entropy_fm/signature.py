"""
The signature sound of a room: its lead and its bass or drone in each style are the same on every day.

Each (style, layer) has a pool of voices that were approved by ear. A room takes the member that a stable hash of
the room, the style and the layer points at; everything else in a piece keeps varying from day to day. Voices that
sit at different loudness at the same velocity carry a trim in decibels, so that every member of a pool plays at
the level of the voice the mix was balanced with.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from .score import (
    LAYER_BASS,
    LAYER_DRONE,
    LAYER_KEYS,
    LAYER_LEAD,
    LAYER_PIANO,
    LAYER_PLUCK,
    LAYER_PLUCKS,
    LAYER_SEQ_A,
    LAYER_STR,
)
from .styles import STYLE_AMBIENT, STYLE_BERLIN_SCHOOL, STYLE_GLITCH, STYLE_LOFI, STYLE_RITUAL
from .timeline import stable_hash
from .voices import VELOCITY_POWER

if TYPE_CHECKING:
    from .plan import Plan
    from .score import Layer

# decibels per decade of amplitude: gain is 10 ** (db / 20)
AMPLITUDE_DB_DIVISOR = 20.0
# the loudest a note may be after its trim
MAX_VELOCITY = 1.0


class SigVoice(NamedTuple):
    """A member of a pool: the voice, its patch (None keeps the patch of the day) and its trim in decibels."""

    voice: str
    patch: int | None = None
    trim_db: float = 0.0

    @property
    def label(self) -> str:
        """Return the voice and its fixed patch as text, for the log of the choices."""
        return self.voice if self.patch is None else f"{self.voice} p{self.patch}"


# the trims are the level-matching gains of the listening run, whole house on 2026-11-16, against the voice that
# the layer was balanced with before the rooms had signatures
POOLS: dict[str, dict[str, tuple[SigVoice, ...]]] = {
    STYLE_AMBIENT: {
        LAYER_PIANO: (SigVoice("grand_piano"), SigVoice("felt_piano", None, 24.0)),
        LAYER_DRONE: (
            SigVoice("bowed_pad"),
            SigVoice("drawbar_organ", 0, 1.2),
            SigVoice("string_machine", None, -0.6),
        ),
    },
    STYLE_BERLIN_SCHOOL: {
        LAYER_LEAD: (
            SigVoice("analog_lead", 0, -0.4),
            SigVoice("analog_lead", 1, 0.7),
            SigVoice("juno_pad", None, -0.9),
        ),
        LAYER_SEQ_A: (
            SigVoice("moog_seq"),
            SigVoice("moog_bass", None, -2.1),
            SigVoice("acid_bass", 0, -1.3),
        ),
    },
    STYLE_LOFI: {
        # under their level match, by ear: the felt piano by 3 dB, the organ by 9
        LAYER_KEYS: (SigVoice("felt_piano", None, 5.1), SigVoice("drawbar_organ", 1, -5.3)),
        LAYER_PLUCK: (SigVoice("strum_pluck", None, -4.7), SigVoice("juno_pluck", None, -8.6)),
        LAYER_BASS: (SigVoice("moog_bass"), SigVoice("sub_round", 0, 0.7)),
    },
    STYLE_RITUAL: {
        LAYER_PLUCKS: (SigVoice("strum_pluck"), SigVoice("felt_piano", None, 5.5)),
        LAYER_LEAD: (SigVoice("chant_cello"), SigVoice("tape_strings", 1, 1.2)),
        LAYER_DRONE: (
            SigVoice("juno_pad"),
            SigVoice("bowed_pad", None, 1.7),
            SigVoice("drawbar_organ", 0, -0.3),
        ),
    },
    STYLE_GLITCH: {
        LAYER_STR: (
            SigVoice("tape_strings"),
            SigVoice("string_machine", None, 2.2),
            SigVoice("tape_choir", 0, 2.1),
        ),
        LAYER_BASS: (
            SigVoice("glitch:growl"),
            SigVoice("reese", 0, 1.3),
            SigVoice("fm_growl", 0, 3.8),
        ),
    },
}
# a pool that is picked with a seed of its own, chosen by ear so that the rooms of the house split between its voices
POOL_SEEDS: dict[tuple[str, str], int] = {(STYLE_LOFI, LAYER_KEYS): 2}


def gain_of_db[Decibels: (float, np.ndarray)](db: Decibels) -> Decibels:
    """
    Return the gain that a level in decibels stands for.

    :param db: The level in decibels, or an array of them.
    """
    gain: Decibels = 10.0 ** (db / AMPLITUDE_DB_DIVISOR)
    return gain


def signature(plan: Plan, layer: str, choice: str) -> SigVoice:
    """
    Return the signature voice of the room for a layer, and note it in the choices of the piece.

    :param plan: The plan of the piece, which knows the room and the style.
    :param layer: The layer with a signature, such as `lead`.
    :param choice: The name under which the choice is noted in `plan.choices`.
    """
    pool = POOLS[plan.style][layer]
    seed = POOL_SEEDS.get((plan.style, layer))
    parts = (
        (plan.area_key, plan.style, layer)
        if seed is None
        else (plan.area_key, plan.style, layer, seed)
    )
    chosen = pool[stable_hash("signature", *parts) % len(pool)]
    plan.choices[choice] = chosen.label
    return chosen


def trim_layer(layer: Layer, sig: SigVoice) -> None:
    """
    Scale the gain of a layer by the trim of its signature voice.

    :param layer: The layer whose notes are all played by the voice.
    :param sig: The signature voice of the layer.
    """
    layer.gain *= gain_of_db(sig.trim_db)


def trim_velocity(sig: SigVoice, velocity: float) -> float:
    """
    Return the velocity that plays a voice at the level of its trim, for a layer that mixes voices.

    :param sig: The signature voice.
    :param velocity: The velocity before the trim.
    """
    return float(
        min(velocity * 10.0 ** (sig.trim_db / AMPLITUDE_DB_DIVISOR / VELOCITY_POWER), MAX_VELOCITY)
    )


def patch_of(sig: SigVoice, day_patch: int) -> int:
    """
    Return the patch a note plays: the fixed one of the voice, or the one of the day.

    :param sig: The signature voice.
    :param day_patch: The patch that the piece uses for the layer.
    """
    return day_patch if sig.patch is None else sig.patch
