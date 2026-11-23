"""
The gain staging of the layers: how much each layer of each style is trimmed so that the mix is balanced before the master.

Every voice is brought to the same loudness by its own constant gain (`voices`), but a layer is more than one voice:
a sequence of sixteenth notes is louder than a piano that plays now and then. `scripts/gain_staging.py` renders
the layers of the demo days one by one, measures their level before the master and writes the trims in decibels
that bring each to the level of the design (`DESIGN_DB`). The leveler and the limiter then only have to hold the loudness, and the
limiter barely works.

The module imports nothing from Music Assistant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .drums import DRUM_PREFIX
from .score import CURVE_RATE
from .voices import REF_VELOCITY, VELOCITY_POWER

if TYPE_CHECKING:
    from .score import Score

# the level of each layer before the master, RMS in dBFS while it plays alone: the balance of the mix
DESIGN_DB: dict[str, dict[str, float]] = {
    "ambient": {"pads": -23.0, "piano": -38.0, "drone": -28.0, "strings": -27.0, "growl": -31.0},
    "berlin_school": {
        "seqA": -25.0,
        "seqB": -27.0,
        "seqC": -31.0,
        "strings": -27.0,
        "choir": -31.0,
        "bed": -29.0,
        "lead": -28.0,
        "pulse": -34.0,
        "toms": -32.0,
        "shaker": -39.0,
        "growl": -28.0,
    },
    "lofi": {
        "keys": -23.0,
        "bass": -31.0,
        "drums": -28.0,
        "chop": -30.0,
        "pluck": -31.0,
        "strings": -31.0,
    },
    "ritual": {
        "drone": -22.0,
        "sub": -29.0,
        "wash": -29.0,
        "plucks": -26.0,
        "slit": -30.0,
        "block": -37.0,
        "clack": -29.0,
        "lead": -27.0,
        "taiko": -27.0,
        "heart": -33.0,
    },
}
# the trims in dB that scripts/gain_staging.py measured; zero where nothing was measured yet
TRIMS: dict[str, dict[str, float]] = {
    "ambient": {"pads": -6.1, "piano": 1.8, "drone": -4.5},
    "berlin_school": {
        "seqA": -1.9,
        "seqB": -10.7,
        "seqC": -2.5,
        "strings": -10.7,
        "choir": -13.0,
        "bed": -4.9,
        "lead": -5.2,
        "pulse": -4.3,
        "toms": -0.9,
        "shaker": 16.5,
        "growl": -0.7,
    },
    "lofi": {
        "keys": -6.0,
        "bass": -8.0,
        "drums": 1.2,
        "chop": -8.3,
        "pluck": -5.2,
        "strings": -9.4,
    },
    "ritual": {
        "drone": -2.3,
        "sub": -2.5,
        "wash": -2.8,
        "plucks": -7.0,
        "slit": -16.2,
        "block": -21.2,
        "clack": -19.9,
        "lead": -4.1,
        "taiko": -4.0,
        "heart": -2.6,
    },
}
# the energy of a note of each voice at velocity 0.75, in mean-square seconds, as a + b * the seconds it is held
VOICE_ENERGY: dict[str, tuple[float, float]] = {
    "sub_round": (0.00129, 0.00761),
    "moog_bass": (0.0, 0.01231),
    "reese": (0.00023, 0.00615),
    "fm_growl": (0.0, 0.0084),
    "acid_bass": (0.00045, 0.00794),
    "string_machine": (0.0, 0.00993),
    "juno_pad": (0.0, 0.00923),
    "supersaw": (0.00062, 0.00827),
    "tape_strings": (0.0, 0.01794),
    "tape_choir": (0.0, 0.01063),
    "glass_pad": (0.0, 0.01083),
    "bowed_pad": (0.0, 0.01157),
    "drawbar_organ": (0.0, 0.01117),
    "analog_lead": (0.0, 0.00923),
    "flute_lead": (0.00075, 0.0079),
    "fm_ep": (0.01361, 0.00224),
    "felt_piano": (0.01892, 0.0004),
    "grand_piano": (0.01959, 0.00022),
    "tx_ep": (0.01597, 0.00118),
    "moog_seq": (0.0, 0.00895),
    "blip_synth": (0.02025, 0.0),
    "chord_stab": (0.02023, 1e-05),
    "juno_pluck": (0.02477, 0.0),
    "nylon_ks": (0.00917, 0.0),
    "strum_pluck": (0.0166, 0.00096),
    "slit_drum": (0.02025, 0.0),
    "woodblock": (0.02025, 0.0),
    "chant_cello": (0.0, 0.01046),
    "wash:cello": (0.0, 0.0052),
    "wash:violin": (0.0, 0.00529),
    "wash:felt": (0.0, 0.00546),
}
# the energy of a drum hit at velocity 0.8, in mean-square seconds
HIT_ENERGY: dict[str, float] = {
    "kick_soft": 0.002315,
    "kick_deep": 0.004761,
    "kick_lofi": 0.00146,
    "snare_lofi": 0.000559,
    "rim": 0.000109,
    "hat_closed": 7e-06,
    "hat_open": 3.4e-05,
    "shaker": 1.3e-05,
    "click": 0.000141,
    "tom_soft": 0.00163,
    "odaiko": 0.004351,
    "shime": 0.000602,
    "taiko_rim": 4.7e-05,
    "heart": 0.001252,
}
# a note keeps sounding this long after its key goes up, for the estimate of the loudness
TAIL_SECONDS = 0.8
# the velocity that `HIT_ENERGY` was measured at, the energy of a hit that is not in it, and how long a hit counts for
HIT_ENERGY_VELOCITY = 0.8
DEFAULT_HIT_ENERGY = 0.002
HIT_SPAN_SECONDS = 0.25
# the energy of a voice that is not in `VOICE_ENERGY`, and the least that a note has
DEFAULT_VOICE_ENERGY = (0.01, 0.01)
MIN_NOTE_ENERGY = 1e-6
# a texture is this share of its gain in amplitude
TEXTURE_AMPLITUDE_SHARE = 0.5
POWER_DB_FACTOR = 10.0
MEAN_SQUARE_FLOOR = 1e-9
# the gain that holds the level is kept within these dB
GAIN_MIN_DB = -9.0
GAIN_MAX_DB = 12.0
# the second, narrower smoothing of the gain, in seconds
GAIN_SMOOTH_SECONDS = 3.0
SMOOTH_SECONDS = 8.0


def trim_db(style: str, layer: str) -> float:
    """
    Return the trim of a layer in decibels.

    :param style: The style key.
    :param layer: The key of the layer.
    """
    return TRIMS.get(style, {}).get(layer, 0.0)


def feed_forward(
    score: Score, target_db: float, smooth_seconds: float = SMOOTH_SECONDS
) -> np.ndarray:
    """
    Estimate the loudness of a score from its notes and return the gain in dB, a value every tenth of a second, that holds it.

    A voice is as loud as its calibration says (`VOICE_ENERGY`), a layer as its gain and presence say. The sum over the
    notes that sound at a moment, smoothed over a few seconds, is the mean square that the master would see;
    the gain brings it to the target. It needs no audio, so it is known before the first second plays, it is the same
    for every seek, and it does not pump: a sparse night is lifted and a dense afternoon is held down by the same rule.

    :param score: The score, with the gains of its layers set.
    :param target_db: The mean square level to hold, in dB (RMS dBFS).
    :param smooth_seconds: The width of the smoothing of the estimate.
    """
    count = score.length * CURVE_RATE + 1
    rate = np.zeros(count + 1)
    for note in score.notes:
        layer = score.layers[note.layer]
        if note.voice.startswith(DRUM_PREFIX):
            base = (
                HIT_ENERGY.get(note.voice[len(DRUM_PREFIX) :], DEFAULT_HIT_ENERGY)
                * (note.vel / HIT_ENERGY_VELOCITY) ** 2
            )
            span = HIT_SPAN_SECONDS
        else:
            a, b = VOICE_ENERGY.get(note.voice, DEFAULT_VOICE_ENERGY)
            base = max(a + b * note.dur, MIN_NOTE_ENERGY) * (note.vel / REF_VELOCITY) ** (
                2.0 * VELOCITY_POWER
            )
            span = note.dur + TAIL_SECONDS
        present = layer.gain
        if layer.presence is not None:
            present *= float(layer.presence[min(int(note.t * CURVE_RATE), len(layer.presence) - 1)])
        first = int(note.t * CURVE_RATE)
        last = min(int((note.t + span) * CURVE_RATE) + 1, count)
        if first >= count:
            continue
        rate[first] += base * present**2 / span
        rate[last] -= base * present**2 / span
    mean_square = np.cumsum(rate)[:count]
    for texture in score.textures:
        mean_square = mean_square + (texture.gain * TEXTURE_AMPLITUDE_SHARE) ** 2
    width = int(smooth_seconds * CURVE_RATE)
    kernel = np.hanning(width * 2 + 1)
    padded = np.concatenate(
        (np.full(width, mean_square[0]), mean_square, np.full(width, mean_square[-1]))
    )
    smooth = np.convolve(padded, kernel / kernel.sum(), mode="same")[width:-width]
    gain = np.clip(
        target_db - POWER_DB_FACTOR * np.log10(smooth + MEAN_SQUARE_FLOOR), GAIN_MIN_DB, GAIN_MAX_DB
    )
    narrow = int(GAIN_SMOOTH_SECONDS * CURVE_RATE)
    padded = np.concatenate((np.full(narrow, gain[0]), gain, np.full(narrow, gain[-1])))
    kernel = np.hanning(narrow * 2 + 1)
    return np.asarray(np.convolve(padded, kernel / kernel.sum(), mode="same")[narrow:-narrow])
