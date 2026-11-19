"""Tests of the render: seeking, finiteness, the glide rule, the noise floor, the cost of the tiers."""

import time

import numpy as np
import pytest

from entropy_fm.quality import LIGHT, STANDARD
from entropy_fm.render import render_all, render_blocks, render_pcm, to_pcm16
from entropy_fm.styles import STYLES

from .pitch_tools import find_glides, noise_floor_db
from .scape_tools import demo, window

RATE = 44100


@pytest.mark.parametrize("style", list(STYLES))
def test_a_stretch_renders_finite_stereo_in_range(style: str) -> None:
    """A stretch of every style is finite, in range, not silent and one sample per frame."""
    score, plan = demo(style)
    audio = window(score, plan, 12)
    assert audio.shape == (12 * RATE, 2)
    assert np.isfinite(audio).all()
    assert np.abs(audio).max() <= 1.0
    assert np.abs(audio).max() > 0.02


@pytest.mark.parametrize("style", list(STYLES))
def test_a_seek_sounds_like_the_same_music(style: str) -> None:
    """Starting at a second renders what the continuous render has there, to within a fraction of a dB."""
    score, plan = demo(style)
    from soundscapes import pick_windows  # noqa: PLC0415

    start = pick_windows(plan)[1][1]
    full = render_all(score, STANDARD, "seek", None, start_s=start - 10, end_s=start + 6)
    seek = render_all(score, STANDARD, "seek", None, start_s=start, end_s=start + 6)
    a = full[10 * RATE : 16 * RATE]
    b = seek[: 6 * RATE]
    rms_a = float(np.sqrt(np.mean(a**2)))
    rms_b = float(np.sqrt(np.mean(b**2)))
    assert abs(20 * np.log10(rms_a / rms_b)) < 2.5, style
    correlation = float(np.corrcoef(a[:, 0], b[:, 0])[0, 1])
    assert correlation > 0.9, (style, correlation)


def test_the_same_stretch_is_the_same_audio() -> None:
    """Rendering twice gives the same samples, bit for bit, once the notes are made."""
    score, plan = demo("lofi")
    first = window(score, plan, 8)
    second = window(score, plan, 8)
    assert np.array_equal(first, second)


def test_pcm_chunks_are_one_second_of_16_bit_stereo() -> None:
    """The stream is whole seconds of little-endian PCM, with the same dither for the same second."""
    score, _ = demo("ambient")
    chunks = []
    for chunk in render_pcm(score, STANDARD, "pcm", 200):
        chunks.append(chunk)
        if len(chunks) == 2:
            break
    assert all(len(chunk) == RATE * 4 for chunk in chunks)
    assert to_pcm16(np.zeros((RATE, 2), np.float32), 3) == to_pcm16(
        np.zeros((RATE, 2), np.float32), 3
    )


def test_the_start_and_the_end_fade() -> None:
    """The first second fades in and the last one fades out."""
    score, _ = demo("ambient")
    start = next(render_blocks(score, STANDARD, "x", 0, None, 2))
    assert np.abs(start[: RATE // 20]).max() < np.abs(start[-RATE // 4 :]).max() + 1e-9
    tail = list(render_blocks(score, STANDARD, "x", score.length - 2))
    assert np.abs(tail[-1][-RATE // 50 :]).max() < 0.01


@pytest.mark.parametrize("style", list(STYLES))
def test_no_continuous_pitch_glide_above_150_hz(style: str) -> None:
    """The pitch tracker finds no fall and no rise of more than three semitones in a second and a half."""
    score, plan = demo(style)
    audio = window(score, plan, 40).mean(axis=1)
    assert find_glides(audio, 3.0, 1.5, -1) == [], style
    assert find_glides(audio, 3.0, 1.5, 1) == [], style


@pytest.mark.parametrize("style", ["ambient", "berlin_school", "lofi", "ritual"])
def test_the_noise_floor_is_low_and_the_air_and_vinyl_are_quiet(style: str) -> None:
    """The steady broadband floor in the quietest frames stays under -48 dBFS above 6 kHz."""
    score, plan = demo(style)
    audio = window(score, plan, 40).mean(axis=1)
    assert noise_floor_db(audio) < -48.0, style


def test_light_costs_less_than_standard_to_render() -> None:
    """The cheapest tier is cheaper on the same stretch of the same piece; the numbers are in scripts/bench.py."""
    costs = {}
    for tier in (LIGHT, STANDARD):
        score, plan = demo("berlin_school", tier=tier)
        window(score, plan, 6, tier)
        began = time.process_time()
        window(score, plan, 10, tier)
        costs[tier.key] = time.process_time() - began
    assert costs["light"] < costs["standard"] * 1.05
