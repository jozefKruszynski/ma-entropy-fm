"""Tests of the Auto setting of the render quality: which style it measures and which tier it picks."""

import asyncio
import sys
from unittest.mock import MagicMock

import pytest

from entropy_fm import autotune
from entropy_fm.quality import AUTO_BUDGET, LIGHT, ORDER, RICH, STANDARD, choose_tier
from entropy_fm.styles import STYLES
from music_assistant.mass import MusicAssistant
from tests.test_library_provider import _quiet
from tests.test_provider import _setup


def test_the_dearest_offered_style_is_the_one_that_is_measured() -> None:
    """Cinematic Glitch when it is offered, then Berlin School, Ritual, Lo-fi and Ambient."""
    assert autotune.heaviest_style(STYLES) == "glitch"
    assert autotune.heaviest_style(["ambient", "berlin_school", "lofi"]) == "berlin_school"
    assert autotune.heaviest_style(["ambient", "ritual", "lofi"]) == "ritual"
    assert autotune.heaviest_style(["ambient"]) == "ambient"
    assert autotune.heaviest_style([]) == "glitch"
    assert set(autotune.COST_ORDER) == set(STYLES)


def test_a_slow_machine_gets_a_tier_that_the_dearest_style_fits() -> None:
    """With the costs of the dearest style, no tier is picked that goes over the share of a core."""
    limit = AUTO_BUDGET * 1000.0
    # Berlin School alone would have let Rich through on this machine; Cinematic Glitch costs more
    berlin = {"light": 150.0, "standard": 250.0, "rich": 280.0}
    glitch = {"light": 190.0, "standard": 330.0, "rich": 300.0}
    assert choose_tier(berlin) is RICH
    assert choose_tier(glitch) is LIGHT
    assert choose_tier({"light": 100.0, "standard": 280.0, "rich": 400.0}) is STANDARD
    assert glitch["light"] < limit


@pytest.mark.parametrize("style", ["glitch", "ambient"])
def test_the_benchmark_measures_every_tier_in_the_style_it_is_given(style: str) -> None:
    """A short render of the invented house gives a cost per tier, richer tiers costing more than Light."""
    costs = autotune.benchmark(ORDER, style, 3)
    assert set(costs) == {tier.key for tier in ORDER}
    assert all(cost > 0.0 for cost in costs.values())
    assert costs["standard"] > costs["light"]


async def test_the_provider_benchmarks_the_dearest_offered_style_and_keeps_it_per_style(
    mass: MusicAssistant, fake_hass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Auto asks for the dearest style that is offered, caches under it, and picks from its costs."""
    provider, _ = await _setup(mass, fake_hass)
    # the benchmark that the provider starts by itself is not what this test is about
    assert provider._benchmark_task is not None  # noqa: SLF001
    provider._benchmark_task.cancel()  # noqa: SLF001
    await asyncio.wait({provider._benchmark_task})  # noqa: SLF001
    module = sys.modules[type(provider).__module__]
    asked: list[str] = []

    def fake(_tiers: object, style: str, _seconds: int = 0) -> dict[str, float]:
        asked.append(style)
        return {"light": 100.0, "standard": 200.0, "rich": 400.0}

    monkeypatch.setattr(module, "benchmark", fake)
    await mass.cache.clear()
    _quiet(provider, monkeypatch, render_quality="auto")
    await provider._pick_quality()  # noqa: SLF001
    assert asked == ["glitch"]
    assert provider.quality.key == STANDARD.key
    # the same style again is a cache hit; another set of styles is another measurement
    await provider._pick_quality()  # noqa: SLF001
    assert asked == ["glitch"]
    _quiet(provider, monkeypatch, render_quality="auto", styles=["ambient"])
    await provider._pick_quality()  # noqa: SLF001
    assert asked == ["glitch", "ambient"]
    assert LIGHT.cache_mb < STANDARD.cache_mb
