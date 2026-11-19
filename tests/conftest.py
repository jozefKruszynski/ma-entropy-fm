"""
Enable the server's test fixture.

`mass` gives a started server in dev mode with this repository as its local provider folder.
"""

from collections.abc import AsyncGenerator
from unittest.mock import MagicMock

import pytest

from music_assistant.mass import MusicAssistant
from tests.fake_hass import make_fake_hass_plugin

pytest_plugins = ["music_assistant.testing"]


@pytest.fixture
async def fake_hass(mass: MusicAssistant) -> AsyncGenerator[MagicMock]:
    """Put a fake Home Assistant plugin in the server for the length of a test."""
    fake = make_fake_hass_plugin()
    # a test-only shortcut: the server keeps its providers in this mapping
    mass._providers["hass"] = fake  # noqa: SLF001
    yield fake
    mass._providers.pop("hass", None)  # noqa: SLF001
