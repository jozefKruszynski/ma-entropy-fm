"""
Entropy FM for Music Assistant.

A music source that turns the history of your Home Assistant into generative soundscapes.
Every area of the house is an artist, every day of an area an album, and the track of an album
is one long, slowly evolving piece in each of five styles. The house steers the texture, the
density, the brightness, the space and the movement of the piece; it never writes the notes. The
audio is synthesised while it plays, from samples that ship with the provider; the audio of past
days may be kept in its own cache, and nothing leaves the server.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .provider import SUPPORTED_FEATURES, EntropyFMProvider

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType

__all__ = ["EntropyFMProvider"]


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize the provider for one settings entry."""
    return EntropyFMProvider(mass, manifest, config, SUPPORTED_FEATURES)
