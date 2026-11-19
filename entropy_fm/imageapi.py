"""
The image engines of Music Assistant, behind a guard.

The image engine API exists only on newer servers (the dev branch); on a stable 2.11 the helpers
are not there. Every access goes through this module, which falls back to "no engines" on an
import error or any other failure, so the provider loads on every channel and covers are drawn
when the server cannot make them.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ConfigEntry

    from music_assistant.mass import MusicAssistant

LOGGER = logging.getLogger(__name__)


def _module() -> Any | None:
    """Return the module of the server with the engine helpers, or None when it has none."""
    try:
        from music_assistant.helpers import plugin_engines  # noqa: PLC0415

        module: Any = plugin_engines
        _ = module.get_image_engines
    except Exception:  # noqa: BLE001 - ImportError, AttributeError, whatever else
        return None
    return module


async def engines(mass: MusicAssistant) -> list[Any]:
    """
    Return the image engines of the loaded plugins, or an empty list.

    :param mass: The server.
    """
    module = _module()
    if module is None:
        return []
    try:
        return list(await module.get_image_engines(mass))
    except Exception:  # noqa: BLE001 - an engine that fails to list is no engine
        return []


async def resolve(mass: MusicAssistant, setting: str | None) -> Any | None:
    """
    Return the engine that a setting picks (an empty setting follows the default), or None.

    Servers differ: some have `resolve_image_engine_setting`, some only `resolve_image_engine`
    (which takes an engine uid), some neither; the first of the engines is the last resort.

    :param mass: The server.
    :param setting: The stored value of the engine setting.
    """
    module = _module()
    if module is None:
        return None
    try:
        setter = getattr(module, "resolve_image_engine_setting", None)
        if setter is not None:
            return await setter(mass, setting)
        if setting and setting != "global":
            return await module.resolve_image_engine(mass, setting)
        found = await module.get_image_engines(mass)
        return found[0] if found else None
    except Exception:  # noqa: BLE001
        return None


async def picker_entries(
    mass: MusicAssistant, key: str, depends_on: str, depends_on_value: str
) -> tuple[ConfigEntry, ...]:
    """
    Return the settings entries that let the user choose an image engine, or none.

    The helper of the server makes them when it has one; otherwise a plain picker is built from the
    list of engines. With no engines there is nothing to pick.

    :param mass: The server.
    :param key: The key of the setting.
    :param depends_on: The setting the picker belongs to.
    :param depends_on_value: The value of that setting that shows the picker.
    """
    module = _module()
    if module is None or not await engines(mass):
        return ()
    try:
        maker = getattr(module, "create_image_engine_config_entries", None)
        if maker is not None:
            return tuple(
                await maker(mass, key, depends_on=depends_on, depends_on_value=depends_on_value)
            )
    except Exception:
        LOGGER.debug("The server could not make the image engine picker", exc_info=True)
    from music_assistant_models.config_entries import (  # noqa: PLC0415
        ConfigEntry,
        ConfigValueOption,
    )
    from music_assistant_models.enums import ConfigEntryType  # noqa: PLC0415

    options = [ConfigValueOption("global", "Default image engine")]
    options += [
        ConfigValueOption(str(engine.uid), f"{engine.provider.name} | {engine.name}")
        for engine in await engines(mass)
    ]
    return (
        ConfigEntry(
            key=key,
            type=ConfigEntryType.STRING,
            default_value="global",
            required=False,
            options=options,
            depends_on=depends_on,
            depends_on_value=depends_on_value,
            category="advanced",
        ),
    )


async def generate(mass: MusicAssistant, setting: str | None, prompt: str) -> bytes:
    """
    Make an image with the engine of a setting.

    :param mass: The server.
    :param setting: The stored value of the engine setting.
    :param prompt: The description of the image.
    :raises RuntimeError: When there is no engine.
    """
    engine = await resolve(mass, setting)
    if engine is None:
        raise RuntimeError("no image engine")
    return bytes(await engine.provider.ai_generate_image(prompt, engine.id))
