"""A stand-in for the core Home Assistant plugin, with canned registries, states and history."""

from datetime import datetime
from typing import Any, NamedTuple
from unittest.mock import MagicMock

from music_assistant_models.enums import ProviderType


class RegistryEntry(NamedTuple):
    """The fields of the entity registry that the real plugin mirrors."""

    platform: str
    device_id: str | None
    area_id: str | None


AREAS: dict[str, Any] = {
    "kitchen": {"area_id": "kitchen", "name": "Kitchen", "aliases": [], "picture": None},
    "living_room": {
        "area_id": "living_room",
        "name": "Living Room",
        "aliases": [],
        "picture": None,
    },
    "garage": {"area_id": "garage", "name": "Garage", "aliases": [], "picture": None},
}
DEVICES = {
    "dev_fridge": {"id": "dev_fridge", "area_id": "kitchen"},
    "dev_lamp": {"id": "dev_lamp", "area_id": "living_room"},
}
ENTITIES = {
    # the area comes from the device
    "sensor.fridge_power": RegistryEntry("shelly", "dev_fridge", None),
    # the area is set on the entity and wins over the device
    "light.lamp": RegistryEntry("hue", "dev_lamp", "kitchen"),
    "binary_sensor.kitchen_motion": RegistryEntry("zha", None, "kitchen"),
    "binary_sensor.kitchen_door": RegistryEntry("zha", None, "kitchen"),
    "sensor.kitchen_temperature": RegistryEntry("zha", None, "kitchen"),
    "sensor.kitchen_humidity": RegistryEntry("zha", None, "kitchen"),
    "sensor.washing_machine_power": RegistryEntry("shelly", None, "kitchen"),
    # a German name, a device that is not named at all but carries a label, and a plain meter
    "sensor.geschirrspueler_leistung": RegistryEntry("shelly", None, "kitchen"),
    "sensor.kitchen_machine_power": RegistryEntry("shelly", None, "kitchen"),
    "light.living_room_ceiling": RegistryEntry("hue", None, "living_room"),
    "binary_sensor.living_room_window": RegistryEntry("zha", None, "living_room"),
    # private: never part of the music
    "person.jozef": RegistryEntry("person", None, "living_room"),
    "device_tracker.phone": RegistryEntry("mobile_app", None, "living_room"),
    "lock.front_door": RegistryEntry("nuki", None, "living_room"),
    "camera.garden": RegistryEntry("generic", None, "living_room"),
    "sensor.router_password": RegistryEntry("fritz", None, "living_room"),
    # no area at all
    "sensor.cellar_power": RegistryEntry("shelly", None, None),
    "sun.sun": RegistryEntry("sun", None, None),
}


def _state(entity_id: str, state: str, **attributes: Any) -> dict[str, Any]:
    return {"entity_id": entity_id, "state": state, "attributes": attributes}


STATES = [
    _state("sensor.fridge_power", "75", device_class="power", unit_of_measurement="W"),
    _state("light.lamp", "off", friendly_name="Counter lamp"),
    _state("binary_sensor.kitchen_motion", "off", device_class="motion", friendly_name="Motion"),
    _state("binary_sensor.kitchen_door", "off", device_class="door", friendly_name="Back door"),
    _state(
        "sensor.kitchen_temperature", "21", device_class="temperature", unit_of_measurement="°C"
    ),
    _state("sensor.kitchen_humidity", "50", device_class="humidity", unit_of_measurement="%"),
    _state(
        "sensor.washing_machine_power",
        "2",
        device_class="power",
        unit_of_measurement="W",
        friendly_name="Washing machine",
    ),
    _state(
        "sensor.geschirrspueler_leistung",
        "1",
        device_class="power",
        unit_of_measurement="W",
        friendly_name="Geschirrspüler Leistung",
    ),
    _state(
        "sensor.kitchen_machine_power",
        "1",
        device_class="power",
        unit_of_measurement="W",
        friendly_name="Kitchen machine",
    ),
    _state("light.living_room_ceiling", "on", friendly_name="Ceiling"),
    _state(
        "binary_sensor.living_room_window", "off", device_class="window", friendly_name="Window"
    ),
    _state("person.jozef", "home"),
    _state("device_tracker.phone", "home"),
    _state("lock.front_door", "locked"),
    _state("sensor.router_password", "1234"),
    _state("sun.sun", "above_horizon"),
]


def _events(entity_id: str, start: float, end: float) -> list[dict[str, Any]]:  # noqa: C901
    """Return canned history: offsets in seconds from the start of the day."""
    hour = 3600.0
    if entity_id == "sensor.fridge_power":
        pattern = [(0, "2"), (1, "80"), (1.2, "2"), (3, "82"), (3.3, "2"), (5, "78"), (5.4, "2")]
    elif entity_id == "light.lamp":
        pattern = [(0, "off"), (6.5, "on"), (8, "off"), (17.5, "on"), (22, "off")]
    elif entity_id == "binary_sensor.kitchen_motion":
        pattern = [(0, "off"), (7, "on"), (7.02, "off"), (7.2, "on"), (7.25, "off")]
    elif entity_id == "binary_sensor.kitchen_door":
        pattern = [(0, "off"), (7.1, "on"), (7.12, "off"), (12.4, "on"), (12.45, "off")]
    elif entity_id == "sensor.kitchen_temperature":
        pattern = [(0, "20.0"), (6, "20.4"), (12, "21.5"), (18, "22.8"), (23, "20.1")]
    elif entity_id == "sensor.kitchen_humidity":
        pattern = [(0, "45"), (8, "58"), (18, "64")]
    elif entity_id == "sensor.washing_machine_power":
        pattern = [(0, "1"), (10, "2000"), (10.3, "150"), (11, "400"), (11.5, "1")]
    elif entity_id in {"sensor.geschirrspueler_leistung", "sensor.kitchen_machine_power"}:
        pattern = [
            (0, "1"),
            (20.5, "2000"),
            (20.75, "90"),
            (21.5, "1900"),
            (21.7, "80"),
            (22.4, "1"),
        ]
    elif entity_id == "sun.sun":
        pattern = [(0, "below_horizon"), (7.3, "above_horizon"), (16.2, "below_horizon")]
    else:
        pattern = [(0, "off"), (19, "on"), (21, "off")]
    entries = []
    for offset, state in pattern:
        stamp = start + offset * hour
        if stamp <= end:
            entries.append({"s": state, "lu": max(stamp, start)})
    return entries


class FakeClient:
    """The websocket client behind the plugin."""

    connected = True

    def __init__(self) -> None:
        """Start without any recorded commands."""
        self.history_calls: list[dict[str, Any]] = []
        # when True the recorder has forgotten every day: the history is empty
        self.purged = False

    async def get_config(self) -> dict[str, Any]:
        """Return the configuration of the house."""
        return {"time_zone": "Europe/Vienna"}

    async def send_command(self, command: str, **kwargs: Any) -> dict[str, Any]:
        """Answer the history command with canned events."""
        if command == "config/label_registry/list":
            return [{"label_id": "lbl1", "name": "entropy_fm_dishwasher"}]  # type: ignore[return-value]
        assert command == "history/history_during_period"
        assert kwargs["minimal_response"] is True
        assert kwargs["no_attributes"] is True
        assert kwargs["significant_changes_only"] is False
        self.history_calls.append(kwargs)
        start = datetime.fromisoformat(kwargs["start_time"]).timestamp()
        end = datetime.fromisoformat(kwargs["end_time"]).timestamp()
        if self.purged:
            return {entity_id: [] for entity_id in kwargs["entity_ids"]}
        return {entity_id: _events(entity_id, start, end) for entity_id in kwargs["entity_ids"]}


async def _get_area_registry() -> dict[str, Any]:
    """Return the areas."""
    return AREAS


async def _get_entity_registry() -> dict[str, RegistryEntry]:
    """Return the entities."""
    return ENTITIES


async def _get_device_registry() -> dict[str, Any]:
    """Return the devices."""
    return DEVICES


async def _get_states(
    *,
    entity_ids: list[str] | None = None,
    domains: Any = None,  # noqa: ARG001
) -> list[dict[str, Any]]:
    """Return the states of the entities asked for."""
    wanted = set(entity_ids or [])
    return [state for state in STATES if state["entity_id"] in wanted]


async def _get_entity_registry_entries(entity_ids: list[str]) -> dict[str, Any]:
    """Return the full registry entries, with platform and labels."""
    return {
        entity_id: {
            "entity_id": entity_id,
            "platform": ENTITIES[entity_id].platform,
            "translation_key": None,
            "device_id": ENTITIES[entity_id].device_id,
            "labels": ["lbl1"] if entity_id == "sensor.kitchen_machine_power" else [],
        }
        for entity_id in entity_ids
        if entity_id in ENTITIES
    }


def make_fake_hass_plugin() -> MagicMock:
    """
    Return what the Home Assistant plugin offers, as far as Entropy FM uses it.

    The server looks at every provider it holds, so this is a mock that answers anything with a
    mock, with the real behaviour only where Entropy FM calls it.
    """
    plugin = MagicMock()
    plugin.type = ProviderType.PLUGIN
    plugin.domain = "hass"
    plugin.instance_id = "hass"
    plugin.available = True
    plugin.is_streaming_provider = False
    plugin.manifest.depends_on = None
    plugin.hass = FakeClient()
    plugin.get_area_registry = _get_area_registry
    plugin.get_entity_registry = _get_entity_registry
    plugin.get_device_registry = _get_device_registry
    plugin.get_states = _get_states
    plugin.get_entity_registry_entries = _get_entity_registry_entries
    return plugin
