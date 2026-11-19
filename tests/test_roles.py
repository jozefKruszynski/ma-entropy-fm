"""Tests of the layered classifier that decides which voice an entity gets."""

import pytest

from entropy_fm.roles import EntityMeta, classify, parse_overrides
from entropy_fm.timeline import VoiceKind


def _kind(entity_id: str, **kwargs: object) -> tuple[VoiceKind, str] | None:
    """Classify a power sensor by default and return the kind and the source."""
    arguments: dict[str, object] = {
        "name": entity_id,
        "device_class": "power",
        "unit": "W",
        "state": "5",
    }
    arguments.update(kwargs)
    role = classify(entity_id, **arguments)  # type: ignore[arg-type]
    return None if role is None else (role.kind, role.source)


@pytest.mark.parametrize(
    ("entity_id", "device_class", "unit", "state", "expected"),
    [
        ("sensor.fridge_power", "power", "W", "80", VoiceKind.DRONE),
        ("sensor.house_energy", "energy", "kWh", "3.1", VoiceKind.DRONE),
        ("sensor.temp", "temperature", "°C", "21", VoiceKind.TEMPERATURE),
        ("sensor.hum", "humidity", "%", "50", VoiceKind.HUMIDITY),
        ("sensor.co2", None, "ppm", "800", VoiceKind.PAD_PARTIAL),
        ("sensor.battery", "battery", "%", "90", None),
        ("sensor.status_text", None, None, "fine", None),
        ("binary_sensor.motion", "motion", None, "off", VoiceKind.PLUCK),
        ("binary_sensor.door", "door", None, "off", VoiceKind.PLUCK),
        ("binary_sensor.other", None, None, "off", VoiceKind.BLIP),
        ("binary_sensor.wifi", "connectivity", None, "on", None),
        ("light.ceiling", None, None, "on", VoiceKind.CHORD),
        ("switch.plug", None, None, "off", VoiceKind.CHORD),
        ("input_boolean.guest", None, None, "off", VoiceKind.BLIP),
        ("sun.sun", None, None, "above_horizon", VoiceKind.SUN),
        ("sensor.sun_solar_elevation", None, "°", "12", VoiceKind.SUN),
        ("cover.garage", None, None, "open", VoiceKind.COVER),
    ],
)
def test_device_class_layer(
    entity_id: str,
    device_class: str | None,
    unit: str | None,
    state: str,
    expected: VoiceKind | None,
) -> None:
    """Without hints the domain and device class decide, in any language."""
    role = classify(entity_id, entity_id, device_class, unit, state)
    assert (role.kind if role else None) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Dishwasher power", VoiceKind.DISHWASHER),
        ("Geschirrspüler Leistung", VoiceKind.DISHWASHER),
        ("Spülmaschine", VoiceKind.DISHWASHER),
        ("Vaatwasser", VoiceKind.DISHWASHER),
        ("Lave-vaisselle", VoiceKind.DISHWASHER),
        ("Lavavajillas", VoiceKind.DISHWASHER),
        ("Lavastoviglie", VoiceKind.DISHWASHER),
        ("Zmywarka", VoiceKind.DISHWASHER),
        ("Waschmaschine", VoiceKind.DRUM),
        ("Washing machine", VoiceKind.DRUM),
        ("Wasmachine", VoiceKind.DRUM),
        ("Lavadora", VoiceKind.DRUM),
        ("Pralka", VoiceKind.DRUM),
        ("Trockner", VoiceKind.DRUM),
        ("Tumble dryer", VoiceKind.DRUM),
        ("Droger", VoiceKind.DRUM),
        ("Secadora", VoiceKind.DRUM),
        ("Asciugatrice", VoiceKind.DRUM),
        ("Kühlschrank", VoiceKind.DRONE),
        ("Backofen", VoiceKind.DRONE),
        ("Living room lamp", VoiceKind.DRONE),
    ],
)
def test_name_keywords_are_the_last_resort(name: str, expected: VoiceKind) -> None:
    """A power sensor is told apart by its name in several languages, if nothing else helps."""
    role = classify("sensor.x_power", name, "power", "W", "5")
    assert role is not None
    assert role.kind == expected
    assert role.source in {"name", "device class"}


def test_dishwasher_is_not_a_washing_machine() -> None:
    """A dishwasher contains the word wash but has its own voice, not the washer's drums."""
    assert _kind("sensor.dishwasher_power") == (VoiceKind.DISHWASHER, "name")
    assert _kind("sensor.washing_machine_power") == (VoiceKind.DRUM, "name")
    assert _kind("sensor.dishwasher_energy", device_class="energy", unit="kWh") == (
        VoiceKind.DISHWASHER,
        "name",
    )


def test_short_keywords_must_stand_alone() -> None:
    """The word four is a French oven, but not part of fourth."""
    assert _kind("sensor.x", name="Four cuisine") == (VoiceKind.DRONE, "name")
    role = classify("sensor.x", "Fourth floor", "power", "W", "5")
    assert role is not None
    assert role.source == "device class"


def test_integration_hints_beat_the_name() -> None:
    """A translation key tells the appliance whatever the entity is called."""
    meta = EntityMeta(platform="home_connect", translation_key="dishwasher_power_state")
    assert _kind("sensor.kitchen_x", name="Kitchen machine", meta=meta) == (
        VoiceKind.DISHWASHER,
        "integration",
    )
    assert _kind("sensor.x", name="Waschmaschine", meta=EntityMeta(model="Dishwasher SX")) == (
        VoiceKind.DISHWASHER,
        "integration",
    )


def test_hints_only_apply_to_power_sensors() -> None:
    """A light of a dishwasher device stays a light."""
    meta = EntityMeta(model="Dishwasher")
    role = classify("light.dw_interior", "Interior", None, None, "off", meta=meta)
    assert role is not None
    assert role.kind == VoiceKind.CHORD


def test_a_label_wins_over_everything() -> None:
    """The label entropy_fm_dishwasher makes any power sensor a dishwasher, whatever its name."""
    meta = EntityMeta(labels=frozenset({"entropy_fm_dishwasher"}), model="Washer")
    role = classify("sensor.x", "Waschmaschine", "power", "W", "5", meta=meta)
    assert role is not None
    assert (role.kind, role.source, role.role) == (VoiceKind.DISHWASHER, "label", "dishwasher")
    assert role.description == "role: dishwasher (label)"


def test_the_mute_label_and_override_silence_an_entity() -> None:
    """Muted entities make no voice."""
    muted = EntityMeta(labels=frozenset({"entropy_fm_mute"}))
    assert classify("light.hall", "Hall", None, None, "on", meta=muted) is None
    assert classify("light.hall", "Hall", None, None, "on", override="mute") is None


def test_an_unknown_label_is_ignored() -> None:
    """A label that is not a role changes nothing."""
    meta = EntityMeta(labels=frozenset({"entropy_fm_nonsense", "kitchen"}))
    role = classify("light.hall", "Hall", None, None, "on", meta=meta)
    assert role is not None
    assert role.source == "domain"


def test_override_comes_before_the_label() -> None:
    """The settings of the user win over a label in Home Assistant."""
    meta = EntityMeta(labels=frozenset({"entropy_fm_fridge"}))
    role = classify("sensor.x", "x", "power", "W", "5", meta=meta, override="dishwasher")
    assert role is not None
    assert (role.kind, role.source) == (VoiceKind.DISHWASHER, "override")


def test_parse_overrides() -> None:
    """Overrides are lines or comma separated pairs; bad ones are skipped."""
    text = "sensor.a=dishwasher\n Sensor.B = Washer ,sensor.c=nonsense; novalue\nlight.d=mute"
    assert parse_overrides(text) == {
        "sensor.a": "dishwasher",
        "sensor.b": "washer",
        "light.d": "mute",
    }
    assert parse_overrides("") == {}


def test_presence_is_only_classified_when_included() -> None:
    """Persons make no voice unless the user asks for them."""
    assert classify("person.jozef", "Jozef", None, None, "home") is None
    included = classify("person.jozef", "Jozef", None, None, "home", include_presence=True)
    assert included is not None
    assert included.kind == VoiceKind.BLIP
