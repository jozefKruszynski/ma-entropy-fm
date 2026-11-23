"""
Decide which voice an entity gets, in layers that do not depend on the language of its name.

The layers, in the order they are tried:

1. A role the user gave: a Home Assistant label `entropy_fm_<role>` on the entity or its
   device, or a line `entity_id=role` in the settings.
2. Hints from the integration: the platform, the translation key of the entity and the
   manufacturer and model of its device. A small table, best effort.
3. The domain and device class: power, energy, temperature, humidity, motion, door, light, ...
4. Only for power and energy sensors, and only as a last resort, keywords in the name or the
   entity id, in several languages: they tell a dishwasher from a washer from a plain meter.

The module is pure and imports nothing from Music Assistant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .timeline import (
    DIAGNOSTIC_WORDS,
    IGNORED_BINARY_CLASSES,
    IGNORED_DEVICE_CLASSES,
    PLUCK_DEVICE_CLASSES,
    PRESENCE_DOMAINS,
    SUN_ELEVATION_ENTITY,
    SUN_ENTITY,
    VoiceKind,
    is_number,
)

ROLE_SUN = "sun"
ROLE_MEDIA = "media"
ROLE_COVER = "cover"
ROLE_EVENT = "event"
ROLE_PLUCK = "pluck"
ROLE_BLIP = "blip"
ROLE_CHORD = "chord"
ROLE_HOB = "hob"
ROLE_PROGRESS = "progress"
ROLE_DRONE = "drone"
ROLE_TEMPERATURE = "temperature"
ROLE_HUMIDITY = "humidity"
ROLE_CO2 = "co2"
ROLE_PAD = "pad"
ROLE_MUTE = "mute"
SOURCE_LABEL = "label"
SOURCE_OVERRIDE = "override"
SOURCE_INTEGRATION = "integration"
SOURCE_DEVICE_CLASS = "device class"
SOURCE_NAME = "name"
SOURCE_DOMAIN = "domain"
# the domains that the decision table tells apart
DOMAIN_SENSOR = "sensor"
DOMAIN_BINARY_SENSOR = "binary_sensor"
DOMAIN_MEDIA_PLAYER = "media_player"
DOMAIN_COVER = "cover"
SELECT_DOMAINS = frozenset({"select", "input_select"})
LIGHT_DOMAINS = frozenset({"light", "switch"})
BLIP_DOMAINS = frozenset({"input_boolean", "fan"})
DEVICE_CLASS_POWER = "power"
DEVICE_CLASS_ENERGY = "energy"
DEVICE_CLASS_ENUM = "enum"
DEVICE_CLASS_PRESENCE = "presence"
DEVICE_CLASS_TEMPERATURE = "temperature"
DEVICE_CLASS_HUMIDITY = "humidity"
DEVICE_CLASS_CO2 = "carbon_dioxide"
PERCENT_UNIT = "%"
PPM_UNIT = "ppm"
HOB_LEVEL_WORD = "power_level"
CO2_WORD = "co2"
# a state that says nothing, and the longest text that is the word of a state machine
EMPTY_STATES = frozenset({"unknown", "unavailable", "none", ""})
MAX_STATE_TEXT = 40
# a keyword up to this long must stand alone as a word
MAX_STANDALONE_WORD = 5
LABEL_PREFIX = "entropy_fm_"
# role name to what it plays as; a role with None is muted
ROLES: dict[str, VoiceKind | None] = {
    "dishwasher": VoiceKind.DISHWASHER,
    "washer": VoiceKind.DRUM,
    "washing_machine": VoiceKind.DRUM,
    "dryer": VoiceKind.DRUM,
    "drum": VoiceKind.DRUM,
    ROLE_DRONE: VoiceKind.DRONE,
    "fridge": VoiceKind.DRONE,
    "oven": VoiceKind.DRONE,
    ROLE_TEMPERATURE: VoiceKind.TEMPERATURE,
    ROLE_HUMIDITY: VoiceKind.HUMIDITY,
    ROLE_CO2: VoiceKind.AIR,
    "air": VoiceKind.AIR,
    ROLE_PLUCK: VoiceKind.PLUCK,
    ROLE_CHORD: VoiceKind.CHORD,
    ROLE_BLIP: VoiceKind.BLIP,
    ROLE_PAD: VoiceKind.PAD_PARTIAL,
    ROLE_EVENT: VoiceKind.PHASE,
    ROLE_HOB: VoiceKind.HOB,
    ROLE_MEDIA: VoiceKind.MEDIA,
    ROLE_COVER: VoiceKind.COVER,
    ROLE_PROGRESS: VoiceKind.PROGRESS,
    ROLE_SUN: VoiceKind.SUN,
    ROLE_MUTE: None,
}
# translation keys, platforms and models that name an appliance; checked as substrings of the
# lower case text, which these integrations write in English
INTEGRATION_HINTS: tuple[tuple[str, str], ...] = (
    ("dishwasher", "dishwasher"),
    ("dish_washer", "dishwasher"),
    ("washing_machine", "washer"),
    ("washer", "washer"),
    ("laundry", "washer"),
    ("tumble_dryer", "dryer"),
    ("dryer", "dryer"),
    ("refrigerator", "fridge"),
    ("freezer", "fridge"),
    ("oven", "oven"),
)
# last resort keywords per role, lower case
NAME_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "dishwasher",
        (
            "dishwasher",
            "dish washer",
            "dish_washer",
            "geschirrspül",
            "geschirrspul",
            "geschirrspuel",
            "spülmaschine",
            "spulmaschine",
            "spuelmaschine",
            "vaatwasser",
            "lave-vaisselle",
            "lave vaisselle",
            "lavavajillas",
            "lavastoviglie",
            "zmywarka",
        ),
    ),
    (
        "dryer",
        (
            "dryer",
            "tumble",
            "trockner",
            "wäschetrockner",
            "droger",
            "sèche-linge",
            "seche-linge",
            "seche_linge",
            "secadora",
            "asciugatrice",
            "suszarka",
        ),
    ),
    (
        "washer",
        (
            "washing",
            "washer",
            "waschmaschine",
            "wasmachine",
            "lave-linge",
            "lave_linge",
            "machine à laver",
            "lavadora",
            "lavatrice",
            "pralka",
        ),
    ),
    ("fridge", ("fridge", "refrigerator", "kühlschrank", "koelkast", "frigo", "nevera", "lodówka")),
    ("oven", ("oven", "backofen", "herd", "four", "horno", "forno", "piekarnik")),
)
# entity categories that never play
SKIPPED_CATEGORIES = frozenset({"diagnostic", "config"})
STATE_WORDS = ("operation_state", "program_phase", "phase", "_state", "program", "mode")
HOB_WORDS = ("hob", "cooktop", "induction")
PROGRESS_WORDS = ("program_progress", "progress")
POWER_UNITS = frozenset({"w", "kw"})
ENERGY_UNITS = frozenset({"wh", "kwh"})
TEMPERATURE_UNITS = frozenset({"°c", "°f", "c", "f"})
WORD = re.compile(r"[^\W_]+|_", re.UNICODE)


@dataclass(frozen=True)
class EntityMeta:
    """What the registries say about an entity, besides its state."""

    platform: str = ""
    translation_key: str = ""
    manufacturer: str = ""
    model: str = ""
    # the names of the labels on the entity and on its device, lower case
    labels: frozenset[str] = field(default_factory=frozenset)
    # "diagnostic" or "config" for entities that are about the device, not what it does
    entity_category: str = ""
    device_id: str = ""


@dataclass(frozen=True)
class Role:
    """The voice of an entity, and why it has it."""

    kind: VoiceKind
    role: str
    # label, override, integration, device class or name
    source: str
    cumulative: bool = False

    @property
    def description(self) -> str:
        """Return a short text such as "role: dishwasher (label)"."""
        return f"role: {self.role} ({self.source})"


def parse_overrides(text: str) -> dict[str, str]:
    """
    Read role overrides from lines such as `sensor.geschirrspueler_power=dishwasher`.

    :param text: Lines, or comma separated pairs; lines without `=` and unknown roles are skipped.
    """
    found: dict[str, str] = {}
    for part in re.split(r"[\n,;]+", text or ""):
        entity_id, _, role = part.partition("=")
        entity_id, role = entity_id.strip().lower(), role.strip().lower()
        if entity_id and role in ROLES:
            found[entity_id] = role
    return found


def classify(  # noqa: PLR0911 - a flat decision table
    entity_id: str,
    name: str,
    device_class: str | None,
    unit: str | None,
    state: str | None,
    *,
    meta: EntityMeta | None = None,
    override: str | None = None,
    include_presence: bool = False,
) -> Role | None:
    """
    Decide which voice an entity gets, or None when it makes no sound.

    :param entity_id: The entity id.
    :param name: The friendly name.
    :param device_class: The device class from the attributes.
    :param unit: The unit of measurement.
    :param state: The current state, which tells a numeric sensor from a text one.
    :param meta: What the registries say about the entity.
    :param override: A role from the settings of the user.
    :param include_presence: Whether persons and device trackers play.
    :return: The role, or None for no voice.
    """
    meta = meta or EntityMeta()
    domain = entity_id.split(".", 1)[0]
    dclass = (device_class or "").lower()
    unit_l = (unit or "").lower()
    is_power = domain == DOMAIN_SENSOR and (dclass == DEVICE_CLASS_POWER or unit_l in POWER_UNITS)
    is_energy = domain == DOMAIN_SENSOR and (
        dclass == DEVICE_CLASS_ENERGY or unit_l in ENERGY_UNITS
    )
    cumulative = is_energy and not is_power

    # 1. the user
    if override and override in ROLES:
        return _from_role(override, SOURCE_OVERRIDE, cumulative=cumulative)
    if (label_role := _role_from_labels(meta.labels)) is not None:
        return _from_role(label_role, SOURCE_LABEL, cumulative=cumulative)

    # 2. the integration, for the sensors that measure an appliance
    if (is_power or is_energy) and (hint := _role_from_integration(meta)) is not None:
        return _from_role(hint, SOURCE_INTEGRATION, cumulative=cumulative)

    # 3. domain and device class
    if entity_id in (SUN_ENTITY, SUN_ELEVATION_ENTITY):
        return Role(VoiceKind.SUN, ROLE_SUN, SOURCE_DEVICE_CLASS)
    if meta.entity_category in SKIPPED_CATEGORIES or _is_diagnostic(entity_id, dclass, meta):
        return None
    if domain == DOMAIN_MEDIA_PLAYER:
        return Role(VoiceKind.MEDIA, ROLE_MEDIA, SOURCE_DOMAIN)
    if domain == DOMAIN_COVER:
        return Role(VoiceKind.COVER, ROLE_COVER, SOURCE_DOMAIN)
    if domain in SELECT_DOMAINS:
        return Role(VoiceKind.PHASE, ROLE_EVENT, SOURCE_DOMAIN)
    if domain == DOMAIN_SENSOR:
        return _classify_sensor(
            entity_id,
            name,
            dclass,
            unit_l,
            state,
            meta,
            is_power=is_power,
            is_energy=is_energy,
        )
    if domain == DOMAIN_BINARY_SENSOR:
        if dclass in PLUCK_DEVICE_CLASSES:
            return Role(VoiceKind.PLUCK, ROLE_PLUCK, SOURCE_DEVICE_CLASS)
        if dclass in IGNORED_BINARY_CLASSES or (
            dclass == DEVICE_CLASS_PRESENCE and not include_presence
        ):
            return None
        return Role(VoiceKind.BLIP, ROLE_BLIP, SOURCE_DOMAIN)
    if domain in LIGHT_DOMAINS:
        return Role(VoiceKind.CHORD, ROLE_CHORD, SOURCE_DOMAIN)
    if domain in BLIP_DOMAINS:
        return Role(VoiceKind.BLIP, ROLE_BLIP, SOURCE_DOMAIN)
    if domain in PRESENCE_DOMAINS and include_presence:
        return Role(VoiceKind.BLIP, ROLE_BLIP, SOURCE_DOMAIN)
    return None


def _role_from_labels(labels: frozenset[str]) -> str | None:
    """Return the role named by an `entropy_fm_<role>` label, if there is a known one."""
    for label in sorted(labels):
        if label.startswith(LABEL_PREFIX) and (role := label.removeprefix(LABEL_PREFIX)) in ROLES:
            return role
    return None


def _role_from_integration(meta: EntityMeta) -> str | None:
    """Return a role the integration data names."""
    text = f"{meta.translation_key} {meta.platform} {meta.model}".lower()
    text = text.replace(" ", "_")
    for hint, role in INTEGRATION_HINTS:
        if hint in text:
            return role
    return None


def _role_from_name(text: str) -> str | None:
    """Return a role named by keywords, the last resort."""
    lowered = text.lower()
    for role, words in NAME_KEYWORDS:
        for word in words:
            # a short word must stand alone ("four", "herd"), a long one may sit inside
            if (len(word) <= MAX_STANDALONE_WORD and word in WORD.findall(lowered)) or (
                len(word) > MAX_STANDALONE_WORD and word in lowered
            ):
                return role
    return None


def _is_diagnostic(entity_id: str, dclass: str, meta: EntityMeta) -> bool:
    """Return True for an entity about the device itself: signal, firmware, voltage, versions."""
    text = f"{dclass} {meta.translation_key} {entity_id.split('.', 1)[1]}".lower()
    return any(word in text for word in DIAGNOSTIC_WORDS)


def _is_hob_level(text: str) -> bool:
    """Return True for the power level of one zone of a hob."""
    return any(word in text for word in HOB_WORDS) and HOB_LEVEL_WORD in text


def _is_state_text(state: str) -> bool:
    """Return True for a state that is a word of a state machine, not 'unknown' or a time."""
    lowered = state.lower()
    return lowered not in EMPTY_STATES and len(lowered) < MAX_STATE_TEXT and ":" not in lowered


def _from_role(role: str, source: str, *, cumulative: bool) -> Role | None:
    """Make a Role from a named role; muted roles give None."""
    kind = ROLES[role]
    if kind is None:
        return None
    return Role(kind, role, source, cumulative)


def _classify_sensor(
    entity_id: str,
    name: str,
    dclass: str,
    unit_l: str,
    state: str | None,
    meta: EntityMeta,
    *,
    is_power: bool,
    is_energy: bool,
) -> Role | None:
    """Decide the voice of a sensor, or None when it makes no sound."""
    if dclass in IGNORED_DEVICE_CLASSES - {DEVICE_CLASS_ENUM}:
        return None
    text_key = f"{entity_id} {meta.translation_key}".lower()
    if _is_hob_level(text_key):
        return Role(VoiceKind.HOB, ROLE_HOB, SOURCE_INTEGRATION if meta.platform else SOURCE_NAME)
    if unit_l == PERCENT_UNIT and any(word in text_key for word in PROGRESS_WORDS):
        return Role(
            VoiceKind.PROGRESS, ROLE_PROGRESS, SOURCE_INTEGRATION if meta.platform else SOURCE_NAME
        )
    if dclass == DEVICE_CLASS_ENUM or (
        state is not None
        and not is_number(state)
        and _is_state_text(state)
        and any(word in text_key for word in STATE_WORDS)
    ):
        return Role(VoiceKind.PHASE, ROLE_EVENT, SOURCE_DEVICE_CLASS)
    if is_power or is_energy:
        return _classify_power(entity_id, name, cumulative=is_energy and not is_power)
    return _classify_measurement(dclass, unit_l, text_key, state)


def _classify_power(entity_id: str, name: str, *, cumulative: bool) -> Role | None:
    """Decide the voice of a power or energy sensor."""
    # 4. the last resort tells which appliance it is
    if (word_role := _role_from_name(f"{entity_id} {name}")) is not None:
        return _from_role(word_role, SOURCE_NAME, cumulative=cumulative)
    return Role(VoiceKind.DRONE, ROLE_DRONE, SOURCE_DEVICE_CLASS, cumulative)


def _classify_measurement(
    dclass: str, unit_l: str, text_key: str, state: str | None
) -> Role | None:
    """Decide the voice of a sensor that is neither an appliance nor a state machine."""
    if dclass == DEVICE_CLASS_TEMPERATURE or unit_l in TEMPERATURE_UNITS:
        return Role(VoiceKind.TEMPERATURE, ROLE_TEMPERATURE, SOURCE_DEVICE_CLASS)
    if dclass == DEVICE_CLASS_HUMIDITY:
        return Role(VoiceKind.HUMIDITY, ROLE_HUMIDITY, SOURCE_DEVICE_CLASS)
    if dclass == DEVICE_CLASS_CO2 or (unit_l == PPM_UNIT and CO2_WORD in text_key.replace("_", "")):
        return Role(VoiceKind.AIR, ROLE_CO2, SOURCE_DEVICE_CLASS)
    if is_number(state):
        return Role(VoiceKind.PAD_PARTIAL, ROLE_PAD, SOURCE_DOMAIN)
    return None
